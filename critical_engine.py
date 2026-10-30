"""Interface-independent analysis of reversible positive-field isotherms.

H is an input alias for mu0*H_applied, in tesla after conversion; it is not B
inside the sample. Cutoff selection does not perform a demagnetising correction.
Loss-based samples are sensitivity distributions, not calibrated posteriors.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate, stats
from scipy.optimize import minimize, curve_fit
from scipy.integrate import cumulative_trapezoid

VERSION = '0.2.0'
BASE_COMMIT = '28b612a0b18270ae4314ae6c46cd853ae20b7e4e'
BOUNDS = ((0.05, 1.2), (0.2, 3.5))
REFERENCES = {
    'Mean-field': (0.5, 1.0, 'Landau mean field'),
    'Tricritical mean-field': (0.25, 1.0, 'Landau tricritical mean field'),
    '3D Ising': (0.326, 1.237, 'Guida and Zinn-Justin 1997'),
    '3D Heisenberg': (0.3689, 1.396, 'Campostrini et al. 2002'),
    '3D XY': (0.3486, 1.3178, 'Campostrini et al. 2006, PRB 74 144506'),
    '2D Ising': (0.125, 1.75, 'exact zero-field 2D Ising'),
}
FIELD_FACTORS = {'T': 1., 'mT': 1e-3, 'Oe': 1e-4, 'kOe': 0.1, 'A/m': 4e-7*np.pi,
                 'relative': 1.}
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AnalysisError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                   indent=2, allow_nan=False), encoding='utf-8')


def _aggregate(x, y):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    u, inv = np.unique(x, return_inverse=True)
    return u, np.bincount(inv, weights=y)/np.bincount(inv)


def _source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


LOADED_SOURCE_HASHES = _source_hashes()


class PhysicsEngine:
    def __init__(self):
        self.data_files = []
        self.results = {}
        self.hc_value = 0.
        self.config = {}
        self.mce_data = None
        self._gp_cache = {}
        self.universality_exponents = {
            k: dict(beta=v[0], gamma=v[1], source=v[2]) for k, v in REFERENCES.items()}

    def load_files(self, filepaths, *, field_unit=None, magnetization_unit=None):
        """Parse explicit column units or caller units; never infer units from size.

        Two-column files require temperature in the filename (e.g. sample_100K.csv).
        Three-column tables require named T/H/M columns. Partial loads are recorded.
        """
        self.data_files = []
        self.load_errors = []
        for path in filepaths:
            try:
                self.data_files.extend(self._load_single_file(
                    path, field_unit=field_unit, magnetization_unit=magnetization_unit))
            except (ValueError, OSError) as exc:
                self.load_errors.append(dict(path=str(path), reason=str(exc)))
        self.data_files.sort(key=lambda d: d['T'])
        return len(self.data_files)

    def _load_single_file(self, path, *, field_unit=None, magnetization_unit=None):
        path = Path(path)
        lines = [s.strip().lstrip('#').strip() for s in path.read_text(
            encoding='utf-8-sig').splitlines() if s.strip() and not s.strip().startswith('//')]
        # Preserve useful commented headers; discard comment prose.
        numeric = re.compile(r'^[\s,;\t+\-\d.eEdD]+$')
        start = next((i for i,s in enumerate(lines) if numeric.fullmatch(s)), None)
        if start is None:
            raise ValueError('No numeric observations')
        first = lines[start]
        sep = ',' if ',' in first else (';' if ';' in first else r'\s+')
        header = lines[start-1] if start and re.search(r'\b(H|B|M|Field|Temperature|T)\b', lines[start-1], re.I) else None
        from io import StringIO
        raw = '\n'.join(([header] if header else [])+lines[start:])
        raw = re.sub(r'(?<=\d)[dD](?=[+\-]?\d)', 'E', raw)
        df = pd.read_csv(StringIO(raw), sep=sep, header=0 if header else None, engine='python')
        names = [str(c) for c in df.columns]
        def pick(expr):
            return next((c for c in names if re.search(expr,c,re.I)), None)
        tcol = pick(r'^(T|Temp|Temperature)(\b|_)') if header else None
        hcol = pick(r'^(H|B|Field|mu0H)(\b|_)') if header else names[0]
        mcol = pick(r'^(M|Magnetization|Magnetisation|Moment)(\b|_)') if header else names[1]
        if hcol is None or mcol is None or hcol == mcol:
            raise ValueError('Expected H and M columns; specify T/H/M headers for long tables')
        df.columns = names
        unit_in_header = None
        for u in ['kOe', 'Oe', 'mT', 'A/m', 'T']:
            if re.search(r'(?:\(|\[|_|\s)'+re.escape(u)+r'(?:\)|\]|$)', hcol):
                unit_in_header = u
                break
        if field_unit and unit_in_header and field_unit != unit_in_header:
            raise ValueError('Field unit argument conflicts with header')
        unit = field_unit or unit_in_header
        if unit not in FIELD_FACTORS:
            raise ValueError('Unknown field unit: supply T, mT, Oe, kOe or A/m explicitly')
        h = pd.to_numeric(df[hcol], errors='coerce').to_numpy()*FIELD_FACTORS[unit]
        m = pd.to_numeric(df[mcol], errors='coerce').to_numpy()
        if tcol:
            temperatures = pd.to_numeric(df[tcol], errors='coerce').to_numpy()
        else:
            match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*K', path.stem, re.I)
            if not match:
                raise ValueError('Temperature missing: use T column or a filename ending in 100K')
            temperatures = np.full(len(h), float(match.group(1)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out=[]
        for t in np.unique(temperatures[np.isfinite(temperatures)]):
            sel = temperatures == t
            out.append(dict(file=path.name, path=str(path.resolve()), T=float(t),
                            H=h[sel], M=m[sel], field_unit='relative' if unit=='relative' else 'T', input_field_unit=unit,
                            magnetization_unit=magnetization_unit or 'unspecified', sha256=digest))
        return out

    def _get_filtered_data(self, d, *, cutoff=True, include_zero=False):
        h, m = np.asarray(d['H'],float), np.asarray(d['M'],float)
        good = np.isfinite(h)&np.isfinite(m)
        h,m = h[good],m[good]
        branch = self.config.get('branch','auto')
        # Detect reversals in acquisition order before sorting or aggregating.
        dh = np.diff(h)
        tolerance = 1e-12*max(np.ptp(h) if h.size else 0., 1e-100)
        significant = dh[np.abs(dh)>tolerance]
        if self.config.get('ordered_sweep',True) and significant.size and np.any(significant>0) and np.any(significant<0):
            if branch == 'auto':
                raise AnalysisError('sweep_reversal', 'Multiple sweep paths: select ascending or descending explicitly')
            boundaries = np.r_[0, np.where(np.sign(significant[1:]) != np.sign(significant[:-1]))[0]+1]
            # Segment original indices, including the turning point in both segments.
            sign = np.sign(dh)
            for i in range(1,len(sign)):
                if sign[i]==0: sign[i]=sign[i-1]
            turns = np.r_[0, np.where(sign[1:]*sign[:-1]<0)[0]+1, len(h)-1]
            want = 1 if branch=='ascending' else -1
            segments = [(a,b) for a,b in zip(turns[:-1],turns[1:]) if np.sign(h[b]-h[a])==want]
            if len(segments)!=1:
                raise AnalysisError('ambiguous_sweep','Provide one sweep cycle per file')
            a,b = segments[0]; h,m=h[a:b+1],m[a:b+1]
        good = (h>=0) if include_zero else (h>0)&(m>0)
        if cutoff: good &= h>=self.hc_value
        h,m=h[good],m[good]
        return _aggregate(h,m)

    def prepare(self):
        if not self.config.get('reversible',False):
            raise AnalysisError('equilibrium_unconfirmed','Confirm reversible equilibrium measurements before critical analysis')
        groups={}
        records=[]
        for d in self.data_files:
            if not np.isfinite(d['T']) or d['T']<=0:
                raise AnalysisError('invalid_temperature','Temperatures must be positive kelvin')
            h,m=self._get_filtered_data(d,cutoff=False,include_zero=True)
            npositive=int(np.sum((h>0)&(m>0)))
            if npositive<6:
                raise AnalysisError('too_few_points',f"{d['file']}: fewer than six positive points")
            records.append(dict(file=d['file'],T=d['T'],raw_points=len(d['H']),positive_unique_points=npositive))
            groups.setdefault(d['T'],[]).append(dict(d,H=h,M=m))
        merged=[]
        for t,curves in sorted(groups.items()):
            # Repeated temperatures are measurement replicates, never independent donors.
            if len(curves)>1:
                lo=max(c['H'][0] for c in curves); hi=min(c['H'][-1] for c in curves)
                if lo>=hi: raise AnalysisError('replicate_no_overlap','Repeated-temperature curves have no common fields')
                grid=np.unique(np.concatenate([c['H'][(c['H']>=lo)&(c['H']<=hi)] for c in curves]))
                magnet=np.mean([np.interp(grid,c['H'],c['M']) for c in curves],axis=0)
                merged.append(dict(curves[0],H=grid,M=magnet,replicate_count=len(curves)))
            else: merged.append(curves[0])
        self.data_files=merged
        if len(merged)<4: raise AnalysisError('too_few_temperatures','At least four distinct temperatures required for two branches')
        self.h0=max(np.max(d['H']) for d in merged)
        self.field_representation='relative' if any(d.get('field_unit')=='relative' for d in merged) else 'T'
        self.m0=np.median(np.concatenate([d['M'][d['M']>0] for d in merged]))
        self.results['preprocessing']=records
        self.results['reference_scales']=dict(H0_value=self.h0,H0_unit=self.field_representation,M0=self.m0)

    def _fit_modified_arrott_isotherm(self,H,M,*,beta,gamma,min_pts=6):
        """Dimensionless alternating-point validation, with extrapolation penalty.

        Selection score = CV MSE / range(y)^2 + 1e-4(1+e^2)/n.
        e is the distance from x=0 to the observed window in units of range(x).
        The penalty is a disclosed heuristic; all candidate windows are retained.
        """
        H,M=np.asarray(H),np.asarray(M)
        if len(H)<min_pts: return None
        h0=getattr(self,'h0',max(H)); m0=getattr(self,'m0',np.median(M))
        x=(H/h0/(M/m0))**(1/gamma); y=(M/m0)**(1/beta)
        yscale=max(float(np.ptp(y)),1e-14)
        candidates=[]
        for frac in self.config.get('window_fractions',[.4,.55,.7,.85,1.]):
            n=max(min_pts,int(np.ceil(frac*len(H))))
            xx,yy=x[-n:],y[-n:]
            if np.ptp(xx)<=1e-14: continue
            fit=stats.linregress(xx,yy)
            if fit.slope<=0 or not np.isfinite(fit.slope): continue
            prediction=[]
            for phase in [0,1]:
                train=np.arange(n)%2==phase
                reg=stats.linregress(xx[train],yy[train])
                prediction.extend((yy[~train]-reg.intercept-reg.slope*xx[~train])**2)
            e=float(np.min(xx)/np.ptp(xx))
            mse=float(np.mean(prediction))/yscale**2
            residual=yy-fit.intercept-fit.slope*xx
            candidates.append(dict(slope=fit.slope,intercept=fit.intercept,r2=fit.rvalue**2,
                n=n,hcut=float(H[-n]),score=mse+1e-4*(1+e*e)/n,cv_mse=mse,
                extrapolation=e,intercept_se=fit.intercept_stderr,
                residual_rmse=float(np.sqrt(np.mean(residual**2))),residuals=residual.tolist()))
        if not candidates: return None
        best=dict(min(candidates,key=lambda c:c['score']))
        best['candidates']=candidates
        best['intercept_range']=float(np.ptp([c['intercept'] for c in candidates]))
        return best

    def perform_kf_and_correction(self,use_correction=False,*,omega=.5,fit_omega=True,
                                  override_exponents=False,update_results=True):
        b,g=self.results['beta'],self.results['gamma']
        rows=[]
        for d in self.data_files:
            h,m=self._get_filtered_data(d)
            f=self._fit_modified_arrott_isotherm(h,m,beta=b,gamma=g)
            if f is None: continue
            a,c=f['slope'],f['intercept']
            rows.append(dict(T=d['T'],Ms=c**b if c>0 else np.nan,
                ChiInv=(-c/a)**g if c<0 else np.nan,MAP_intercept=c,MAP_slope=a,
                MAP_r2=f['r2'],fit=f))
        t=np.array([r['T'] for r in rows])
        ms=np.array([r['Ms'] for r in rows]); chi=np.array([r['ChiInv'] for r in rows])
        fits={}; lines={}
        for key,values in [('Ms',ms),('ChiInv',chi)]:
            sel=np.isfinite(values)&(values>0)
            sel &= t<self.results['Tc'] if key=='Ms' else t>self.results['Tc']
            tt,yy=t[sel],values[sel]
            fit=None; full=np.full(len(t),np.nan)
            if len(tt)>=4:
                # Smooth log(y) against log distance to the CURRENT Tc seed. A
                # leading power law is linear here, avoiding endpoint derivative
                # bias of a cubic interpolant of sqrt(Tc-T). Seed dependence is
                # explicit and the whole step is repeated in the outer bootstrap.
                distance=tt-self.results['Tc']; lx=np.log(np.abs(distance)); ly=np.log(yy)
                order=np.argsort(lx)
                smooth=float(self.config.get('kf_smoothing',.001))*len(tt)*np.var(ly)
                spline=interpolate.UnivariateSpline(lx[order],ly[order],s=smooth,k=min(3,len(tt)-1))
                derivative=yy*spline.derivative()(lx)/distance
                valid=np.isfinite(derivative)&(np.abs(derivative)>1e-12)
                kf=np.full(len(tt),np.nan); kf[valid]=yy[valid]/derivative[valid]
                full[sel]=kf
                if valid.sum()>=4:
                    r=stats.linregress(tt[valid],kf[valid])
                    tc=-r.intercept/r.slope if r.slope!=0 else np.nan
                    # No class-specific exponent gate. Numerical/physical domain only.
                    exponent=1/r.slope if r.slope>0 else np.nan
                    if np.isfinite(exponent) and t.min()<tc<t.max():
                        fit=dict(slope=r.slope,intercept=r.intercept,Tc=tc,r2=r.rvalue**2,
                                 n=int(valid.sum()),exponent=exponent,
                                 slope_se=r.stderr, uncertainty='KF regression diagnostic only')
            fits[key]=fit; lines[key]=full
        valid=[f for f in fits.values() if f]
        tc_kf=float(np.mean([f['Tc'] for f in valid])) if valid else None
        crossing=None
        for r1,r2 in zip(rows[:-1],rows[1:]):
            c1,c2=r1['MAP_intercept'],r2['MAP_intercept']
            if c1*c2<=0 and c1!=c2:
                crossing=r1['T']-c1*(r2['T']-r1['T'])/(c2-c1); break
        if update_results:
            self.results.update(Tc_KF=tc_kf,Tc_Arrott=crossing,kf_map=rows,kf_arrays=(t,ms,chi),
                kf_fits=fits,kf_lines=lines,beta_KF=fits['Ms']['exponent'] if fits['Ms'] else None,
                gamma_KF=fits['ChiInv']['exponent'] if fits['ChiInv'] else None,
                kf_sides=len(valid))
            self.results['kf_side_difference_K']=abs(valid[0]['Tc']-valid[1]['Tc']) if len(valid)==2 else None
        if use_correction:
            self._correction_fit(t,ms,chi,omega=omega,fit_omega=fit_omega)
        return rows,(t,ms,chi)

    def _correction_fit(self,t,ms,chi,*,omega,fit_omega):
        output={}; tc=self.results['Tc']
        for name,y,below,start in [('Ms',ms,True,self.results['beta']),('ChiInv',chi,False,self.results['gamma'])]:
            sel=np.isfinite(y)&(y>0)&((t<tc) if below else (t>tc))
            u=np.abs((t[sel]-tc)/tc); v=y[sel]
            if len(v)<7:
                output[name]=dict(success=False,reason='fewer_than_seven_temperatures'); continue
            def model(u,amplitude,exponent,correction,Delta):
                return amplitude*u**exponent*(1+correction*u**Delta)
            fun=model if fit_omega else lambda u,A,e,a:model(u,A,e,a,omega)
            initial=[max(v),start,.01]+([omega] if fit_omega else [])
            bounds=([0,.05,-5]+([.1] if fit_omega else []),[np.inf,3.5,5]+([2.] if fit_omega else []))
            try:
                p,cov=curve_fit(fun,u,v,p0=initial,bounds=bounds,maxfev=20000)
                output[name]=dict(success=True,amplitude=p[0],exponent=p[1],correction=p[2],
                    Delta=p[3] if fit_omega else omega,Delta_fixed=not fit_omega,covariance=cov)
            except (ValueError,RuntimeError) as exc:
                output[name]=dict(success=False,reason=str(exc))
        self.results['correction_fit']=output
        return output

    def _scaled_curves(self,beta,gamma,tc,sample_per_curve=100):
        curves=[]
        if not np.isfinite(tc) or tc<=0: return curves
        h0=getattr(self,'h0',max(np.max(d['H']) for d in self.data_files))
        m0=getattr(self,'m0',np.median(np.concatenate([d['M'] for d in self.data_files])))
        grid=np.unique([d['T'] for d in self.data_files])
        exclusion=float(self.config.get('critical_exclusion_K',.1*np.min(np.diff(grid)))) if len(grid)>1 else 0.
        for d in self.data_files:
            eps=(d['T']-tc)/tc
            if abs(d['T']-tc)<=exclusion or abs(eps)<1e-8: continue
            h,m=self._get_filtered_data(d)
            if len(h)<6: continue
            if sample_per_curve and len(h)>sample_per_curve:
                idx=np.linspace(0,len(h)-1,sample_per_curve).astype(int); h,m=h[idx],m[idx]
            x=np.log(h/h0)-(beta+gamma)*np.log(abs(eps))
            y=np.log(m/m0)-beta*np.log(abs(eps))
            curves.append(dict(T=d['T'],side=1 if eps>0 else -1,x=x,y=y))
        return curves

    def collapse_diagnostics(self,beta,gamma,tc,*,sample_per_curve=100):
        if not all(lo<=v<=hi for v,(lo,hi) in zip([beta,gamma],BOUNDS)):
            return dict(cost=np.inf,reason='parameter_bounds')
        curves=self._scaled_curves(beta,gamma,tc,sample_per_curve)
        for c in curves:
            c['predictor']=interpolate.PchipInterpolator(c['x'],c['y'],extrapolate=False)
        by_curve=[]; support=[]
        for target in curves:
            # Prediction uses ONLY other distinct temperatures on the same side.
            predictions=[]
            for donor in curves:
                if donor['T']==target['T'] or donor['side']!=target['side']: continue
                p=donor['predictor'](target['x'])
                predictions.append(p)
            if not predictions:
                return dict(cost=np.inf,reason='fewer_than_two_isotherms_per_side')
            pred=np.array(predictions)
            count=np.isfinite(pred).sum(axis=0)
            valid=count>0
            coverage=float(np.mean(valid))
            support.append(dict(T=target['T'],side=target['side'],coverage=coverage,
                                n_predicted=int(valid.sum()),n_points=len(valid)))
            if valid.sum()<6 or coverage<float(self.config.get('minimum_overlap',.35)):
                return dict(cost=np.inf,reason='insufficient_common_support',support=support)
            mean=np.nansum(pred[:,valid],axis=0)/count[valid]
            residual=target['y'][valid]-mean
            by_curve.append(float(np.mean(residual**2)))
        sides={c['side'] for c in curves}
        if len(curves)<4 or sides!={-1,1}:
            return dict(cost=np.inf,reason='missing_temperature_branch')
        return dict(cost=float(np.mean(by_curve)),reason=None,support=support,
                    n_curves=len(curves),curve_costs=by_curve,
                    definition='equal-isotherm LOIO squared log prediction error')

    def _collapse_cost(self,beta,gamma,tc,*,sample_per_curve=100,xbins=None,use_logx=True):
        # xbins retained only for compatibility: this objective does not bin data.
        return self.collapse_diagnostics(beta,gamma,tc,sample_per_curve=sample_per_curve)['cost']

    def optimize_data_collapse(self,init_beta,init_gamma,fixed_params=False,init_tc=None,*,
                               tc_fixed=None,tc_prior=None,update_results=True):
        temps=np.array([d['T'] for d in self.data_files]); tmin,tmax=temps.min(),temps.max()
        fixed=tc_fixed is not None
        if fixed and not tmin<tc_fixed<tmax:
            raise AnalysisError('tc_out_of_range','Fixed Tc must lie inside the observed temperature range')
        bounds=list(BOUNDS)+([] if fixed else [(tmin+1e-6,tmax-1e-6)])
        def obj(p):
            b,g=p[:2]; tc=tc_fixed if fixed else p[2]
            if any(v<lo or v>hi for v,(lo,hi) in zip(p,bounds)): return 1e6
            cost=self._collapse_cost(b,g,tc)
            if not np.isfinite(cost): return 1e6
            if tc_prior is not None and not fixed: cost+=((tc-tc_prior[0])/tc_prior[1])**2
            return cost
        tc0=init_tc if init_tc is not None else np.mean(temps)
        seeds=[(init_beta,init_gamma)]
        if self.config.get('multistart',True) and not fixed_params:
            seeds += [(v[0],v[1]) for v in REFERENCES.values()]
            seeds += [(.23,2.2)]  # numerical seed; not an asserted XY fixed point
        candidates=[]
        for b,g in seeds:
            p=np.array([b,g]+([] if fixed else [tc0]),float)
            if fixed_params:
                if not fixed: raise ValueError('fixed_params requires an explicit fixed Tc')
                candidates.append(dict(x=p,fun=obj(p),success=obj(p)<1e6,message='fixed exponents'))
            else:
                opt=minimize(obj,p,method='Nelder-Mead',bounds=bounds,
                    options=dict(maxiter=int(self.config.get('optimizer_maxiter',400)),xatol=1e-5,fatol=1e-10))
                candidates.append(dict(x=opt.x,fun=opt.fun,success=bool(opt.success),message=str(opt.message)))
        best=min(candidates,key=lambda r:r['fun']); b,g=best['x'][:2]
        tc=float(tc_fixed if fixed else best['x'][2])
        diagnostic=self.collapse_diagnostics(float(b),float(g),tc)
        boundary=any(abs(v-lo)<1e-4 or abs(v-hi)<1e-4 for v,(lo,hi) in zip(best['x'],bounds))
        out=dict(beta=float(b),gamma=float(g),Tc=tc,cost=diagnostic['cost'],
            success=best['success'] and np.isfinite(diagnostic['cost']),tc_fixed=fixed,
            boundary_hit=boundary,diagnostics=diagnostic,multistart=candidates,
            optimizer='bounded multistart Nelder-Mead; no global-optimum guarantee')
        if update_results:
            self.results.update(beta=float(b),gamma=float(g),Tc=tc,delta=1+g/b,
                collapse_cost=out['cost'],collapse_fit=out)
        return out

    def auto_suggest_bc(self):
        """Dimensionless CV diagnostic, not an intercept-to-zero objective.

        Score = median(window selection score) + .01 * fraction excluded.
        Coverage term discourages selecting tiny nearly linear tails.
        """
        if not hasattr(self,'h0'):
            self.h0=max(max(d['H']) for d in self.data_files)
            self.m0=np.median(np.concatenate([d['M'] for d in self.data_files]))
        previous=self.hc_value; rows=[]
        b=self.results.get('beta',.365); g=self.results.get('gamma',1.386)
        for frac in np.linspace(0,.45,10):
            self.hc_value=float(frac*self.h0); scores=[]; retained=0; total=0
            for d in self.data_files:
                allh,_=self._get_filtered_data(d,cutoff=False); h,m=self._get_filtered_data(d)
                total+=len(allh); retained+=len(h)
                f=self._fit_modified_arrott_isotherm(h,m,beta=b,gamma=g)
                if f: scores.append(f['score'])
            valid=len(scores)==len(self.data_files)
            score=float(np.median(scores)+.01*(1-retained/total)) if valid else np.inf
            rows.append(dict(cutoff_value=self.hc_value,field_unit=getattr(self,'field_representation','T'),
                             fraction=float(frac),score=score,valid=valid))
        self.hc_value=previous
        valid=[r for r in rows if r['valid']]
        if not valid: raise AnalysisError('cutoff_failure','No cutoff retains sufficient fitting information')
        best=min(valid,key=lambda r:r['score'])
        self.results['cutoff_scan']=rows
        return best['cutoff_value']

    def bootstrap_analysis(self,n_iter=500,*,callback=None,tc_fixed=None,fixed_params=False,
                           mode='full_pipeline',seed=12345):
        """Wild residual resampling at fixed, unique H/T design points.

        Local linear leave-point-out residuals approximate point noise. This does
        not model temperature calibration, correlated noise, or model bias.
        Full mode reruns the exact point pipeline including automatic cutoff/KF.
        """
        if mode not in {'conditional','full_pipeline'}: raise ValueError('Unknown bootstrap mode')
        rng=np.random.default_rng(seed); samples=[]; failures=[]; review_samples=[]
        base=copy.deepcopy(self.data_files)
        for i in range(n_iter):
            replica=[]
            for d in base:
                h,m=self._get_filtered_data(d,cutoff=False)
                logm=np.log(m)
                # Smooth local trend with a fixed five-point polynomial filter.
                from scipy.signal import savgol_filter
                trend=savgol_filter(logm,min(7,len(logm) if len(logm)%2 else len(logm)-1),2)
                residual=logm-trend
                new=trend+residual*rng.choice([-1.,1.],len(m))
                replica.append(dict(d,H=h.copy(),M=np.exp(new)))
            e=PhysicsEngine(); e.data_files=replica
            cfg=dict(self.config,bootstrap=0,mcmc=0,use_gp=False,use_eos=False,use_mce=False,
                     use_correction=False,joint_fit=False)
            if mode=='conditional':
                cfg['tc_fixed']=tc_fixed if tc_fixed is not None else self.results['Tc']
                cfg['auto_cutoff']=False; cfg['H_min']=self.hc_value
            try:
                run_pipeline(e,cfg)
                if e.results['status']!='success':
                    if e.results['status']=='review_required':
                        review_samples.append(dict(iteration=i,flags=e.results.get('flags',[]),
                            parameters=[e.results[k] for k in ['beta','gamma','Tc','delta']]))
                    failures.append(dict(iteration=i,status=e.results['status'],
                        reason=e.results.get('failure') or e.results.get('flags',[]))); continue
                samples.append([e.results[k] for k in ['beta','gamma','Tc','delta']])
            except (ValueError,RuntimeError) as exc:
                failures.append(dict(iteration=i,reason=str(exc)))
            if callback: callback(int(100*(i+1)/n_iter))
        arr=np.asarray(samples).reshape(-1,4)
        out=dict(mode=mode,n_requested=n_iter,n_success=len(arr),failures=failures,
            n_review=len(review_samples),review_samples=review_samples,
            seed=seed,samples=arr,columns=['beta','gamma','Tc','delta'],
            interpretation='uncalibrated wild-residual bootstrap; design and equilibrium assumptions fixed',
            preview=n_iter<500)
        if len(arr)>=3:
            out['quantiles_025_16_50_84_975']=np.quantile(arr,[.025,.16,.5,.84,.975],axis=0)
            out['covariance']=np.cov(arr,rowvar=False)
            out['split_quantile_difference']=np.quantile(arr[:len(arr)//2],[.16,.84],axis=0)-np.quantile(arr[len(arr)//2:],[.16,.84],axis=0) if len(arr)>5 else None
        self.results.setdefault('bootstrap',{})[mode]=out
        if mode=='full_pipeline' and 'covariance' in out:
            self.results['joint_covariance']=out['covariance'][:3,:3]
        return out

    def mcmc_uncertainty(self,*,target='collapse',n=4000,burn=1000,tc_fixed=None,
                         loss_temperature=1.,chains=4,seed=9841):
        if target!='collapse':
            raise ValueError('GP profile likelihood is diagnostic only; GP posterior sampling is disabled')
        if not (n-burn>=4 and burn>=0 and chains>=2 and loss_temperature>0):
            raise ValueError('Require n>burn>=0, >=2 chains, positive loss temperature')
        if tc_fixed is None and self.config.get('tc_mode','fixed')=='fixed': tc_fixed=self.results['Tc']
        fixed=tc_fixed is not None
        temps=np.array([d['T'] for d in self.data_files]); span=np.ptp(temps)
        bounds=np.array(list(BOUNDS)+([] if fixed else [(min(temps)+1e-5,max(temps)-1e-5)]))
        centre=np.array([self.results['beta'],self.results['gamma']]+([] if fixed else [self.results['Tc']]))
        step=np.array([.015,.035]+([] if fixed else [.01*span]))
        rng=np.random.default_rng(seed)
        def logp(p):
            if np.any(p<=bounds[:,0]) or np.any(p>=bounds[:,1]): return -np.inf
            cost=self._collapse_cost(p[0],p[1],tc_fixed if fixed else p[2])
            return -cost/loss_temperature if np.isfinite(cost) else -np.inf
        if not np.isfinite(logp(centre)):
            raise AnalysisError('sensitivity_target_undefined','No supported loss at the point estimate; chains were not generated')
        traces=[]; accepts=[]; initials=[]
        for _ in range(chains):
            p=np.clip(centre+rng.normal(size=len(centre))*step*2,bounds[:,0]+1e-4,bounds[:,1]-1e-4)
            if not np.isfinite(logp(p)): p=centre.copy()
            lp=logp(p); initials.append(p.copy()); trace=[]; accepted=0
            for i in range(n):
                q=p+rng.normal(size=len(p))*step; lq=logp(q)
                if np.isfinite(lq) and (not np.isfinite(lp) or np.log(rng.random())<lq-lp):
                    p,lp=q,lq; accepted+=1
                trace.append(p.copy())
            traces.append(trace); accepts.append(accepted/n)
        trace=np.array(traces); samples=trace[:,burn:,:]
        rhat,ess=chain_diagnostics(samples)
        out=dict(kind='loss-based sensitivity distribution',calibrated=False,
            density='uniform bounded prior * exp(-C / loss_temperature)',loss_temperature=loss_temperature,
            Tc_mode='fixed' if fixed else 'joint',Tc_fixed=tc_fixed,prior_bounds=bounds,
            columns=['beta','gamma']+([] if fixed else ['Tc']),seed=seed,n=n,burn=burn,
            initials=initials,trace=trace,samples=samples,acceptance=accepts,
            split_rhat=rhat,ess=ess,
            diagnostics_pass=bool(np.all(rhat<1.01) and np.all(ess>400)),
            quantiles_16_50_84=np.quantile(samples.reshape(-1,len(centre)),[.16,.5,.84],axis=0))
        self.results['mcmc']=out
        return out

    def compare_universality_classes(self,*,include_eos=False):
        cov=self.results.get('joint_covariance')
        scores={}
        if cov is None or np.shape(cov)[0]<2:
            self.results.update(scores={},best_model=None,universality_status='indeterminate_no_joint_uncertainty')
            return scores
        cov=np.asarray(cov)[:2,:2]
        if not np.all(np.isfinite(cov)) or np.linalg.eigvalsh(cov).min()<=1e-14:
            self.results.update(scores={},best_model=None,universality_status='indeterminate_singular_covariance')
            return scores
        estimate=np.array([self.results['beta'],self.results['gamma']])
        for name,(b,g,source) in REFERENCES.items():
            diff=estimate-[b,g]; distance=float(diff@np.linalg.solve(cov,diff))
            scores[name]=dict(S=distance,mahalanobis_squared=distance,source=source,
                collapse_cost=self._collapse_cost(b,g,self.results['Tc']),
                interpretation='heuristic compatibility; not class probability')
        compatible=[k for k,v in scores.items() if v['S']<=5.991]
        self.results.update(scores=scores,best_model=compatible[0] if len(compatible)==1 else None,
            universality_status='heuristic_single_match' if len(compatible)==1 else 'indeterminate' if compatible else 'not_applicable',
            compatible_references=compatible,
            universality_rule='Mahalanobis squared <= 5.991; uncalibrated bootstrap covariance')
        return scores

    def independent_delta(self):
        tc=self.results['Tc']; d=min(self.data_files,key=lambda c:abs(c['T']-tc))
        h,m=self._get_filtered_data(d)
        fit=stats.linregress(np.log(h),np.log(m))
        out=dict(T=d['T'],temperature_offset=d['T']-tc,delta=1/fit.slope if fit.slope>0 else None,
            r2=fit.rvalue**2,n=len(h),field_range_T=[min(h),max(h)],
            interpretation='near-critical isotherm diagnostic; finite temperature offset is not corrected')
        self.results['delta_isotherm']=out
        return out

    def mce_analysis(self,advanced_smooth=False,*,deriv_method='pchip',**kwargs):
        if not self.config.get('reversible'): raise AnalysisError('irreversible_mce','Maxwell relation requires reversible equilibrium data')
        if getattr(self,'field_representation','T')=='relative':
            raise AnalysisError('mce_field_units','Absolute Maxwell entropy requires confirmed physical field units')
        unit=self.config.get('magnetization_unit')
        if unit not in {'A m2/kg','A/m'}:
            raise AnalysisError('mce_units','Specify mass magnetisation A m2/kg or volume magnetisation A/m')
        curves=[self._get_filtered_data(d,cutoff=False,include_zero=True) for d in self.data_files]
        lo=max(h[0] for h,m in curves); hi=min(h[-1] for h,m in curves)
        if lo>=hi: raise AnalysisError('mce_no_overlap','No shared measured field interval')
        grid=np.linspace(lo,hi,150); temperatures=np.array([d['T'] for d in self.data_files])
        matrix=np.array([np.interp(grid,h,m) for h,m in curves])
        derivative=interpolate.PchipInterpolator(temperatures,matrix,axis=0).derivative()(temperatures)
        ds=cumulative_trapezoid(derivative,grid,axis=1,initial=0)
        self.mce_data=dict(T=temperatures,H=grid,dS=ds,sigma=None,
            entropy_unit='J kg-1 K-1' if unit=='A m2/kg' else 'J m-3 K-1',
            field_interval_T=[lo,hi],low_field_missing=lo>0,
            interpretation='entropy difference from measured lower field; not zero-field entropy change',
            uncertainty='not estimated; temperature correlation and calibration require a measurement model')
        self.results['mce']=self.mce_data
        self.results['Tc_for_mce']=self.results['Tc']
        self.results['n_theo']=1+(self.results['beta']-1)/(self.results['beta']+self.results['gamma'])
        return self.mce_data

    def _build_scaled_dataset(self,beta,gamma,tc,*,max_points=1200,per_curve=40):
        curves=self._scaled_curves(beta,gamma,tc,per_curve)
        if not curves: return np.empty((0,2)),np.empty(0)
        X=np.concatenate([np.column_stack([c['x'],np.full(len(c['x']),c['side'])]) for c in curves])
        y=np.concatenate([c['y'] for c in curves])
        if len(y)>max_points:
            idx=np.linspace(0,len(y)-1,max_points).astype(int); X,y=X[idx],y[idx]
        return X,y

    def gp_diagnostic(self):
        """Separate GP per branch at final parameters; no Bayesian exponent claim.

        Fixed input/output scales and deterministic subsets. Hyperparameters are
        profiled once for this descriptive plot, not sampled as physical inference.
        """
        if not SKLEARN_AVAILABLE:
            self.results['gp']=dict(success=False,reason='scikit-learn unavailable'); return self.results['gp']
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        X,y=self._build_scaled_dataset(self.results['beta'],self.results['gamma'],self.results['Tc'])
        branches={}
        for side in [-1,1]:
            sel=X[:,1]==side
            if sel.sum()<12:
                branches[str(side)]=dict(success=False,reason='insufficient points'); continue
            kernel=ConstantKernel(1.,(.01,100))*RBF(1.,(.05,50))+WhiteKernel(.001,(1e-6,.5))
            gp=GaussianProcessRegressor(kernel=kernel,normalize_y=False,random_state=0,n_restarts_optimizer=0)
            gp.fit(X[sel,:1],y[sel])
            grid=np.linspace(min(X[sel,0]),max(X[sel,0]),150)
            mean,sd=gp.predict(grid[:,None],return_std=True)
            branches[str(side)]=dict(success=True,x=grid,mean=mean,sd=sd,n_used=int(sel.sum()),
                kernel=str(gp.kernel_),profile_log_marginal_likelihood=float(gp.log_marginal_likelihood_value_))
        out=dict(success=True,branches=branches,interpretation='descriptive empirical-Bayes branch curves at fixed exponents; no posterior over exponents')
        self.results['gp']=out
        return out

    def eos_diagnostics(self):
        """Direct Arrott-Noakes H predictions on one fixed point set.

        This replaces the previous approximate parametric root fits. Loss is mean
        squared log(H_pred/H_observed), not reduced chi-square. Invalid predictions
        invalidate a candidate; observations are never dropped to improve a model.
        """
        ts=[]; hs=[]; ms=[]
        for d in self.data_files:
            h,m=self._get_filtered_data(d)
            idx=np.linspace(0,len(h)-1,min(40,len(h))).astype(int)
            hs.extend(h[idx]/self.h0); ms.extend(m[idx]/self.m0); ts.extend([d['T']]*len(idx))
        t,h,m=map(np.asarray,[ts,hs,ms]); eps=(t-self.results['Tc'])/self.results['Tc']; output={}
        for name,(beta,gamma,source) in REFERENCES.items():
            def prediction(p):
                a,b=np.exp(p); base=a*eps+b*m**(1/beta)
                if np.any(base<=0): return None
                return m*base**gamma
            def objective(p):
                if np.any(np.abs(p)>20): return 1e6
                pred=prediction(p)
                return float(np.mean(np.log(pred/h)**2)) if pred is not None else 1e6
            candidates=[minimize(objective,np.array([a,b],float),method='Nelder-Mead',
                                options=dict(maxiter=500,xatol=1e-6,fatol=1e-10))
                        for a,b in [(-3,0),(-6,0),(-10,1)]]
            fit=min(candidates,key=lambda r:r.fun); pred=prediction(fit.x)
            valid=pred is not None and np.all(np.isfinite(pred)) and fit.fun<1e6
            output[name]=dict(success=bool(fit.success and valid),model='Arrott-Noakes reference-exponent diagnostic',
                beta=beta,gamma=gamma,Tc=self.results['Tc'],scale_a=float(np.exp(fit.x[0])),scale_b=float(np.exp(fit.x[1])),
                log_field_mse=float(fit.fun) if valid else None,n_points=len(h),n_used=len(h) if valid else 0,
                invalid_prediction_fraction=0. if valid else 1.,T=t,H_observed_T=h*self.h0,
                H_predicted_T=pred*self.h0 if valid else None,log_residuals=np.log(pred/h) if valid else None)
        self.results['eos_fits']=output
        return output


def chain_diagnostics(samples):
    """Split R-hat and initial-positive-pair autocorrelation ESS (not rank R-hat)."""
    c,n,d=samples.shape; half=n//2
    split=np.concatenate([samples[:,:half,:],samples[:,-half:,:]],axis=0)
    W=np.mean(np.var(split,axis=1,ddof=1),axis=0)
    B=half*np.var(np.mean(split,axis=1),axis=0,ddof=1)
    v=(half-1)/half*W+B/half
    rhat=np.sqrt(np.divide(v,W,out=np.full(d,np.inf),where=W>0))
    ess=[]
    for j in range(d):
        centred=split[:,:,j]-split[:,:,j].mean(axis=1,keepdims=True)
        fft=np.fft.rfft(centred,n=2*half,axis=1)
        acov=np.fft.irfft(fft*np.conj(fft),axis=1)[:,:half]/half
        rho=1-(W[j]-acov.mean(axis=0))/v[j] if v[j]>0 else np.ones(half)
        pairs=[]
        for k in range(1,half-1,2):
            pair=rho[k]+rho[k+1]
            if pair<0: break
            pairs.append(pair)
        ess.append(min(2*c*half,2*c*half/max(1.,1+2*sum(pairs))))
    return rhat,np.array(ess)


def run_pipeline(engine,cfg,*,log=None):
    """Only supported point-analysis pipeline; called identically by both UIs."""
    log=log or (lambda message:None)
    start=time.perf_counter(); engine.config=copy.deepcopy(cfg)
    engine.results=dict(status='running',flags=[],errors={},history=[])
    engine.mce_data=None; engine._gp_cache={}
    engine.hc_value=float(cfg.get('H_min',cfg.get('Bc',0.)))
    history=engine.results['history']; flags=engine.results['flags']
    try:
        if getattr(engine,'load_errors',[]):
            raise AnalysisError('input_load_failure',str(engine.load_errors))
        inputs=[]
        for d in engine.data_files:
            digest=hashlib.sha256(np.asarray(d['H'],dtype='<f8').tobytes()+np.asarray(d['M'],dtype='<f8').tobytes()).hexdigest()
            inputs.append(dict(file=d['file'],T=d['T'],array_sha256=digest,
                file_sha256=d.get('sha256'),input_field_unit=d.get('input_field_unit','T supplied programmatically')))
        engine.results['provenance']=dict(version=VERSION,base_commit=BASE_COMMIT,
            source_hashes=LOADED_SOURCE_HASHES,inputs=inputs,config=copy.deepcopy(cfg),
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__),
            platform=platform.platform())
        engine.prepare(); history.append(dict(state='prepared'))
        if engine.field_representation=='relative': flags.append('relative_field_units_only')
        fractions=cfg.get('window_fractions',[.4,.55,.7,.85,1.])
        if not fractions or any(not 0<f<=1 for f in fractions):
            raise AnalysisError('invalid_windows','Window fractions must lie in (0,1]')
        if not 0<=engine.hc_value<engine.h0:
            raise AnalysisError('invalid_cutoff','Cutoff must lie within the measured field range')
        temperatures=np.array([d['T'] for d in engine.data_files])
        engine.results['temperature_resolution_K']=np.diff(temperatures).tolist()
        engine.results.update(beta=float(cfg.get('init_beta',.365)),gamma=float(cfg.get('init_gamma',1.386)),Tc=float(np.mean(temperatures)))
        if cfg.get('auto_cutoff',False): engine.hc_value=engine.auto_suggest_bc()
        engine.results['H_min_T']=engine.hc_value if engine.field_representation=='T' else None
        engine.results['H_min_value']=engine.hc_value
        engine.results['field_unit']=engine.field_representation
        fixed=cfg.get('tc_fixed')
        if fixed is not None:
            tc=float(fixed); source='user_fixed'; history.append(dict(state='explicit_Tc',Tc=tc))
        else:
            source=None; previous=None; converged=False
            for i in range(int(cfg.get('kf_max_iter',12))):
                engine.perform_kf_and_correction()
                tc=engine.results.get('Tc_KF')
                history.append(dict(state='MAP_KF',iteration=i,Tc=tc,beta=engine.results['beta'],gamma=engine.results['gamma'],sides=engine.results.get('kf_sides')))
                if tc is None: break
                if engine.results['kf_sides']==1: flags.append('one_sided_KF')
                current=np.array([engine.results['beta'],engine.results['gamma'],tc])
                engine.results['Tc']=tc
                if previous is not None and abs(current[2]-previous[2])<float(cfg.get('kf_tol',.001)) and np.max(np.abs(current[:2]-previous[:2]))<.001:
                    converged=True; break
                previous=current
                if not cfg.get('fixed_params',False):
                    for key,kk,bound in [('beta','beta_KF',BOUNDS[0]),('gamma','gamma_KF',BOUNDS[1])]:
                        v=engine.results.get(kk)
                        if v is not None and bound[0]<v<bound[1]:
                            engine.results[key]=.5*engine.results[key]+.5*v
            if tc is not None:
                source='KF';
                if not converged: flags.append('KF_iteration_limit')
            elif cfg.get('allow_collapse_fallback',True):
                log('KF unavailable; evaluating measured-temperature candidates')
                candidates=[]
                for candidate in temperatures[2:-2] if len(temperatures)>=5 else [(temperatures[1]+temperatures[2])/2]:
                    f=engine.optimize_data_collapse(engine.results['beta'],engine.results['gamma'],tc_fixed=float(candidate),update_results=False)
                    candidates.append(f)
                good=[f for f in candidates if f['success']]
                engine.results['fallback_candidates']=candidates
                if not good: raise AnalysisError('no_transition_estimate','KF and collapse fallback failed')
                best=min(good,key=lambda f:f['cost']); tc=best['Tc']; source='collapse_discrete'
                engine.results['Tc_collapse']=tc
                engine.results.update(beta=best['beta'],gamma=best['gamma'])
                engine.results['fallback_candidates']=candidates; flags.append('KF_unavailable')
            else: raise AnalysisError('KF_unavailable','No usable KF temperature and fallback disabled')
        engine.results.update(Tc=tc,Tc_source=source)
        before={k:engine.results[k] for k in ['beta','gamma','Tc']}
        fit=engine.optimize_data_collapse(engine.results['beta'],engine.results['gamma'],
            tc_fixed=tc,fixed_params=bool(cfg.get('fixed_params',False)))
        history.append(dict(state='final_fixed_Tc_collapse',before=before,after={k:engine.results[k] for k in before}))
        if not fit['success']: raise AnalysisError('collapse_failure',fit['diagnostics'].get('reason') or 'Optimizer did not converge')
        if fit['boundary_hit']: flags.append('parameter_boundary')
        engine.independent_delta()
        engine.results['status']='review_required' if flags else 'success'
        if cfg.get('use_correction'): engine.perform_kf_and_correction(True,omega=cfg.get('omega',.5),fit_omega=cfg.get('fit_omega',True))
        if cfg.get('use_eos'): engine.results['eos_fits']=engine.eos_diagnostics()
        if cfg.get('use_gp'): engine.gp_diagnostic()
        if cfg.get('use_mce'):
            try: engine.mce_analysis()
            except AnalysisError as exc: engine.results['mce']=dict(success=False,reason=exc.code,detail=str(exc))
        if cfg.get('joint_fit'):
            engine.results['joint_fit']=dict(success=False,reason='disabled_without_M_and_derived_MCE_covariance_model')
        nboot=cfg.get('bootstrap',0)
        if isinstance(nboot,bool): nboot=cfg.get('bootstrap_n',500) if nboot else 0
        if nboot:
            for mode in ['conditional','full_pipeline']:
                engine.bootstrap_analysis(int(nboot),mode=mode,seed=int(cfg.get('seed',12345)))
        engine.compare_universality_classes()
        nmc=cfg.get('mcmc',0)
        if isinstance(nmc,bool): nmc=cfg.get('mcmc_n',4000) if nmc else 0
        if nmc:
            engine.mcmc_uncertainty(n=int(nmc),burn=int(cfg.get('burn',int(nmc)//4)),
                loss_temperature=float(cfg.get('loss_temperature',1.)),tc_fixed=tc if cfg.get('tc_mode','fixed')=='fixed' else None)
    except (AnalysisError,ValueError,RuntimeError) as exc:
        engine.results.update(status='failed',failure=dict(code=getattr(exc,'code','analysis_exception'),message=str(exc)),best_model=None)
        log(str(exc))
    engine.results['elapsed_seconds']=time.perf_counter()-start
    engine.results['flags']=list(dict.fromkeys(engine.results['flags']))
    return engine.results

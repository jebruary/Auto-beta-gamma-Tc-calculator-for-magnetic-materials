"""Generate reproducible numerical checks and objective surfaces.

These are synthetic method checks, not eight-material experimental validation or
interval-coverage calibration. Failures remain in the result table.
"""
import argparse
from pathlib import Path
import numpy as np
from critical_engine import PhysicsEngine,run_pipeline,save_json
from synthetic import mean_field,independent_scaling
from background import estimate_from_blank,subtract


def prepare(data):
    e=PhysicsEngine(); e.data_files=data; e.config=dict(reversible=True,multistart=False); e.prepare(); return e


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='benchmarks/generated')
    ap.add_argument('--replicates',type=int,default=12); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ranking=[]; recoveries=[]
    models=[('mean_field',.5,1.),('independent_XY_effective',.23,2.2),
            ('independent_Ising2D',.125,1.75),('independent_tricritical',.25,1.)]
    for name,b,g in models:
        for noise in [0.,.005,.02]:
            n=args.replicates if noise else 1
            for seed in range(n):
                data=mean_field(noise=noise,seed=seed,n_fields=80) if name=='mean_field' else independent_scaling(b,g,noise=noise,seed=seed,n_fields=80)
                e=prepare(data)
                candidates=[(b,g),(b*.8,g*1.15),(b*1.2,g*.85)]
                costs=[e._collapse_cost(bb,gg,100) for bb,gg in candidates]
                ranking.append(dict(model=name,noise=noise,seed=seed,candidates=candidates,costs=costs,truth_best=bool(costs[0]<min(costs[1:]))))
                fit=e.optimize_data_collapse(b*.95,g*1.05,tc_fixed=100,update_results=False)
                recoveries.append(dict(model=name,noise=noise,seed=seed,beta_true=b,gamma_true=g,
                    beta=fit['beta'],gamma=fit['gamma'],success=fit['success'],boundary_hit=fit['boundary_hit']))
            print(name,noise,flush=True)
    save_json(out/'truth_ranking.json',ranking); save_json(out/'recoveries.json',recoveries)
    base=prepare(mean_field())
    beta_grid=np.linspace(.25,.8,50); gamma_grid=np.linspace(.6,2.,50)
    surface=np.array([[base._collapse_cost(b,g,100) for b in beta_grid] for g in gamma_grid])
    np.savez_compressed(out/'objective_surface.npz',beta=beta_grid,gamma=gamma_grid,cost=surface)
    invariance=[]
    for units,factor in [('T',1.),('mT',1000.),('Oe',10000.)]:
        for density in [40,80,160]:
            data=mean_field(n_fields=density)
            for d in data: d['H']*=factor
            e=prepare(data)
            fit=e.optimize_data_collapse(.45,1.1,tc_fixed=100,update_results=False)
            invariance.append(dict(unit=units,density=density,beta=fit['beta'],gamma=fit['gamma'],cost=fit['cost']))
    save_json(out/'invariance.json',invariance)
    background=[]; rng=np.random.default_rng(440)
    h=np.linspace(.01,1,100)
    # Calibrator receives only an independent noisy blank, never injected chi.
    for kind in ['zero','linear','nonlinear']:
        for seed in range(100):
            chi=0 if kind=='zero' else .2
            blank=chi*h+(.1*h**3 if kind=='nonlinear' else 0)+rng.normal(0,.003,len(h))
            fit=estimate_from_blank(h,blank)
            background.append(dict(kind=kind,seed=seed,true_chi=chi,**fit))
    save_json(out/'background_blank_controls.json',background)
    timings=[]
    for variant,options in [('fixed_Tc',dict(tc_fixed=100)),('reestimate_Tc',{}),
                            ('fixed_window',dict(window_fractions=[.7])),('auto_cutoff',dict(auto_cutoff=True))]:
        e=PhysicsEngine();e.data_files=mean_field(n_fields=80)
        cfg=dict(reversible=True,init_beta=.5,init_gamma=1,multistart=False,**options)
        r=run_pipeline(e,cfg); save_json(out/(variant+'.json'),r)
        timings.append(dict(variant=variant,**{k:r.get(k) for k in ['status','flags','beta','gamma','Tc','elapsed_seconds']}))
    save_json(out/'pipeline_comparisons.json',timings)
    summary=dict(n_ranking=len(ranking),truth_best=sum(r['truth_best'] for r in ranking),
        n_recoveries=len(recoveries),fit_success=sum(r['success'] for r in recoveries),
        parameter_metrics=[])
    for name,b,g in models:
        rows=[r for r in recoveries if r['model']==name and r['success']]
        error=np.array([[r['beta']-b,r['gamma']-g] for r in rows])
        summary['parameter_metrics'].append(dict(model=name,n=len(rows),bias=error.mean(axis=0),rmse=np.sqrt(np.mean(error**2,axis=0))))
    summary['interval_coverage']='not established; loss-based and bootstrap intervals are not labelled calibrated'
    save_json(out/'summary.json',summary); print(summary,flush=True)


if __name__=='__main__': main()

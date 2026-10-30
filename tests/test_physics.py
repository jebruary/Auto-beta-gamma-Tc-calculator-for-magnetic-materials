import copy
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from critical_engine import PhysicsEngine,run_pipeline,AnalysisError,json_safe,chain_diagnostics
from synthetic import mean_field,independent_scaling
from analysis_gui import run_gui_analysis
from analysis_cli import run_full_pipeline


def prepared(data=None):
    e=PhysicsEngine(); e.data_files=mean_field() if data is None else data
    e.config=dict(reversible=True); e.prepare(); return e


@pytest.mark.parametrize('noise',[0.,.005,.02])
def test_review_counterexample_correct_truth_ranking(noise):
    e=prepared(mean_field(noise=noise))
    truth=e._collapse_cost(.5,1,100)
    wrong=[e._collapse_cost(b,g,100) for b,g in [(.35,1.4),(.23,2.2),(.8,.8)]]
    assert truth<min(wrong)


@pytest.mark.parametrize('beta,gamma',[(.23,2.2),(.125,1.75),(.326,1.237),(.25,1.)])
def test_non_arrott_noakes_scaling_truth(beta,gamma):
    e=prepared(independent_scaling(beta,gamma))
    truth=e._collapse_cost(beta,gamma,100)
    for b,g in [(beta*1.2,gamma*.85),(beta*.8,gamma*1.15)]:
        assert truth<e._collapse_cost(b,g,100)


def test_unit_and_density_invariance():
    e=prepared(); costs=[e._collapse_cost(.5,1,100,sample_per_curve=n) for n in [60,100,160]]
    for factor in [1000.,10000.]:
        data=copy.deepcopy(mean_field())
        for c in data: c['H']*=factor; c['M']*=100
        other=prepared(data)
        assert other._collapse_cost(.35,1.4,100)==pytest.approx(e._collapse_cost(.35,1.4,100),rel=1e-9)
    assert max(costs)<1e-5


def test_curve_not_compared_to_itself():
    e=prepared(); e.data_files=[e.data_files[0],e.data_files[-1]]
    assert not np.isfinite(e._collapse_cost(.5,1,100))


def test_hysteresis_rejected_and_explicit_sweep():
    e=PhysicsEngine(); d=dict(H=np.r_[np.arange(10),np.arange(8,-1,-1)],M=np.ones(19))
    with pytest.raises(AnalysisError,match='Multiple sweep'): e._get_filtered_data(d)
    e.config['branch']='ascending'; h,m=e._get_filtered_data(d); assert len(h)==9


def test_duplicate_temperature_aggregates_before_derivative():
    data=mean_field(); data.append(copy.deepcopy(data[0]))
    e=prepared(data); assert len(e.data_files)==21
    e.results=dict(beta=.5,gamma=1,Tc=100)
    e.perform_kf_and_correction()
    assert e.results['kf_sides']==2


def test_cli_gui_same_pipeline_and_final_refinement():
    cfg=dict(reversible=True,tc_fixed=100,init_beta=.5,init_gamma=1,multistart=False)
    a=PhysicsEngine(); b=PhysicsEngine(); a.data_files=mean_field(); b.data_files=mean_field()
    ra=run_full_pipeline(a,cfg); rb=run_gui_analysis(b,cfg)
    assert ra['status']=='success'
    for k in ['beta','gamma','Tc','collapse_cost','Tc_source']: assert ra[k]==rb[k]
    assert ra['history'][-1]['state']=='final_fixed_Tc_collapse'
    assert abs(ra['beta']-.5)<.01 and abs(ra['gamma']-1)<.02


def test_loader_units_and_fortran(tmp_path):
    p=tmp_path/'sample_100K.csv'; p.write_text('H_mT,M\n1D+0,2\n2D+0,3\n3D+0,4\n')
    e=PhysicsEngine(); assert e.load_files([p])==1
    assert np.allclose(e.data_files[0]['H'],[.001,.002,.003])
    p.write_text('1e0 2\n2e0 3\n3e0 4\n')
    assert e.load_files([p],field_unit='T')==1
    assert len(e.data_files[0]['H'])==3
    assert e.load_files([p])==0 and e.load_errors


def test_mce_no_extrapolation_and_physical_units():
    e=prepared(); e.results=dict(Tc=100,beta=.5,gamma=1)
    e.config['magnetization_unit']='A m2/kg'
    mce=e.mce_analysis()
    assert mce['H'][0]==max(d['H'][0] for d in e.data_files)
    assert mce['H'][-1]==min(d['H'][-1] for d in e.data_files)
    assert np.isfinite(mce['dS']).all() and mce['low_field_missing']


def test_fixed_and_joint_mcmc_dimensions():
    e=prepared(mean_field(n_fields=30,n_temperatures=9)); e.results=dict(beta=.5,gamma=1,Tc=100)
    e.config['tc_mode']='fixed'
    fixed=e.mcmc_uncertainty(n=30,burn=10,chains=2)
    assert fixed['samples'].shape==(2,20,2) and fixed['Tc_fixed']==100
    e.config['tc_mode']='joint'
    joint=e.mcmc_uncertainty(n=30,burn=10,chains=2)
    assert joint['samples'].shape==(2,20,3) and not joint['calibrated']


def test_strict_json():
    import json
    assert json.dumps(json_safe(dict(a=np.array([1,np.nan]),b=np.bool_(True))),allow_nan=False)


def test_fixed_correction_parameter_removed_from_vector():
    e=prepared(); e.results=dict(Tc=100,beta=.5,gamma=1)
    t=np.r_[np.linspace(90,99,12),np.linspace(101,110,12)]; u=abs((t-100)/100)
    ms=np.where(t<100,2*u**.5*(1+.1*u**.5),np.nan)
    chi=np.where(t>100,3*u*(1+.1*u**.5),np.nan)
    out=e._correction_fit(t,ms,chi,omega=.5,fit_omega=False)
    assert all(v['success'] and v['Delta']==.5 for v in out.values())


def test_eos_outputs_predictions_not_chi_square():
    e=prepared(mean_field(n_fields=24,n_temperatures=9)); e.results=dict(Tc=100,beta=.5,gamma=1)
    out=e.eos_diagnostics()['Mean-field']
    assert out['success'] and out['n_used']==out['n_points']
    assert out['log_field_mse']<1e-6 and 'chi2_red' not in out


def test_full_mean_field_pipeline_estimates_temperature():
    e=PhysicsEngine();e.data_files=mean_field(n_fields=80)
    r=run_pipeline(e,dict(reversible=True,init_beta=.5,init_gamma=1,multistart=False))
    assert r['status']=='success' and r['Tc_source']=='KF'
    assert abs(r['Tc']-100)<.01
    assert abs(r['beta']-.5)<.01 and abs(r['gamma']-1)<.02


def test_mce_retains_measured_zero_field():
    data=mean_field()
    for c in data:
        c['H']=np.r_[0.,c['H']]; c['M']=np.r_[0.,c['M']]
    e=prepared(data); e.results=dict(Tc=100,beta=.5,gamma=1);e.config['magnetization_unit']='A m2/kg'
    assert e.mce_analysis()['H'][0]==0.


def test_two_temperature_branches_are_separate_for_gp():
    e=prepared(); X,y=e._build_scaled_dataset(.5,1,100,max_points=100)
    assert set(X[:,1])=={-1,1}
    X2,y2=e._build_scaled_dataset(.5,1,100,max_points=100)
    assert np.array_equal(X,X2) and np.array_equal(y,y2)


def test_relative_fields_cannot_generate_absolute_entropy():
    data=mean_field()
    for c in data: c['field_unit']='relative'
    e=prepared(data); e.results=dict(Tc=100,beta=.5,gamma=1);e.config['magnetization_unit']='A m2/kg'
    with pytest.raises(AnalysisError,match='Absolute Maxwell'): e.mce_analysis()


def test_maxwell_integral_against_analytic_reversible_model():
    h=np.linspace(0,2,40)
    data=[dict(file=f'{t}K',T=t,H=h.copy(),M=(t+2)*h) for t in np.linspace(20,25,8)]
    e=prepared(data); e.config['magnetization_unit']='A m2/kg';e.results=dict(Tc=22,beta=.5,gamma=1)
    out=e.mce_analysis()
    assert np.allclose(out['dS'],.5*out['H'][None,:]**2,atol=1e-9)

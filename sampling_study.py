"""Reproducible sampling diagnostics; intervals are explicitly uncalibrated."""
import argparse
from pathlib import Path
from critical_engine import PhysicsEngine,run_pipeline,save_json
from synthetic import mean_field


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='benchmarks/sampling')
    ap.add_argument('--bootstrap',type=int,default=80);ap.add_argument('--steps',type=int,default=4000)
    args=ap.parse_args(); out=Path(args.out)
    e=PhysicsEngine();e.data_files=mean_field(noise=.01,seed=912,n_fields=60,n_temperatures=13)
    run_pipeline(e,dict(reversible=True,init_beta=.5,init_gamma=1,multistart=False,tc_fixed=100))
    save_json(out/'point.json',e.results)
    for mode in ['conditional','full_pipeline']:
        config=dict(e.config)
        if mode=='full_pipeline':
            e.config.pop('tc_fixed',None)
        result=e.bootstrap_analysis(args.bootstrap,mode=mode,seed=321)
        save_json(out/(mode+'.json'),result); e.config=config
        print(mode,result['n_success'],'/',result['n_requested'],flush=True)
    for tau in [1.,.0002]:
        sample=e.mcmc_uncertainty(n=args.steps,burn=args.steps//4,loss_temperature=tau,tc_fixed=100)
        save_json(out/('sensitivity_tau_'+str(tau)+'.json'),sample)
        print('tau',tau,'Rhat',sample['split_rhat'],'ESS',sample['ess'],flush=True)


if __name__=='__main__':main()

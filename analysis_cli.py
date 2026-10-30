"""Command line adapter for the shared pipeline."""
import argparse
import glob
import json
from pathlib import Path

from critical_engine import PhysicsEngine, run_pipeline, save_json, FIELD_FACTORS


def run_full_pipeline(engine,cfg):
    return run_pipeline(engine,cfg)


def main(argv=None):
    ap=argparse.ArgumentParser(description='Reversible magnetic criticality analysis with recorded diagnostics')
    ap.add_argument('--data',nargs='+',required=True)
    ap.add_argument('--field-unit',choices=list(FIELD_FACTORS))
    ap.add_argument('--magnetization-unit',default='unspecified')
    ap.add_argument('--reversible',action='store_true',help='Confirm reversible equilibrium data in the selected domain')
    ap.add_argument('--branch',choices=['auto','ascending','descending'],default='auto')
    ap.add_argument('--h-min','--bc',dest='H_min',type=float,default=0.,help='Low field exclusion in tesla after conversion, or declared relative field units')
    ap.add_argument('--auto-cutoff',action='store_true')
    ap.add_argument('--tc-fixed',type=float)
    ap.add_argument('--init-beta',type=float,default=.365)
    ap.add_argument('--init-gamma',type=float,default=1.386)
    ap.add_argument('--fixed-params',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=0,help='Replicates in EACH conditional/full-pipeline analysis')
    ap.add_argument('--mcmc',type=int,default=0,help='Steps per sensitivity chain; this is not calibrated posterior inference')
    ap.add_argument('--tc-mode',choices=['fixed','joint'],default='fixed')
    ap.add_argument('--loss-temperature',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=12345)
    ap.add_argument('--use-gp',action='store_true')
    ap.add_argument('--use-eos',action='store_true')
    ap.add_argument('--use-mce',action='store_true')
    ap.add_argument('--use-correction',action='store_true')
    ap.add_argument('--omega',type=float,default=.5,help='Effective temperature correction exponent Delta')
    ap.add_argument('--fit-omega',action='store_true')
    ap.add_argument('--config',help='JSON configuration; explicit command line values take precedence')
    ap.add_argument('--out',default='out')
    args=ap.parse_args(argv)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    # Only supplied flags override a configuration file, not parser defaults.
    import sys
    supplied=list(argv if argv is not None else sys.argv[1:])
    parsed=vars(args)
    if args.config:
        for action in ap._actions:
            if any(s in supplied or any(v.startswith(s+'=') for v in supplied) for s in action.option_strings):
                if action.dest in parsed: cfg[action.dest]=parsed[action.dest]
    else: cfg.update(parsed)
    paths=[]
    for pattern in args.data: paths.extend(sorted(glob.glob(pattern)) or [pattern])
    e=PhysicsEngine()
    e.load_files(paths,field_unit=cfg.get('field_unit'),magnetization_unit=cfg.get('magnetization_unit'))
    results=run_pipeline(e,cfg)
    out=Path(args.out); save_json(out/'summary.json',results)
    print(f"{results['status']}: {out/'summary.json'}")
    return 2 if results['status']=='failed' else 0


if __name__=='__main__':
    raise SystemExit(main())

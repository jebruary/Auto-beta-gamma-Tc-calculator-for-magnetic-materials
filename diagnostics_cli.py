"""Shared dimensionless MAP diagnostics; scores are not class probabilities."""
import argparse
import numpy as np
from critical_engine import PhysicsEngine,REFERENCES,FIELD_FACTORS,save_json


def main(argv=None,*,field_correction=False):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data','--files',nargs='+',required=True,dest='data')
    ap.add_argument('--field-unit',choices=list(FIELD_FACTORS),required=True)
    ap.add_argument('--branch',choices=['auto','ascending','descending'],default='auto')
    ap.add_argument('--reversible',action='store_true')
    ap.add_argument('--out',default='map_diagnostics.json')
    if field_correction:
        ap.add_argument('--field-coefficient',type=float,required=True,
            help='User supplied kappa in T per input M unit for mu0H_internal = mu0H_applied - kappa*M')
    args=ap.parse_args(argv)
    engine=PhysicsEngine(); engine.load_files(args.data,field_unit=args.field_unit)
    if engine.load_errors: ap.error(str(engine.load_errors))
    engine.config=dict(reversible=args.reversible,branch=args.branch)
    if field_correction:
        for d in engine.data_files: d['H']=d['H']-args.field_coefficient*d['M']
    engine.prepare(); output={}
    for name,(beta,gamma,source) in REFERENCES.items():
        fits=[]
        for d in engine.data_files:
            h,m=engine._get_filtered_data(d)
            fit=engine._fit_modified_arrott_isotherm(h,m,beta=beta,gamma=gamma)
            fits.append(dict(T=d['T'],fit=fit))
        output[name]=dict(beta=beta,gamma=gamma,source=source,fits=fits,
            median_cv_score=float(np.median([r['fit']['score'] for r in fits if r['fit']])))
    save_json(args.out,dict(models=output,field_coefficient=getattr(args,'field_coefficient',None),
        interpretation='window and field-correction sensitivity; no zero-intercept class ranking'))
    print(args.out)


if __name__=='__main__': main()

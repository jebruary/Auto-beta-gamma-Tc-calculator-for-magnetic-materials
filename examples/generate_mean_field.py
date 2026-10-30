"""Generate explicitly synthetic mean-field CSV isotherms for the CLI example."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from synthetic import mean_field


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='examples/generated')
    args=parser.parse_args()
    folder=Path(args.out); folder.mkdir(parents=True,exist_ok=True)
    for curve in mean_field(n_fields=80):
        path=folder/f"synthetic_mean_field_{curve['T']:g}K.csv"
        np.savetxt(path,np.column_stack([curve['H'],curve['M']]),delimiter=',',
                   header='H_T,M',comments='')
    print(f'Synthetic data written to {folder}')


if __name__=='__main__': main()

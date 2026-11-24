"""Self-convergence of the independent L1/centered-FD numerical reference."""
import argparse
import numpy as np
from pathlib import Path
from numerical_reference import solve
from common import atomic_json,write_csv


def compare(sequence,space_fixed=None,time_fixed=None):
    data=[]
    for count in sequence:
        M=space_fixed or count;N=time_fixed or count
        x,t,psi,meta=solve(M,N)
        data.append(psi)
    common_M=space_fixed or sequence[0];common_N=time_fixed or sequence[0]
    restricted=[p[::(p.shape[0]-1)//common_N,::(p.shape[1]-1)//common_M] for p in data]
    first=np.linalg.norm(restricted[1]-restricted[0])/np.linalg.norm(restricted[1])
    second=np.linalg.norm(restricted[2]-restricted[1])/np.linalg.norm(restricted[2])
    return dict(refinement='time' if space_fixed else 'space',intervals='->'.join(map(str,sequence)),
                fixed_intervals=space_fixed or time_fixed,common_grid=[common_M,common_N],
                delta_1=float(first),delta_2=float(second),observed_order=float(np.log2(first/second)))


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('--output',type=Path,default=Path('runs/C02_gradient_v2/reference/convergence'))
    q.add_argument('--smoke',action='store_true')
    a=q.parse_args(argv)
    rows=[compare([8,16,32],time_fixed=32),compare([8,16,32],space_fixed=32)] if a.smoke else [
        compare([200,400,800],time_fixed=2048),compare([512,1024,2048],space_fixed=800)]
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'self_convergence.csv',rows)
    atomic_json(a.output/'self_convergence.json',dict(rows=rows,smoke_only=a.smoke,
                method='independent implicit L1 + centered spatial FD',
                interpretation='observed differences only; not a proof of continuum accuracy; no ADI/fourth-order claim'))
    print(rows)


if __name__=='__main__':main()

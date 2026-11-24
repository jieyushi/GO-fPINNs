"""Case 1: independent alpha operator, per-stage early stop capped at 5000."""
import sys
from train_single_operator import run

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if any(x.split('=')[0]=='--operator' for x in argv):
        raise ValueError('This entry fixes --operator alpha; use train_single_operator.py to choose.')
    return run(['--operator','alpha']+argv)

if __name__=='__main__':main()

"""Small finite flow-vector percentiles; same linear statistic, less dispatch.

Used only on the female image-evidence path. Unsupported arrays or NumPy
versions use the original implementation, not an approximate alternative.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=64)
def _index(n,q):
    index=(n-1)*(q/100.)
    lo=int(index)
    return lo,min(lo+1,n-1),index-lo


def _small(a,q):
    values=np.sort(a)
    lo,hi,gamma=_index(len(values),q)
    x,y=values[lo],values[hi]
    delta=y-x
    return x+delta*gamma if gamma<.5 else y-delta*(1-gamma)


# NumPy's weak scalar promotion determines rounding of float32 flow values.
# Keep the generic reference on other versions until equivalence is tested.
_enabled=np.__version__=='2.4.6'


def percentile(a,q):
    if (_enabled and isinstance(a,np.ndarray) and a.ndim==1 and
        1<=a.size<=18 and a.dtype in (np.dtype('float32'),np.dtype('float64')) and
        type(q) is int and q in (80,90) and np.isfinite(a).all()):
        return _small(a,q)
    return np.percentile(a,q)

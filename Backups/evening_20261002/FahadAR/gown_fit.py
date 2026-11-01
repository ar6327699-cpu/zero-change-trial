"""Reversible shoulder-seam lift, exclusively for the two new gowns.

Source vertices, UVs, hem, cuff endpoints and other suits are unchanged.
"""
import numpy as np

TOPS={'0293bab834355628b971d50ef1ffef03','c3221bc4c35857d3933573e4657ff147'}
DRAPE='c567eaf2b4b35fb9aba8aa9a1300b59b'
KEY='Gown shoulder seam lift'

def apply(obj):
    if obj is None or obj.get('gs_id') not in TOPS|{DRAPE}:return False
    if not obj.data.shape_keys:obj.shape_key_add(name='Basis')
    keys=obj.data.shape_keys.key_blocks;co=np.array([v.co[:] for v in keys[0].data])
    x,y,z=co.T
    def smooth(v):
        v=np.clip(v,0,1);return v*v*(3-2*v)
    upper=smooth((y-296)/12)
    # Lift only the top-facing shoulder seam. Preserve the centre neckline,
    # axilla, chest, elbow and cuff; do not translate the whole gown.
    lateral=smooth((np.abs(x)-12)/18)*(1-smooth((np.abs(x)-45)/30))
    delta=4.5*upper*lateral
    key=keys.get(KEY) or obj.shape_key_add(name=KEY)
    dest=co.copy();dest[:,1]+=delta
    key.data.foreach_set('co',dest.ravel());key.value=1.
    key.slider_min=0.;key.slider_max=1.
    return True

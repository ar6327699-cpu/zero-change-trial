"""Scoped plain-sleeve finish for the three existing Male Eastern photo suits."""
TOPS={'7125317243db4845b83a5d9555b8717c','b59ffb92529b4d8283166e3c308b1b4c','ce96b46cd01c4fb2aa5370dcecf6ae67'}
BOTTOMS={'f3b56f5efb604bfd97b8fe9c4d7e3cef','6352468beb8c4b73b3134f03cd22891d','af76fb63bad046cb99ef4ee945d2a427'}

def owned(obj):return obj is not None and obj.get('gs_id') in TOPS|BOTTOMS

def revise(obj):
    if obj is None or obj.get('gs_id') not in TOPS or obj.data.get('gs_plain_sleeves_version',0)>=1:return False
    region=obj.data.attributes.get('GS_pattern_region');blend=obj.data.color_attributes.get('GS_side_blend')
    if region is None or blend is None:raise ValueError('Male Eastern sleeve region/material mask missing')
    # The tube geometry is already smooth. These diagonal folds were baked
    # into the product photograph, so subdivision cannot remove them. Reuse
    # its existing sampled cloth shader ONLY on the two plain sleeve regions.
    # Body/placket/buttons/UVs/image pixels and other outfits stay unchanged.
    for face in obj.data.polygons:
        if region.data[face.index].value not in (1,2):continue
        for loop in face.loop_indices:blend.data[loop].color=(1.,1.,1.,1.)
    obj.data['gs_plain_sleeves_version']=1
    obj.data.update();return True

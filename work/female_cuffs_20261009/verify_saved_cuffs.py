import bpy,json,sys
sys.dont_write_bytecode=True
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'FahadAR'))
import body_profiles
# Exercise the same saved-style registration used by the trial launcher.
body_profiles.register(None)
s=body_profiles.scene();keys=bpy.data.objects['Tops'].data.shape_keys.key_blocks
assert abs(s.gs_female_style_sleeve_length+.10)<1e-6
assert abs(s.gs_female_style_sleeve_width-.30)<1e-6
assert abs(keys['Style sleeve_length'].value+.10)<1e-6
assert abs(keys['Style sleeve_width'].value-.30)<1e-6
report=dict(saved_profile_reloads=True,registered_controls_retain_values=True,
            sleeve_length=s.gs_female_style_sleeve_length,sleeve_width=s.gs_female_style_sleeve_width,
            active_shape_values_match=True,camera_started=False)
(OUT/'roundtrip_validation.json').write_text(json.dumps(report,indent=2));print('CUFF_ROUNDTRIP '+json.dumps(report))

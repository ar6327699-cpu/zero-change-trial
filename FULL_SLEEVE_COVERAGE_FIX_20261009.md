# Female sleeves and side coverage — 9 October 2026

**ROLLED BACK on 9 October 2026 after the user's 23:31 recording showed a regression.** The changes below are historical and are no longer active on disk. All seven modified sources were restored byte-for-byte from the pre-update backup; both new modules were archived. See `ROLLBACK_FULL_SLEEVE_20261009.md`.

Inspected the supplied 68.53-second recording `2026-10-09 21-11-20.mp4`, the four screenshots, the overview across the recording, and 0.3-second transition samples around lowering the arms and raising them overhead. Inspection artifacts are in `work/full_sleeve_fit_20261009`.

## Findings

The earlier final-filter adjustment did not address all causes of visible garment separation.

1. The camera requested a 640x480 sensor image and cropped it to 640x360. This discarded 60 sensor pixels above and below before detection. In the recording, raised hands leave the displayed frame and the sleeves then retract substantially.
2. The solver rejected arm joints within 1% of the frame edge, while garment residual fitting rejected points outside its 0.5% margin. A confident wrist near the top could therefore stop driving both the rig and its cuff correction. A controlled native Blender probe reproduced roughly 36 pixels of missed cuff correction when the target crossed that boundary. This is a reproduced failure mechanism, not a measurement of raw detector error in the recording.
3. Moving image fusion and the final filter could still leave both cuffs away from a strong current wrist observation. Detailed palm roots were available to foreground compositing but were not used to anchor garment wrists.
4. Joint alignment alone did not fit the visible clothing outline. The kameez also has authored open side vents, through which underlying clothing can remain visible even with correct skeletal tracking.

The saved session reported roughly 6 processed frames per second and a capture-to-apply median in the hundreds of milliseconds. These changes do not establish zero latency.

## Changes

- Female framing now letterboxes the whole camera sensor onto the same 640x360 canvas used by detection and display. The user may see side bars on a 4:3 camera. Other profiles retain the previous framing.
- Confident female elbow/wrist observations have a consistent, bounded 3% edge allowance in the solver, final stabilizer, and residual fitter. Distant or missing joints remain rejected; confidence is not raised.
- Associated current palm roots refine wrist XY. Both arms have a bounded current-observation correction during corroborated movement. Quiet, stale, weak, implausible, and unrelated-hand cases retain their guards. Corrected filter and image-reference state are synchronized.
- The existing local Holistic model provides a current person mask. Numeric torso-outline and sleeve-width samples travel with that frame's pose and RGB. Lateral arm corridors are separated from torso samples to avoid inflating the chest when an arm touches the body.
- A native live modifier expands supported cloth edges to that outline, with bounded growth and a fixed outline deadband. It runs after landmark fitting and before surface subdivision. Unsupported measurements clear their offsets.
- A matching cloth lining closes both side vents during live try-on. It samples the existing deformed boundary vertices and uses the existing side material. Source mesh coordinates, UVs, saved fitting settings, and the saved `.blend` remain unchanged.
- Coverage helpers are removed when changing outfits and disabled outside the female profile. Numeric diagnostic records include wrist-anchor usage and coverage expansion.

## Validation

- `validate_tracking.py`: 147 assertions, including 120 stationary frames with exactly unchanged pose/world output, both-arm movement correction, edge visibility, missing/stale observations, hand association, mask rejection, and full-solver raised arms. A 60-frame moving comparison kept other-profile solver output and diagnostics exactly unchanged.
- `validate_surface.py`: 27 native Blender checks across front, both side directions, lowered arms, and raised arms. The control-cage prediction agreed with Blender to substantially below 0.01 processing pixel. Controlled outline targets were covered; repeated stationary updates did not creep. Native topology checks confirmed both side vents were closed by the live lining. Authored vertices and UVs were unchanged.
- `validate_worker.py` and `validate_runtime.py`: 16 recorded-reference packets passed through the actual camera worker, socket format, solver, saved binding, rig, residual fitter, coverage modifier, and native mesh evaluation. No webcam was opened. RGB, pose, and mask evidence used matching timestamps. This is an integration test using an already composited recording, not a raw-camera replay or live accuracy measurement.
- Native rendered geometry was visually inspected in `native_surface_qa.png`.
- All 61 Python modules parsed. Hash validation confirmed the saved `.blend` was unchanged and only the intended seven existing sources plus two new source modules were affected.

The new surface calculation took approximately 4–9 ms in the headless controlled cases. The recorded-reference integration run measured about 84 ms for the worker and 106 ms for Blender application, separately. These are not live camera-to-display latency measurements.

## Scope and use

All changes, backups, and artifacts are inside `C:\Users\eisha\Downloads\Zero Change Trial`. The protected Desktop AR Garment Studio folder was not accessed or changed.

Backups: `Backups/full_sleeve_fit_20261009` contains the pre-change Python sources, diagnostic snapshots, and source/scene hashes.

Close the running Trial Blender process, reopen this folder's `START_GARMENT_STUDIO.cmd`, and use **New Person / Recalibrate** once with the new full-sensor view. A fresh live wearer recording is still needed to assess actual visual accuracy. Partially invisible limbs and uncertain segmentation cannot support a guarantee of perfect tracking or complete clothing coverage in every pose.

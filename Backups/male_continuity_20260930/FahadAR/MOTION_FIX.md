# Sleeve and shorts motion repair — 2026-09-13

Cause confirmed in both the saved Desktop blend and the open Blender session:
`Armature.data.pose_position` was `REST`. Bone rotations could be written, but
Blender did not skin clothing to those rotations. The small landmark fitting
corrections are not a replacement for armature skinning.

`ar_runtime.py` now restores `POSE` when starting AR and before applying each
tracking frame. When AR is stopped, Rest Position can still be used to inspect
the mesh. No base mesh, UV, rig weights, camera model, calibration algorithm,
or original avatar file was changed. The Desktop blend was not overwritten.

Verification against this Desktop blend:

- Reproduced zero sleeve and shorts movement in REST despite changed poses.
- Arms-out to arms-down deforms sleeves in POSE.
- Wide stance deforms both short legs in POSE.
- Independent left/right tests move only their corresponding lower hem.
- UV coordinates are unchanged; evaluated vertex positions remain finite.
- Before/after renders inspected; synthetic calibration and full-turn tests pass.

Live follow-up: the camera was started and the actual arms-out then arms-down
poses were visually checked. Both short sleeves followed the lowered arms.
The person then left the frame; a live wide-stance pose was not observed.
The calibration remained at SIDES, so a complete full-turn live test is pending.
These results are not a claim of perfect fit or cloth physics.
Full sleeves/trousers and photo-to-garment import are not implemented by this fix.

Use the existing `START_FAHAD_AR.cmd`, then complete normal calibration. Test
arms fully down, both legs comfortably apart, and each leg separately. Keep
shoulders, elbows, hips, knees and ankles in view.

Backup (before this repair):
`C:/Users/eisha/Documents/Codex/2026-09-10/yar/work/AR_motion_backup_20260913`

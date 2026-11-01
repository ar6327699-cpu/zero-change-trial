# Female kameez sleeve tracking investigation — 5 October 2026

All edits, backups and diagnostic artifacts are inside this Zero Change Trial
folder. The Desktop AR Garment Studio folder was neither opened nor edited.
The supplied video was read without modification. No camera was started and
no .blend file was saved.

## Evidence and root cause

Reviewed the supplied 90.25-second, 60-FPS screen recording, an overview across
the recording, and a closer 64–66-second arm-raise sequence. This is a screen
recording of an already composited garment, not the original unobstructed
camera input. Running pose inference on that overlay would not validate the
wearer's original arm measurements.

The copied session diagnostics match the recording: the retained 120 samples
span 12:06:09.618–12:06:26.703 Pakistan time. The saved whole-session throughput
was 6.28 processed FPS. Its last-window capture-to-apply median was 294.5 ms.
Across the retained 120 samples, worker p50/p95 was 134.4/189.6 ms, Blender apply
95.0/142.7 ms, and capture-to-apply 292.7/380.4 ms. These stage medians are not
additive and do not include a measured monitor scanout delay.

The traced path is camera capture -> Holistic -> regional image motion ->
image-supported arm fusion -> translation repair -> landmark/quiet/sleeve
locks -> regional locks -> final ArmStabilizer -> skeletal solver -> Blender
render holds and residual garment fitting -> synchronized RGB/matte commit.

Two specific problems were found in the arm handover:

1. ArmImageFollower correctly set quiet=False when current optical articulation
   was confirmed, but retained an older cumulative rigid_translation vote.
   ArmStabilizer accepted quiet OR rigid_translation as stillness. The recorded
   decisions contain 13 contradictions on each arm (26 arm samples total),
   among 40 left and 42 right active-motion samples. This selected the 0.3-Hz
   still filter and could teach real movement as noise.
2. Even after image fusion and bypassing upstream XY locks, the final sleeve
   filter low-passed those same supported moving coordinates again. This
   shifted the garment away from the camera frame sent in the same packet.

RGB and pose were already transported together, and both camera and Blender
queues already kept only the latest packet. There was no evidence that adding
prediction or another queue would resolve the confirmed filter conflict.

## Changes

- FahadAR/female_arm_image.py: explicitly records which elbow/wrist has current
  image support; active articulation supersedes both quiet and rigid hold
  votes, preserving the original votes as diagnostics.
- FahadAR/female_arm_stabilizer.py: current supported moving XY uses the fused
  observation directly and synchronizes internal filter anchors. Unknown,
  quiet, stale and unsupported observations keep the original filter. World
  depth remains filtered and observation confidence is never increased.
- FahadAR/ar_camera.py and FahadAR/ar_runtime.py: retain bounded numeric final
  arm-filter diagnostics so future live checks expose supported joints and
  residual XY lag. No camera images or landmark recordings were added.

## Verification

24 offline regression checks passed. Quiet, unknown, stale and body-translation
sequences were bit-for-bit identical to the previous filter. Checks also cover
independent arms, partial support, lost/out-of-frame joints, invalid timestamps,
reversal, settling and the combined camera-side lock chain.

Synthetic supported-arm motion at 6.3, 7, 15, 30 and 60 FPS had zero added XY
error in the final filter, compared with 10.8–14.6 pixels maximum in the
reproduced old contradictory-still scenario. This is not a claim of zero
detector error, zero rendered garment error or zero camera latency. Replaying
the recorded decision evidence changed the 26 contradictory votes to zero;
the logs do not contain the original body coordinates for a full pose replay.

A separate headless Blender process loaded this trial blend and imported the
runtime successfully. Left/right arm deformation was finite, base mesh and
UVs were unchanged, and the pose returned to its initial geometry. This
smoke test measured roughly 46–51 ms per evaluated geometry update; it is not
a live display test. Existing duplicate-addon unregister warnings appeared
at shutdown; the assertions passed and Blender exited with code 0.

All FahadAR Python files parse. A source-hash comparison confirms only the
four source files listed above changed.

## Use and remaining live verification

Open START_GARMENT_STUDIO.cmd from THIS Zero Change Trial folder in a fresh
Blender session, then start/calibrate the camera as usual. The recording's
Desktop session uses a different folder and will not receive these changes.
Do not run two camera owners simultaneously.

Live wearer verification is still needed for actual wrist/cuff alignment and
motion-to-stillness transitions. The change removes an identified additional
tracking delay; it does not establish total camera-to-display latency below
100 ms. The recorded worker alone exceeded 100 ms at its median, so that total
latency target remains unachieved/unverified by this patch.

Backups: Backups/sleeve_precision_20261005/

Checks and evidence: work/sleeve_precision_20261005/ (validate_sleeves.py,
validation.json, blender_check.py, blender_validation.json,
scope_validation.json, overview.jpg and movement_sequence.jpg).

To roll back, close the trial session and copy the four backed-up Python files
over their names in this folder's FahadAR directory. No asset rollback is
needed because scene and garment assets were not saved or modified on disk.

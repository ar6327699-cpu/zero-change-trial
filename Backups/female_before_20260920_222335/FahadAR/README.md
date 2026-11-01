# Fahad AR: guided visual calibration experiment

Double-click `outputs/START_FAHAD_AR.cmd`. It opens ONLY `fahadGenius_AR_experiment.blend` and starts the local camera. Do not launch a second copy while the first camera is running.

In Blender's 3D viewport, press N and choose **Fahad AR** (not PoseCap). Use **Stop AR Camera** to release the camera and clear the video texture. **Start AR Camera** restarts it.

Stand facing the camera, one person only, with both shoulders, hips and knees visible (preferably head to feet). Keep arms slightly away from the torso initially. The picture is not mirrored. **Existing shirt + shorts** is now the default: the avatar body and head stay hidden while its grey shirt and existing shorts overlay the real camera feed. The optional shirt + shorts + shoes mode needs visible feet too. Change modes before Start, or recalibrate after changing. Width, Length and position controls remain visual adjustments, not measurements.

## Calibration flow

1. Start AR: face front and stand still for 40 stable samples (at least 2.5 seconds). Outfit mode additionally requires at least 24 stable, recent hip/knee samples. Clothes stay hidden until this stage succeeds. Bad visibility, strong tilt or unstable proportions delay calibration.
2. Turn slowly through side, back, other side and front until both side counters and the back counter reach 16. A provisional outfit is shown. **Calibrate back view** is enabled by default; disabling it before Start skips the back checkpoint. The system accepts wrapped headings throughout a 360-degree turn, but a single camera can still mislabel or lose hidden joints. Completion records view coverage, not a guaranteed back-view scan or accurate hidden body depth.
3. Face front again. Review the shirt and adjust Chest, Waist, Depth (side fit) and Sleeve length if necessary, then click **Confirm Visual Fit**.
4. For another person, click **New Person / Recalibrate**. This is a one-person system, without automatic identity recognition.

Shoulder-to-torso proportion is automatically locked from multiple frames. Chest, waist and depth are NOT exact automatic measurements: a loose or open shirt hides those dimensions. The system intentionally does not use the outer clothing silhouette as the person's naked-body outline. The regional controls are visual shape-key trims, not centimetres. It cannot guarantee an edge-to-edge physical fit or determine whether a real retail size fits. No full-body anthropometric scan is implemented.

The side projection now uses visible same-side shoulder/hip height, rather than dividing by shoulder width that vanishes at 90 degrees. Losing all reliable torso anchors still hides the garment instead of pretending a stale pose is valid.

This is a free, local prototype using the already installed Blender and MediaPipe runtime/model. No camera frames are sent to an external service or saved by the AR scripts. Numerical session diagnostics go to `last_session.json` when stopped; engine messages go to `runtime.log`. Do not save the Blender file while live camera pixels are present if you want to avoid embedding a camera frame. Stop first, then save only this AR copy if desired.

## Current limitations

- Calibrated visual overlay, not realistic cloth simulation or accurate size/fit prediction.
- Exposed hands/forearms and the face/neck have a same-frame foreground matte, constrained by landmarks and estimated depth. Smooth mode uses adaptive skin colour; Semantic mode uses a heavier segmentation model. Fast motion, overlapping limbs, gloves, long sleeves and low-resolution hands can still give imperfect edges or ordering. This is not a depth-camera guarantee.
- The real shirt is not removed; edges can show around the virtual garment.
- Side-angle projection has synthetic tests; live turn quality must also be checked. Back views, deep bends and hidden body shape are not guaranteed.
- Camera angle, light and tracking latency affect alignment.

The original `fahadGenius_posecap_v3_checked.blend`, its meshes, installed addon and `FahadLive/fahad_backend.py` were not changed. Keep the sibling `FahadLive` folder: the AR producer imports its solver read-only. Protection hashes are in `source_protection.json`. Offline projection checks are in `offline_checks.json`; those are not a physical-fit guarantee.

Calibration-version tests are in `calibration_checks.json`. The previous front-only runtime was backed up under `work/FahadAR_before_calibration` before these edits.

## Realism / foreground update

**Real hands / neck in front** toggles the local skin matte for A/B inspection. The matte is synchronized with the same RGB frame and pose. It reveals actual camera pixels in front of the virtual shirt; it does not create synthetic skin. Face and neck are limited to a narrow head/neck region, so an open shirt's bare chest is not deliberately restored through the garment. Mask errors remain possible.

**Lower neckline** lowers only the central upper-neck garment vertices with a smooth falloff; it does not move the entire shirt or modify original UVs. Default is now 0 to undo the excessive automatic lowering reported during live use.

**Mode (restart)** defaults to Smooth: it estimates skin colour from the currently visible face and gates it with current limb/depth landmarks. Skin-coloured fabric and changing light can confuse this faster mask. Semantic mode retains the official MediaPipe SelfieMulticlass model (see `models/SOURCE.md`) running alongside pose inference, at additional CPU cost. Stop and Start AR after changing mode.

**Relax shoulder caps** now defaults to 0.25 instead of 0.70, retaining more of the original shoulder/neck coverage. It does not move shoulder joints or change UVs. Set 0 for the original cap shape, or adjust visually; this is not a measured body contour.

## Regional outfit fitting (version 2)

Global fitting pins to the observed shoulder line rather than averaging shoulders and hips into the translation. A post-armature Geometry Nodes modifier then blends current shoulder/elbow/hip residuals across the shirt and independent hip/knee residuals across the shorts. Original mesh Basis and UVs are unchanged; this is a reversible AR-copy-only correction. Missing or excessive residuals are rejected, with nearby current proximal anchors used for occluded regions. The original 3D garment and rig retain the turn/depth appearance; this correction only changes projected alignment.

Shorts are hidden when no reliable hip-to-knee pair is visible. They are not shown at a stale position while the lower body is off camera. The real person's existing trousers/shorts are not removed, and the virtual shorts cannot conceal real fabric extending beyond their hem. Hidden-limb pose and fast motion remain uncertain.

Tests: `outfit_validation.json` contains 13 synthetic turn checkpoints, unchanged UV checks, independent left-knee deformation and missing-lower-body visibility checks. Front, back and both side renders were inspected. These are synthetic tests, not evidence of physical sizing or verified real-person 360-degree performance. Backup: `work/FahadAR_before_full_outfit_v2`.

The camera is continuously drained into a one-frame queue to avoid accumulating old frames during inference. Unchanged shape keys are not rewritten each frame, and the projection reuses one evaluated skeleton. These changes reduce avoidable work; they cannot make CPU inference instantaneous. Numerical performance is written to `performance_live.json` every 30 applied frames, including capture-to-apply time (not total screen/display latency).

No camera image/matte is uploaded or saved by the runtime. Stop clears BOTH live textures before saving. The material is still a rendered garment: correct lighting, natural cloth motion and complete removal of the person's old clothing are not solved by foreground masking alone.

Backup before the shoulder/smooth update: `work/FahadAR_before_smooth_shoulders`.

Backup before this update: `work/FahadAR_before_occlusion`. Unit checks are in `occlusion_unit_checks.json`; the synthetic Blender composite render is `work/occlusion_composite_test.png`.

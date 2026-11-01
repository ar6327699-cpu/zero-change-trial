# Phase 1 — Male Eastern foundation

Date: 2026-09-19. Scope: KURTA and SHALWAR only. No Phase 2 garments and no Female support.

## Result

**PASS for the implemented foundation and controlled Blender validation described below.**
**Live-camera wearer acceptance: NOT TESTED in this phase.** This is not a claim of perfect tailoring, physical cloth simulation, or a release certification for all users/poses.

The existing project was extended, not rewritten. No AI model was added. Camera, tracking solver, calibration, person-fit/fit-lock and occlusion algorithms were not replaced.

## Backup and preservation

Rollback snapshot: `C:/Users/eisha/OneDrive/Desktop/AR Garment Studio/Backups/phase1_before_20260919`.

It contains the main blend, FahadAR, FahadLive, garment library and launcher as they existed before implementation. No test saved over the main blend. Test-built garments were stored in the separate workspace test library, NOT the user's garment library.

Verified unchanged main blend SHA256: `cd0ee03988e924ca79d2b42c342097a7dfb2e2cc04851d605fad29213f429603`.

Verified unchanged library index SHA256: `b062bb7d5ab9b32fbe2060f3d6a8c8cc5ffb9f5154ce36aa9cd9ec3330a35917`.

All existing files under Garments were byte-compared with the backup. Existing templates in the blend were not regenerated or overwritten. New categories are appended to the enum so existing saved category indices retain their meanings.

## Files inspected

Under `C:/Users/eisha/OneDrive/Desktop/AR Garment Studio/FahadAR`: garment_templates.py, garment_studio.py, garment_fit.py, garment_photos.py, waist_layer.py, ar_runtime.py, studio_bootstrap.py, calibration.py (relevant fitting sections). Also inspected the garment library manifest, Blender rig/bone coordinates and mesh bounds, and prior movement/waist test scripts in the workspace.

## Production files changed

All paths below are under `C:/Users/eisha/OneDrive/Desktop/AR Garment Studio/FahadAR`.

| File | Exact responsibility/change |
| --- | --- |
| eastern_foundation.py — new | Isolated Eastern category definitions, Kurta lower-panel construction and full sleeves, independent Shalwar pattern construction, safe group-schema compatibility, and long-upper landmark weights. |
| garment_templates.py | Appends KURTA/SHALWAR and dispatches only those categories to the new builders. Existing five Western builders remain on their original paths. |
| garment_studio.py | Allows the extra Eastern pelvis/upper-leg bone groups safely during replacement; uses existing library/photo/preview/activation workflow; adds flat-lay and whole-body guidance for Eastern categories. |
| garment_fit.py | Kurta uses wrist anchors plus smoothly blended hip-to-knee lower-panel anchors. Shalwar includes ankle anchors. Western weight calculation paths are unchanged. |
| ar_runtime.py | Kurta requests existing long-sleeve foreground handling. Shalwar uses existing full-length lower visibility rules. Kurta sleeve trim excludes the skirt region. |
| waist_layer.py | Kurta bypasses the short-shirt waistband shrinkwrap rule, which would otherwise pull Shalwar thighs into the long hem. Returning to a Western top re-enables the original rule. |

No changes to ar_camera.py, calibration.py, occlusion.py, latest_camera.py, product_photo.py, garment_photos.py, studio_bootstrap.py or the existing tracking solver were required. This report is an additional documentation file.

## Architecture and geometry

### Kurta

The proven shoulder/chest and arm skinning is reused, but the garment is not merely renamed: the original lower shirt is cut away above the hip and replaced with fourteen graded rings extending to the knee region. The new section includes hip/seat ease and side openings for movement. Skin weights transition from torso/pelvis to the upper legs; knees are not used as direct skirt skinning bones. Both arms extend to wrist-length sleeves using the existing sleeve construction.

New lower-panel geometry, extra rig groups and extended landmark fitting distinguish this from FULL_SHIRT. Broad hip ease was added after an initial visual test exposed Shalwar through the first narrow Kurta prototype.

### Shalwar

A new mesh pattern is generated independently of the Western pants: gathered waistband, loose seat, lowered connected crotch saddle, separate loose thigh tubes, knee transitions and tapered ankle cuffs. Crotch vertices are shared and pelvis-weighted; left/right thigh and knee skinning is independent. It is not the Western pant scaled sideways.

### Reused systems

Both garments occupy the established Tops/Bottoms slots, use the same Mixamo armature and existing library asset format, and use existing packed front/back image materials and UV projection. No additional garment slot or alternative AR runtime was introduced. Native depth and deliberately generous Kurta shell geometry provide the foundation's long-upper layering; this is not a general collision solver.

## Validation performed

Blender 5.1.1 evaluated the actual meshes, modifiers, pose writer and fitting code. Visual renders used Cycles CPU, 8 samples and two threads to avoid additional GPU stress. Test scripts never save the main project.

1. Captured a pre-change baseline for Half Shirt, Full Shirt, Pants and Trousers. Each was evaluated in seven existing tracked-pose fixtures: arms out, arms down, wide stance, left leg only, right leg only, bent elbows and bent knees. Fitting was applied as well as armature motion.
2. Repeated those tests after implementation. Base coordinates, UVs and evaluated vertex arrays matched the original hashes exactly for all four Western categories and all seven fixtures.
3. Evaluated Kurta and Shalwar in the same seven movement fixtures; all evaluated coordinates were finite.
4. Rendered and visually inspected front, side and back views, then arms-out, wide-stance and bent-knee views for all six requested garment types. Also inspected Kurta + Shalwar together. These are controlled pose renders, not a recording of a live wearer turning in front of the camera.
5. Used distinct blue-front/orange-back grid images through the actual photo importer to inspect Eastern front/back mapping. Existing Western test library textures were retained in Western renders.
6. Built new Eastern assets into an isolated test library, saved/reloaded their packed blend assets, rebound the original rig, activated and previewed them, then restored the original Western outfit. Checked normalized bone weights, finite UVs and exact restored original base coordinates/group schema.
7. Launched the actual Blender Studio UI using the existing bootstrap. Visually verified both new categories and their flat-lay controls. Restored the previously selected Half Shirt option. Did not activate a test outfit, start the webcam, or save the UI session.

## Results by garment

| Garment | Controlled visual result | Regression/functional result |
| --- | --- | --- |
| Half Shirt | Front/side/back and arm/lower-body poses inspected | PASS — exact baseline geometry/UV/posed output |
| Full Shirt | Full sleeves and views/poses inspected | PASS — exact baseline geometry/UV/posed output |
| Pants | Leg separation and knee bending inspected | PASS — exact baseline geometry/UV/posed output |
| Trousers | Views, independent leg motion and knee bending inspected | PASS — exact baseline geometry/UV/posed output |
| Kurta | Long hem, full sleeves, side openings and lower-panel movement visible | PASS for foundation tests; limited skinning-based drape |
| Shalwar | Loose seat, shared crotch, distinct thigh shape and tapered cuffs visible | PASS for foundation tests; no physical cloth folds |
| Kurta + Shalwar | Standing/front/back layering corrected; wide/bent poses inspected | PASS for inspected fixtures; not arbitrary-pose collision certification |

## Evidence and reproducibility

Workspace: `C:/Users/eisha/Documents/Codex/2026-09-10/yar/work`.

- phase1_probe.py: read-only rig geometry probe.
- phase1_validate.py: numerical baseline and after-change evaluation.
- phase1_visual.py: actual Blender render sheets.
- phase1_roundtrip.py: photo/library/rebind/restore and preservation checks.
- phase1_results/baseline.json and after.json: regression snapshots.
- phase1_results/roundtrip.json: round-trip results.
- phase1_results/angles.png: front, side, back sheet.
- phase1_results/motion.png: arms, wide stance, bent knees sheet.

These scripts are test tooling, not new runtime dependencies. The original unrelated add-ons emit duplicate-registration/unregister warnings in background Blender; test completion is determined by explicit assertions/results, not process exit code alone.

## How to use

1. Open the existing `START_GARMENT_STUDIO.cmd` launcher. The new templates are generated in memory on startup; the original blend does not need to be overwritten for installation.
2. N sidebar > Garments > Template: choose Kurta or Shalwar.
3. Supply front and back flat-lay photos, whole garment visible, plain contrasting background. Kurta sleeves should be spread. Phase 1 does not introduce Eastern product-photo/arms-down reconstruction.
4. Build photo garment, inspect Preview front/back/sides, then Activate selected for AR.
5. Activate a top and a lower garment separately. Stop the camera before switching. Use the existing AR calibration with hips, knees and ankles visible.

The user's existing active outfit and original garment library have not been replaced by QA assets.

## Remaining issues / limits

- No live-camera wearer acceptance test was performed. Hand occlusion, real-world tracking during turns, fit-lock stability with the new silhouettes, and real-time latency still need live verification. Unchanged code is not a substitute for that test.
- Long cloth is bone-skinned, not simulated. Deep squats/extreme stretches produce a comparatively stiff/flared hem. Arbitrary poses and garment proportions can still intersect; no universal cloth-collision guarantee is made.
- Fit remains guided visual fitting, not exact body measurement or tailoring. Loose clothing can hide true body dimensions.
- Eastern textures currently require flat-lay front/back photos. These tests used diagnostic grids, not brand-specific Kurta/Shalwar photographs.
- No Female garments, Kameez, Pajama, Eastern Trouser or Waistcoat were added.

## Rollback

Close Blender without overwriting the protected blend. Restore the five changed existing Python files listed above from `Backups/phase1_before_20260919/FahadAR` to `FahadAR`. The new eastern_foundation.py is then unused and can be retained for reference. No library/blend rollback is needed unless the user subsequently saved or built new garments. Keep the backup intact; do not replace newer user assets blindly.

**STOP: Phase 1 ends here. No Phase 2 work is authorized by this report.**

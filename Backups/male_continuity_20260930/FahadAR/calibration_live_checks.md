# Calibration-version check, 2026-09-13

The updated runtime was loaded into the actual interactive AR Blender copy and used with the real camera.

- FRONT advanced to SIDES after stable samples.
- SIDES advanced to REVIEW after the user's two side counters completed. This requires at least 16 qualifying samples at each signed heading between 60 and 115 degrees.
- The Confirm Visual Fit control advanced REVIEW to LIVE.
- Front-facing shirt overlay remained visible while the person moved closer/farther and moved their arms. Example status: 2 px joint-projection error and 153 ms capture-to-apply. This is not garment-edge or centimetre accuracy.
- A held exact 90-degree side screenshot was not yet verified at the time of this note; stage completion is tracking evidence, not a guarantee of perfect side appearance.
- Hand-over-shirt occlusion and removal of the person's actual shirt remain unimplemented.
- Chest/waist/depth/sleeves are visual adjustment controls; hidden body contours under loose/open clothing are not measured automatically. No full-body measurement accuracy is claimed.

Additional tests: 41 synthetic projection angles (-100 through +100 degrees), 10 Blender pose cases with finite shape-key geometry and unchanged basis/UVs, and a replay of 811 historical real landmark frames. In that recording's 25 >=60-degree profile frames, valid AR placement improved from 3/25 to 25/25. See `calibration_checks.json` and `side_replay_checks.json`.

The AR copy was saved with default calibration state and a cleared camera image. Live camera imagery was not embedded in that save. Original avatar SHA256 and original backend SHA256 still match the protection record. The interactive preview remains running independently of that default saved state.

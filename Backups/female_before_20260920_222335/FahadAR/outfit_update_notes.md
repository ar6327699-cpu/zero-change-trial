# Outfit / calibration update — 2026-09-13

The original avatar file and original FahadLive backend remain unchanged (verified SHA256).

Delivered on the separate AR copy:

- Existing shirt + existing cargo shorts selected by default. Shoes remain an optional full-outfit mode; no new trousers were invented.
- Undid excessive default neckline lowering (0.45 → 0) and reduced cap lowering (0.70 → 0.25).
- Shoulder-pinned global alignment plus smooth native Geometry Nodes corrections based on current shoulder, elbow, hip and knee positions. Left/right shorts legs have separate regional weights. Mesh Basis, UVs and original files are preserved.
- Front calibration requires stable shoulder/hip proportions and stable recent hip/knee samples for outfit mode. A long interruption discards old front samples. This does not estimate hidden circumferences or physical clothing sizes.
- Side and back coverage checkpoints accept a continuous 360-degree turn. Lost/ambiguous landmarks remain a limitation; coverage counts do not establish an accurate body scan.
- Off-frame lower body hides shorts rather than displaying stale lower-body placement.

Verification:

- 13 synthetic pose/Blender checkpoints through 0–360 degrees, including inspected front/back/side renders.
- Independent left-knee correction, finite evaluated mesh vertices, unchanged UVs, missing-knee visibility guard and bounded edge stretching under the tested synthetic inputs.
- Front/side/back calibration completes with valid synthetic input; missing knees prevent outfit calibration.
- Actual updated camera started in Blender with the correct outfit and defaults. It showed the room ceiling/wall, with zero valid body-fit frames; user-specific neck/shoulder fit and real 360-degree turning have NOT been verified in this session.

Use `START_FAHAD_AR.cmd` for the latest runtime. Frame head-to-feet, face front until stable calibration, turn slowly through both sides and back, return to front and confirm visual fit. Real clothing outside the virtual shirt/shorts boundaries remains visible. This is visual AR fitting, not guaranteed accurate tailoring or realistic fabric simulation.

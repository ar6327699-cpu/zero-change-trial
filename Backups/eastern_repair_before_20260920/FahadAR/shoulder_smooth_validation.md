# Shoulder and smooth-mode update — 2026-09-13

Implemented in the AR copy only. Original avatar and FahadLive backend SHA256 hashes match source_protection.json.

## Changes

- Regional shoulder-cap shape key, default 0.70: lowers the source shirt's raised caps without translating the shoulder skeleton. Original Basis coordinates and UVs are preserved.
- Latest-frame camera queue (capacity one), camera request 640x480 and 640x360 synchronized output.
- Smooth mode: face-adaptive skin colour at half resolution plus full-resolution current landmark/depth gating. Semantic neural matte remains selectable, with a restart.
- Float32 matte composition removes unnecessary float64 frame intermediates.
- No repeated writes to unchanged shape keys; one skeleton evaluation reused for fitting.
- Blender timer poll interval reduced from 30 ms to 5 ms. Inference and rendering still impose latency.

## Verified

- Latest-frame fake-camera regression: drops queued old frames and stops its thread.
- 41 ideal projection yaw cases from -100 to +100 degrees; front/side/review calibration flow and invalid-input rejection.
- Shoulder weights 0, 0.70 and 1: zero reversed rest-mesh triangles, finite coordinates, unchanged UVs. At 0.70 minimum triangle area ratio was 0.329 relative to the original; at 1 it was 0.041, so the default avoids the most compressed endpoint.
- Inspected front, left 90-degree and right 90-degree synthetic Blender renders. Shoulder-cap height visibly reduced. These are not images of a fitted person.
- Synthetic front forearms produce a matte, behind arms do not, and reset/no face clears the adaptive matte.
- Float32 composition differed from the previous formula by at most 5.92e-8 in a randomized regression.
- Latest synthetic fast-matte timing: approximately 10 ms median on this machine. Not total live latency.

## Live session limitation

The updated initial smooth pipeline applied 1682 camera frames, but no valid shoulder/hip calibration was obtained: the camera showed the room/obscured view rather than a clearly framed person. Median no-person inference was 42.6 ms and Blender apply was 32.2 ms. This session preceded the final 5-ms poll and float32/half-resolution matte refinements. It is NOT a valid before/after person benchmark; it does NOT verify shoulder alignment, cross-arm masking or natural fit on the user. Camera was stopped and live textures cleared.

Pending: a clearly framed person facing front, left/right profiles, hands to chest, crossed arms and return to neutral. Check slipping, neck/shoulder edges, occlusion and capture-to-apply timing under the actual tracked-body workload. Full display latency and realistic cloth behaviour remain unmeasured/unsolved.

Smooth mode is a speed/edge-quality tradeoff: skin-coloured clothes and lighting changes can confuse the colour mask. This prototype does not remove the real shirt, estimate hidden body measurements, simulate real fabric, or guarantee a retail fit.

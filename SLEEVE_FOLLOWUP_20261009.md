# Female sleeve follow-up — 9 October 2026

Changes are confined to Zero Change Trial. The Desktop AR Garment Studio
folder was not accessed. No camera was started or scene/garment asset saved.

## Recording evidence

Reviewed the supplied 33.18-second screen recording, an overview across the
clip, full-resolution frames, and a closer sleeve sequence at 8.4–10.6 seconds.
The Blender title confirms this recording uses the trial folder.

The matching retained diagnostics run from 18:41:42.820 to 18:42:05.307 Pakistan
time. The previous fix was operating: all 58 fully supported left-arm samples
and 52 fully supported right-arm samples had zero added XY lag in the final
arm filter. Other samples reached 4.98 and 4.04 camera pixels respectively.
That diagnostic measures the filter's difference from its input, not absolute
wearer/garment accuracy.

Remaining problems traced through the source:

1. After image fusion corrected a moving arm, the quiet/non-moving branch
   discarded that correction and returned raw detector coordinates. A bounded
   detector offset could therefore reappear as the regional stillness locks
   acquired their next anchor. This was reproduced in the combined filter and
   lock chain, including independent left/right cases.
2. When image patches were unavailable during fast motion, the fallback's
   backward-Euler low-pass response retained around 10% of a large displacement
   at the recorded low update rate, even at its maximum motion cutoff.

## Fix

- `FahadAR/female_arm_image.py` now carries previously followed joints into
  quiet mode using the current verified optical step. Existing regional locks
  acquire that corrected position. A quiet start retains the previous behavior.
  Missing support, invalid time gaps, large innovation, shoulder disagreement
  and depth changes still disable this handover; no missing joint is invented.
- `FahadAR/female_arm_stabilizer.py` blends toward the exact exponential filter
  response only for fast motion, starting above 45 px/s of speed excess over
  the learned noise floor and completing at 90 px/s. Quiet and low-speed paths
  retain their prior behavior. Depth filtering and confidence remain unchanged.

Only those two production source files changed, verified against backup hashes.
The earlier confirmed-motion bypass remains in force.

## Checks

22 new regression checks passed. A reproduced stop/handover detector conflict
shifted the previous output by 3.50 pixels on either arm; the revised output
shifted by zero. Synthetic fast-motion fallback mean XY error at 6.3 FPS fell
from 2.31 to 0.98 pixels. It improved at all tested rates (5, 6.3, 7, 15, 30,
60 FPS), but remaining error is nonzero, especially around unsupported reversals.

24 prior regression checks also passed in the follow-up copy of that suite.
One expectation was intentionally updated: fast unsupported wrists now use
the revised filtered fallback, rather than requiring the old numerical filter
output bit-for-bit. The check compares against the new fallback with direct
joint following disabled and verifies that unsupported wrists are not passed
through directly. The original October 5 test script is preserved.

Quiet, low-speed unknown/stale, and tested body-translation sequences stayed
bit-for-bit identical. Checks cover both arms, settling, partial support,
missing joints, discontinuities, depth changes, and upstream outlier rejection.
All FahadAR Python sources parse successfully.

These are offline synthetic/decision checks. The recording already contains a
virtual overlay and is not a raw-camera replay. Actual post-fix wearer alignment
and total display latency remain unverified. The supplied session's retained
capture-to-apply median was 367.8 ms; this change does not claim that the whole
system now runs below 100 ms.

## Run / evidence / rollback

Start a fresh trial Blender session using this folder's
`START_GARMENT_STUDIO.cmd`, then start and calibrate the camera as usual. Existing
Blender Python modules need a fresh session to load these changes.

Evidence and executable tests: `work/sleeve_followup_20261009/`.
Results: `validation.json`, `previous_regressions.json`,
`recording_analysis.json`, and `scope_validation.json` in that directory.

Backup: `Backups/sleeve_followup_20261009/`. It includes both previous source
files and the original recording-session diagnostics. To roll back, close the
trial session and restore those two Python files to its FahadAR directory.

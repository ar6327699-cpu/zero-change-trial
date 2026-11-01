# Wearer's left sleeve response — 9 October 2026

## Scope

Inspected `C:\Users\eisha\Videos\2026-10-09 19-09-27.mp4` (46.25 seconds), including the movement/settling transitions, and the retained tracking diagnostics from the same session. Left means the wearer's own left arm, appearing on the right of the unmirrored camera image.

All edits, backups, and inspection artifacts are inside `C:\Users\eisha\Downloads\Zero Change Trial`. The Desktop AR Garment Studio folder was not accessed or changed.

## Findings

The two arms already used the same thresholds. Their image evidence was different: among 120 retained samples, the left arm had 9 partial-image-fusion samples versus 1 on the right, and 59 fully supported samples versus 74 on the right.

When the left elbow had confirmed image motion but its forearm image fit was unavailable, the elbow followed the current frame directly while the visible wrist remained filtered relative to the shoulder. This delayed the wrist's shared movement with its elbow, producing a temporary mismatch. The final-filter diagnostic reached 4.71 pixels on the left versus 2.26 on the right, measured in the 640 by 360 processing image. These diagnostics measure filter displacement, not absolute tracking accuracy against a ground-truth wrist.

A left-side inferred world-position jump also interrupted image following even when the current image evidence could independently corroborate the movement.

## Changes

- `FahadAR/female_arm_stabilizer.py`: when only the left elbow has fresh confirmed image-follow support and both left joints remain visible, carry the wrist filter's history with its own elbow displacement. Continue filtering the wrist's independent articulation. Existing quiet, missing-joint, stale-evidence, and fully supported paths retain their behavior.
- `FahadAR/female_arm_image.py`: retain left image following across an inferred world-position jump only if both segment fits are valid, the existing segment/shoulder agreement checks pass, and both current detector positions agree with the image prediction within 3 pixels. World coordinates remain filtered; confidence is not raised.

Both arms continue to move independently. The right arm's numerical response is preserved. No garment geometry, sleeve fitting, scene, or saved profile was changed in this update.

## Verification

`work/left_arm_match_20261009/validate_left_match.py` passed 32 checks:

- Reproduced common elbow/wrist travel at 5, 6.3, 7, 15, 30, and 60 processed frames per second. The repaired left response matched the fully supported right response in this controlled scenario.
- Confirmed quiet, stale/missing evidence, joint loss, frame boundaries, independent wrist articulation, and depth-guard rejection cases.
- Compared 250 mixed frames: right pose output and right diagnostics were exactly unchanged.
- Exercised the complete camera lock/filter sequence over 120 generated frames, including partial forearm support followed by stillness. The reproduced mean left wrist filter displacement fell from 1.96 pixels to numerical rounding error; the settled output did not continue drifting.
- Parsed all 53 tracked Python source files. Hash comparison confirmed only the two listed sources changed. The saved Blender scene hash is unchanged, preserving the earlier cuff fitting.

Results: `work/left_arm_match_20261009/validation.json` and `scope_validation.json`.

The movement checks use generated landmarks and image evidence. A new live wearer recording has not yet verified the resulting visual alignment. This does not establish zero camera-to-display latency or perfect tracking during occlusion.

## Use and backup

Close the running Trial application and reopen it using this folder's `START_GARMENT_STUDIO.cmd` to load the updated Python modules.

Pre-change source files, retained diagnostics, and hashes are saved in `Backups/left_arm_match_20261009`.

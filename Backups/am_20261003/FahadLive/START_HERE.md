# Fahad Live — use the repaired v3 file

File: `fahadGenius_posecap_v3_checked.blend` in the parent outputs folder.
It was repaired in place. `fahadGenius_v3_before_live_fix.blend` is the unchanged backup.
The Desktop original was not changed by this repair.

1. Stop any old PoseCap stream. Reopen the repaired v3 from disk. Do not save an older open Blender session over it.
2. In PoseCap, target `Armature`. Pose Backend must be **Fahad Live - Calibrated Mixamo + Fingers**, not Automatic or Aaina.
3. Start Stream. Look straight forward and hold your head still until the camera preview says `HEAD calibrated` (20 stable detected face frames, usually 2–3 seconds).
4. To reset neutral, focus the camera preview and press **C**, then look straight and hold still again.
5. Keep your whole body, including both feet, in view. `left_leg:OK` and `right_leg:OK` mean the landmarks pass visibility checks. Lift one leg at a time.
6. Keep hands clearly visible and well lit. Bring them closer for fists. `NOT SEEN` is a tracking limitation, not a disabled rig joint.
7. ESC in the camera preview ends capture. Stop Stream in Blender before starting another camera app.

Keep this `FahadLive` folder in place: the registered backend points to its Python script. It uses your already installed local MediaPipe runtime and model. No video or photos are uploaded. The verification captures saved numerical landmarks only, not camera video.

## Changes and checks

- Explicit 52-joint mapping: 22 body joints plus all 30 finger joints.
- Independent legs with consistent parent-local compensation; pelvis pitch/roll remain disabled. Side turns rotate the pelvis around the vertical axis.
- Face-based neutral calibration removes the constant downward head offset.
- Measured finger flexion uses the actual palm plane; thumb opposition follows hand landmarks.
- Palm orientation also drives forearm roll (turning palm up/down) and wrist bend, preserving the elbow-to-wrist direction.
- Invalid feet/hands briefly hold, then ease to neutral. A single smoothing stage avoids doubled filter lag.
- 511 camera frames streamed to Blender. Median measured capture-to-apply time: 67 ms; this does not include unknown camera sensor buffering.
- Final hand-axis correction replayed all 511 captured frames; open/fist renders inspected for both hands. This last correction was verified by replay, not a fresh live stream.
- Meshes, materials, UVs and packed textures were preserved in this repair.

## Remaining limits

This is not a fully collision-aware virtual fitting system. Small/hidden hands may lose tracking; body-touching poses can intersect clothing; root translation/ground contact are not solved. The head has no expression blendshapes and garments use skinning, not physical cloth fitting. Do not interpret the repairs as an all-poses/perfect-fit guarantee.

Exact verification details are in `verification.json` and inside the blend's `Fahad Live - Verification` text block.

## Interactive-session correction

The user's actual Blender registry initially exposed only MediaPipe Lite, even though the repaired blend and companion script existed. `connect_in_blender.py` registered Fahad Live from inside that running Blender process and reselected it. The user then confirmed the face and legs were correct. The additional forearm/wrist update is covered by `twist_verification.json`: 14 palm-rotation tests and 511-frame replay, with zero changes to head/legs/torso/shoulders or finger curls.

## Side-profile repair — September 11

The current solver is `side-profile-1`, shown in the camera preview. Stand facing the camera for calibration before turning.

- Full side turns use world depth instead of the collapsing projected shoulder width. Hips, chest and feet turn together.
- Hidden arms briefly hold their body-local pose, then lower beside the torso instead of jumping to a T pose. A lost face follows the body after a short hold.
- The opt-in restart patch uses this armature's verified rest pose when selecting the PoseCap writer. Stopping with bent arms no longer selects a different retargeting mode on the next Start. Other characters are not opted in. An addon reinstall/update can overwrite this patch.
- Ten synthetic side cases, six isolated leg-lift cases and 511 previous camera frames were checked. The calibrated front head/legs and finger curls were preserved in those comparisons.
- The second new capture contained 811 frames; 787 were actually applied in Blender. All 787 matched replayed bone rotations exactly. Observed side frames near -89 and +86 degrees were rendered and inspected, with meshes and textures intact in those samples.
- Test coverage is **not** an all-poses guarantee: people entering the camera view and incomplete framing affected both live runs. Use one visible person. Unseen joints cannot be measured reliably with this single-camera tracker. Cloth/body contact is still not collision simulated.

The `side_*verification.json` files record these checks. The working blend is still the same v3 file; timestamped `fahadGenius_before_side_*.blend` files preserve the earlier scene. Keep the companion folder beside the blend.

# Hand and side-pose occlusion repair

Only FahadAR/occlusion.py was replaced. The previous source is occlusion.py in this folder.

Current hand landmarks no longer require a visible elbow. Depth ordering is checked at the hand/forearm position on the torso. A visible current palm supplies a Smooth-mode skin colour reference when the face disappears in profile. Full sleeves still cover forearms. No hand masks are carried between frames.

The saved Blender scene, garment geometry/UVs, waist layer, calibration and side-turn solver are unchanged. Restart AR Camera to load the new camera-worker code. For each new wearer use New Person / Recalibrate and complete front, both sides and the enabled back view.

Verification includes synthetic regression cases and local semantic-model inference. Live wearer edge accuracy still requires a camera check; this is not a physical garment-fit guarantee.

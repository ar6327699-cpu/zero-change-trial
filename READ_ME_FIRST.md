# AR Garment Studio — isolated prototype

Your working **AR Baised** folder is unchanged. This is a separate project and
runtime. You can always use the old folder and old launcher exactly as before.

## Start / shuru kaise karna hai

1. Double-click **START_GARMENT_STUDIO.cmd** in this folder. Open the Blender
   **N sidebar > Garments** tab. No numpad is needed.
2. Enter a garment name and choose a template: half-sleeve shirt, full-sleeve
   shirt, shorts, tapered pants, or relaxed trousers.
3. Choose Front photo and Back photo using the file buttons.
   For the supplied black T-shirt choose **Product photo**, **Crew / round**
   neckline and **Slim torso = 0.08**. The already-corrected library item is
   **Fahad black crew - corrected**; you do not need to rebuild it.
4. Click **Build photo garment**. This creates a saved library item, without
   replacing your current shirt or shorts.
5. Choose **Preview selected**. Front/Back/Left/Right buttons let you inspect it.
6. Click **Activate selected for AR** only when you approve the preview. A top
   replaces only the top; bottoms replace only the bottoms. Build/select both
   separately for a complete outfit.
7. In **Fahad AR**, click **Start AR Camera** and complete wearer calibration.
   Close/stop the camera in the old Blender first; two sessions cannot share it.
8. To change an outfit, stop the camera, activate another library item, then
   start/calibrate again. For another customer use **New Person / Recalibrate**.
9. **Restore original shirt + shorts** restores the original meshes/materials
   inside Studio. The old AR project is never involved.

Library garments and their packed images save automatically under **Garments**.
Use Blender **Ctrl+S** to remember the current active outfit and scene settings.
The supplied DEMO items are diagnostic coloured textures, not real products.

## Photos / tasveerain kaisi hon

For shirts there are now two photo layouts:

- **Product photo**: front/back product views with sleeves angled down. This
  estimates the torso centre separately from sleeves, preserving central logo
  placement. Crew/V neckline selection changes the new shirt geometry. AUTO
  neckline detection is only a heuristic; inspect and override when necessary.
  Slim torso adjusts only the new garment, not the tracking skeleton.
- **Flat lay**: the original importer for spread-out garments. The guidance
  below applies to this layout. Bottoms currently use this original method.

For either layout, keep the entire garment visible. Cropped cuffs, dark fabric
against a similar background, shadows and perspective reduce reconstruction
quality. Product pictures are not sewing patterns. The supplied cropped black
T-shirt has been visually reviewed, but unseen cuff details remain approximate.

- Photograph ONE garment flat, directly from above, front and back separately.
- Spread sleeves sideways, matching a T-pose. Keep trousers straight and legs
  separately visible. Do not use folded garments, model-worn photos, hangers
  covering fabric, collages, screenshots, or perspective-heavy photos.
- Keep the whole garment in frame, with a plain contrasting background and
  soft, even lighting. Use similar scale/orientation in the two photos.
- Transparent-background PNGs are supported. JPEG/PNG/WebP/TIFF are accepted.
- Auto crop is conservative. Adjust Background tolerance if needed. If it
  rejects a difficult background, manually crop/isolate the garment and disable
  Auto crop. Never interpret a successful crop as a guarantee of correct shape.
- Back photo viewed from behind is enabled by default. Disable only if the
  provided back image was already mirrored. Preview text/logos from both sides.

## What the automation does

- Uses your existing shirt/shorts as the base rigged templates.
- Adds weighted sleeve/leg extensions for full-length categories.
- Prepares front/back images locally and projects them onto template UVs.
- Packs textures into reusable garment library assets.
- Reuses the existing camera tracking and calibration algorithm unchanged.
- Adds wrist/ankle landmark correction only for long garments. Full sleeves
  retain real hands in front without cutting the bare forearm through cloth.
- Refuses garment swaps during a running camera session.
- Automatically rebuilds **AR Waist Under Shirt** when either garment is
  activated and when AR starts. This keeps the overlapping waistband inside
  the shirt after pose/landmark fitting, with a smooth taper below the hem.
  It is an untucked-shirt rule for the supported shared-rig templates, not a
  tucked-in mode or general-purpose cloth simulation. UVs and base vertices
  remain unchanged; only the evaluated waist region is adjusted.

## Honest limits — zaroor parhna

This is **photo-to-template**, not automatic exact 3D reconstruction or physical
tailoring. A photo does not reveal hidden circumference, sewing patterns, fabric
elasticity, drape, or exact size. The system does not infer those measurements.
Current templates are based on the original casual shirt and shorts, with a
round-neck adjustment for product-photo shirts; printed collars/buttons do not
become actual 3D collars/buttons. Pants/relaxed
trousers share the original upper shape, with different tapered leg extensions.
Loose, pleated, asymmetric or substantially different cuts need a custom template.
Front/back lighting or pattern differences may produce visible side seams.

New category templates need live checks, especially elbows, crossed arms, knees,
side/back views, and missing-joint cases. Existing fit accuracy cannot be promised
for every untested garment or camera angle. This is not a clothing-size guarantee.

All new processing is local and free. Live pose tracking uses the existing
MediaPipe AI. Photo preparation uses deterministic image processing; it is NOT
being presented as generative AI. No new cloud service, model download or paid
API was installed. The importer itself uploads no garment images.

## Verified so far

See **studio_validation.json** and **studio_roundtrip_validation.json**:
all five template categories move under synthetic poses; packed front/back
images survive asset reload; original mesh/UV restoration is exact; invalid
imports do not replace the active garment; wrist/ankle corrections are independent;
full pants hide when ankles are unavailable; short-sleeve matte remains identical
to the original path.

The corrected black shirt was additionally checked using the supplied front/back
photos: centred logos, crew neckline and a slimmer torso in CPU-rendered previews.
Seven synthetic movement cases passed after activation, with tailored base
geometry and UVs preserved, independent legs and finite landmark-fit output.
See **corrected_product_validation.json**. The original AR folder hashes and
calibration module match the protected baseline. These offline checks do not
establish live wearer fitting accuracy; that test remains pending.

## Waist layering update (14 September)

The black crew shirt + newly generated pants were rendered before/after: the
waistband no longer cuts through the front hem in the normal standing test.
Six top/bottom category combinations passed 42 synthetic deformation checks
(finite output, unchanged UV/base geometry, and no changes outside the waist).
A knee-bend front render was also reviewed; this is not proof of collision-free
motion at every angle. The closed/thickened hem makes nearest-surface automated
penetration counts ambiguous near exposed hems, so these are diagnostics, not
an all-angle accuracy guarantee. Real-wearer verification is still required.
Local depsgraph evaluation median was about 1.6 ms without versus 4.5 ms with
the modifier; this is not camera-to-display latency. See
**waist_swap_validation.json** and **waist_layer_results.json**.

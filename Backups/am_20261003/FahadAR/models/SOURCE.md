# Local foreground model

Google MediaPipe SelfieMulticlass 256x256, retrieved 2026-09-13 from the official model link.

- Guide: https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter
- Model: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
- Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf
- SHA256: `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`
- Labels: background, hair, body-skin, face-skin, clothes, accessories.

Inference runs locally with the existing MediaPipe runtime. No API subscription or per-frame cloud service is used. Model licensing/redistribution terms should be reviewed before packaging a commercial product; this prototype does not constitute a licensing review.

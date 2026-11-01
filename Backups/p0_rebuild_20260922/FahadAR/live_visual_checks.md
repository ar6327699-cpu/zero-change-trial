# Live visual check — 2026-09-12

Observed in the interactive Blender AR copy, not just background renders:

- Real camera video updates inside Blender.
- Existing grey shirt is visible over the real person's front-facing upper body; avatar head/body remain hidden.
- Standing farther away changes garment size and location with the person.
- Front-facing screenshot showed approximately 5 px joint projection error and 177 ms capture-to-apply status. These numbers are not garment-edge accuracy or measured display latency.
- At a side-facing pose, the confidence/visibility gate hid the garment and requested a front-facing pose. Side-view try-on is not solved in this prototype.
- Stop cleared the camera picture and released the camera. First session applied 1,377 frames, with 3.90 px median joint projection error and 160 ms median capture-to-apply delay. These do not establish physical fit or full display latency.
- Restart was initiated for the user. The launcher script compiled, but its separate-window launch was not tested while the live camera was in use. No claim of exhaustive movement testing.

Original blend SHA256 remained `4da05a7c3d33464ae7d1a7c20cf0fcaa353ecc9d8b3513f8023f05bd97fee3e3`.
Original solver SHA256 remained `ebb6dc4a3b1f23e804d2f0d3c1c9d714677963d302b2162591a2042d7bc80d81`.

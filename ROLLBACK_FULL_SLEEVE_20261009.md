# Latest sleeve/coverage update rolled back

The supplied `2026-10-09 23-31-06.mp4` shows visibly incorrect sleeve alignment and the changed camera framing after the latest update. The pre-update version has been restored, retaining changes made before that update.

- Restored seven Python files from `Backups/full_sleeve_fit_20261009`.
- Archived the two added modules, `female_frame.py` and `female_coverage.py`, plus affected compiled caches. The full-sensor framing, new wrist correction, outline expansion, and live side lining are removed from active source.
- Verified all 60 original source/scene hashes against the pre-update manifest. The saved Blender scene is unchanged.
- The previous 32 tracking regression checks pass after restoration.
- Preserved the rejected version and its diagnostics in `Backups/rollback_full_sleeve_20261009` for recovery and investigation.

Only `C:\Users\eisha\Downloads\Zero Change Trial` was modified. The protected Desktop folder was not accessed.

The already-open Blender process still contains the rejected version in memory. Close it **without saving**, then reopen this Trial folder's `START_GARMENT_STUDIO.cmd` to load the restored source and unchanged scene. Stopping and restarting only the camera does not unload the Blender modules or remove the in-memory modifier.

This restores the prior version; it does not claim to solve its remaining tracking issues. Evidence: `work/rollback_full_sleeve_20261009/rollback_validation.json`.

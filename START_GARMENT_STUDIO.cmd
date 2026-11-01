@echo off
rem Separate garment lab. No camera is started until Start AR Camera is clicked.
set "BLENDER="
for %%V in (5.2 5.1) do (
    if not defined BLENDER if exist "C:\Program Files\Blender Foundation\Blender %%V\blender.exe" set "BLENDER=C:\Program Files\Blender Foundation\Blender %%V\blender.exe"
)
if not defined BLENDER (
    echo Blender 5.2 or 5.1 not found under C:\Program Files\Blender Foundation
    pause
    exit /b 1
)
start "AR Garment Studio" "%BLENDER%" "%~dp0fahadGenius_AR_experiment.blend" --python "%~dp0FahadAR\studio_bootstrap.py"

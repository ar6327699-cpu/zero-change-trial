@echo off
rem Separate garment lab. No camera is started until Start AR Camera is clicked.
start "AR Garment Studio" "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" "%~dp0fahadGenius_AR_experiment.blend" --python "%~dp0FahadAR\studio_bootstrap.py"

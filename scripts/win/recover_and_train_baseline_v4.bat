@echo off
rem Stops the stale v3 run, runs the test suite, and if it passes opens the
rem baseline_simplecnn_v4 training window + dashboard. Report: results\recovery\
title FusionEdge launcher
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Config configs\baseline.yaml -Run baseline_simplecnn_v4 -StopRunning
echo.
echo launcher exit code: %ERRORLEVEL%
pause

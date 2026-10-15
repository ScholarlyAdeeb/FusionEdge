@echo off
rem Unattended pipeline: tests -> baseline v5 -> gate -> FusionEdge -> ablations -> tables.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"

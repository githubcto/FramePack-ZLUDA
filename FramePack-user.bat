@echo off

set PATH=%~dp0.zluda;%HIP_PATH%bin;%PATH%
set PYTHON="%~dp0venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv
set COMMANDLINE_ARGS=
@REM set COMMANDLINE_ARGS= --inbrowser

set DISABLE_ADDMM_CUDA_LT=1
set ZLUDA_COMGR_LOG_LEVEL=1

@REM set INCLUDE="%~dp0venv\include";%INCLUDE%
@REM set ZLUDA_NVRTC_LIB="%~dp0venv\Lib\site-packages\torch\lib\nvrtc_cuda.dll"

%PYTHON% demo_gradio.py %COMMANDLINE_ARGS%

pause


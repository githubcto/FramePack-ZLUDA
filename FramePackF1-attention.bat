@echo off

@REM call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

set PATH=%~dp0.zluda;%HIP_PATH%bin;%PATH%
set PYTHON="%~dp0venv/Scripts/python.exe"
set GIT=
set VENV_DIR=./venv
set COMMANDLINE_ARGS=
@REM set COMMANDLINE_ARGS= --inbrowser

set DISABLE_ADDMM_CUDA_LT=1
set ZLUDA_COMGR_LOG_LEVEL=1
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set "TRITON_CACHE_DIR=%~dp0.triton_cache"
set MIOPEN_FIND_MODE=2
@REM set INCLUDE="%~dp0venv\include";%INCLUDE%
@REM set ZLUDA_NVRTC_LIB="%~dp0venv\Lib\site-packages\torch\lib\nvrtc_cuda.dll"

@REM set HIP_VISIBLE_DEVICES=0
@REM set HIP_VISIBLE_DEVICES=1

%PYTHON% demo_gradio_f1_attention.py %COMMANDLINE_ARGS%

pause


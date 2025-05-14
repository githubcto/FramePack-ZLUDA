@echo on

setlocal enabledelayedexpansion

set "VENV_DIR=%~dp0venv"

if exist "%VENV_DIR%\" (
    echo venv exist.
    pause
    exit /b
)

python.exe -m venv venv

call venv\Scripts\activate.bat

python.exe -m pip install --upgrade pip

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip install --upgrade triton-3.3.0+git5dcd7566-cp310-cp310-win_amd64.whl setuptools
@REM pip install --upgrade triton-3.3.0+git5dcd7566-cp311-cp311-win_amd64.whl setuptools
@REM pip install --upgrade triton-3.3.0+git3d100376-cp310-cp310-win_amd64.whl setuptools
@REM pip install --upgrade triton-3.3.0+git3d100376-cp311-cp311-win_amd64.whl setuptools

@REM zluda v3.9.2
@REM curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.dba64c0966df2c71e82255e942c96e2e1cea3a2d/ZLUDA-windows-rocm6-amd64.zip > zluda.zip
@REM mkdir .zluda && tar -xf zluda.zip -C .zluda  --strip-components=1

@REM zluda v3.9.5
@REM curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-windows-rocm6-amd64.zip > zluda.zip
@REM mkdir .zluda && tar -xf zluda.zip -C .zluda  --strip-components=1

@REM zluda v3.9.2 nightly
@REM curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.dba64c0966df2c71e82255e942c96e2e1cea3a2d/ZLUDA-nightly-windows-rocm6-amd64.zip > zluda.zip
@REM mkdir .zluda && tar -xf zluda.zip -C .zluda

@REM zluda v3.9.5 nightly
curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.5e717459179dc272b7d7d23391f0fad66c7459cf/ZLUDA-nightly-windows-rocm6-amd64.zip > zluda.zip
mkdir .zluda && tar -xf zluda.zip -C .zluda

del zluda.zip

copy venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll venv\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy .zluda\cublas.dll venv\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy .zluda\cusparse.dll venv\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy .zluda\nvrtc.dll venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
copy .zluda\cudnn.dll venv\Lib\site-packages\torch\lib\cudnn64_9.dll /y >NUL

echo .
echo * Finished.
echo .

endlocal
pause

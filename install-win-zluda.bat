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

pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

@REM zluda v3.9.2
curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.dba64c0966df2c71e82255e942c96e2e1cea3a2d/ZLUDA-windows-rocm6-amd64.zip > zluda.zip

@REM zluda v3.9.3
@REM curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.0d1513a017397bf9ebbac0b3c846160c8d4fc700/ZLUDA-windows-rocm6-amd64.zip > zluda.zip

mkdir .zluda && tar -xf zluda.zip -C .zluda  --strip-components=1

@REM zluda v3.9.2 nightly
@REM curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.dba64c0966df2c71e82255e942c96e2e1cea3a2d/ZLUDA-nightly-windows-rocm6-amd64.zip > zluda.zip
@REM mkdir .zluda && tar -xf zluda.zip -C .zluda

@REM zluda v3.9.3 nightly
@REM curl -s -L https://github.com/lshqqytiger/ZLUDA/releases/download/rel.0d1513a017397bf9ebbac0b3c846160c8d4fc700/ZLUDA-nightly-windows-rocm6-amd64.zip > zluda.zip
@REM mkdir .zluda && tar -xf zluda.zip -C .zluda

del zluda.zip

copy venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll venv\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy .zluda\cublas.dll venv\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy .zluda\cusparse.dll venv\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy .zluda\nvrtc.dll venv\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL

echo .
echo * Finished.
echo .

endlocal
pause

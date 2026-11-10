@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "YUE_VS=%%i"
if not defined YUE_VS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "YUE_VS=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
if not defined YUE_VS (
  echo Visual Studio C++ Build Tools were not found.
  exit /b 1
)
call "%YUE_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "PATH=%YUE_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%YUE_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "%~dp0.."
cmake -S vendor/yue2.cpp -B vendor/yue2.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF
if errorlevel 1 exit /b 1
cmake --build vendor/yue2.cpp/build --target yue-server yue-synth yue-plan -j 8
exit /b %errorlevel%

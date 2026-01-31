@echo off
setlocal

set SOURCE=%~dp0dist\tree_sandbox_v10.exe
set TARGET=%USERPROFILE%\TreePackingStudioV10

if not exist "%TARGET%" mkdir "%TARGET%"

if exist "%SOURCE%" (
    copy /Y "%SOURCE%" "%TARGET%\TreePackingStudioV10.exe" >nul
) else (
    echo ERROR: %SOURCE% not found. Run PyInstaller first.
    exit /b 1
)

xcopy "%~dp0comparison" "%TARGET%\comparison" /E /I /Y >nul

echo Installed Tree Packing Studio v10 to %TARGET%
echo Please run "%TARGET%\TreePackingStudioV10.exe" to launch the studio.
pause

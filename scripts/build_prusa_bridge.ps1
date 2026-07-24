[CmdletBinding()]
param(
    [switch]$SkipDependencyBuild
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$prusaSource = Join-Path $projectRoot 'deps\PrusaSlicer-version_2.9.6'
$dependencyBuild = Join-Path $projectRoot 'deps\prusa-deps-build'
$dependencyPrefix = Join-Path $projectRoot 'deps\prusa-deps'
$bridgeBuild = Join-Path $projectRoot 'deps\prusa-bridge-build'
$bridgeSource = Join-Path $projectRoot 'native\prusa_bridge'
$nativeOutput = Join-Path $projectRoot 'kuka_slicer\_native'
$vsDevCmd = 'D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat'
$vsCmake = 'D:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'

foreach ($requiredPath in @($python, $prusaSource, $bridgeSource)) {
    if (-not (Test-Path $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

New-Item -ItemType Directory -Path $nativeOutput -Force | Out-Null
$pybind11Directory = (& $python -m pybind11 --cmakedir).Trim()
if (-not $pybind11Directory) {
    throw 'Could not locate pybind11 CMake package from the project virtual environment.'
}
if (-not (Test-Path $vsDevCmd)) {
    throw "Visual Studio 2022 developer environment was not found: $vsDevCmd"
}
if (-not (Test-Path $vsCmake)) {
    throw "Visual Studio 2022 CMake was not found: $vsCmake"
}

# CMake 3.19 does not recognize the VS2022 generator name, but VS2022 ships
# Ninja. Initialize its x64 compiler environment explicitly and use Ninja.
$vsDevCmdShort = (& cmd.exe /d /c "for %I in (`"$vsDevCmd`") do @echo %~sI").Trim()
if (-not $vsDevCmdShort) {
    throw "Could not resolve a command-safe VS developer script path: $vsDevCmd"
}
$vsCmakeShort = (& cmd.exe /d /c "for %I in (`"$vsCmake`") do @echo %~sI").Trim()
if (-not $vsCmakeShort) {
    throw "Could not resolve a command-safe VS CMake path: $vsCmake"
}

function Invoke-VsNinjaCommand([string]$command) {
    & cmd.exe /d /s /c "call $vsDevCmdShort -arch=x64 -host_arch=x64 >nul && $vsCmakeShort $command"
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $command"
    }
}

if (-not $SkipDependencyBuild) {
    # Ninja's native generator does not propagate CMAKE_SYSTEM_PROCESSOR to
    # ExternalProject children.  libjpeg-turbo 3.x requires it on Windows, so
    # forward the x64 target explicitly through Prusa's shared dependency args.
    Invoke-VsNinjaCommand "-S `"$(Join-Path $prusaSource 'deps')`" -B `"$dependencyBuild`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=AMD64 -DDEP_CMAKE_OPTS:STRING=-DCMAKE_SYSTEM_PROCESSOR:STRING=AMD64 `"-DDESTDIR=$dependencyPrefix`" -DDEP_DEBUG=OFF"
    Invoke-VsNinjaCommand "--build `"$dependencyBuild`" --parallel"
}

if (-not (Test-Path (Join-Path $dependencyPrefix 'usr\local'))) {
    throw "Prusa dependency prefix was not found: $dependencyPrefix"
}

Invoke-VsNinjaCommand "-S `"$bridgeSource`" -B `"$bridgeBuild`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=AMD64 `"-DPRUSA_SOURCE_DIR=$prusaSource`" `"-DKUKA_NATIVE_OUTPUT_DIR=$nativeOutput`" `"-DCMAKE_PREFIX_PATH=$(Join-Path $dependencyPrefix 'usr\local')`" `"-Dpybind11_DIR=$pybind11Directory`" `"-DPython_EXECUTABLE=$python`""
Invoke-VsNinjaCommand "--build `"$bridgeBuild`" --target prusa_bridge --parallel"

$extension = Get-ChildItem -Path $nativeOutput -Filter 'prusa_bridge*.pyd' | Select-Object -First 1
if ($null -eq $extension) {
    throw "Build completed but no prusa_bridge .pyd was produced in $nativeOutput"
}
Write-Host "Prusa bridge ready: $($extension.FullName)"

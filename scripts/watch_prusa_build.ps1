[CmdletBinding()]
param(
    [switch]$Watch,
    [ValidateRange(1, 300)]
    [int]$IntervalSeconds = 10
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$depsBuild = Join-Path $projectRoot 'deps\prusa-deps-build'
$depsPrefix = Join-Path $projectRoot 'deps\prusa-deps\usr\local'
$nativeOutput = Join-Path $projectRoot 'kuka_slicer\_native'

function Get-StampState([string]$name) {
    $stampDir = Join-Path $depsBuild "dep_${name}-prefix\src\dep_${name}-stamp"
    return [bool](Get-ChildItem $stampDir -Filter "dep_${name}-done" -ErrorAction SilentlyContinue)
}

function Show-PrusaBuildStatus {
    $buildProcesses = Get-Process cmake, ninja, cl -ErrorAction SilentlyContinue |
        Group-Object ProcessName |
        ForEach-Object { "$($_.Name)=$($_.Count)" }
    $bridge = Get-ChildItem $nativeOutput -Filter 'prusa_bridge*.pyd' -ErrorAction SilentlyContinue |
        Select-Object -First 1

    $states = @(
        "Z3=$([bool](Get-Item (Join-Path $depsPrefix 'lib\libz3.lib') -ErrorAction SilentlyContinue))"
        "Boost=$(Get-StampState 'Boost')"
        "JPEG=$(Get-StampState 'JPEG')"
        "OCCT=$(Get-StampState 'OCCT')"
        "wxWidgets=$(Get-StampState 'wxWidgets')"
        "Bridge=$([bool]$bridge)"
    )

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $($states -join '  ')"
    Write-Host "活跃编译进程: $(if ($buildProcesses) { $buildProcesses -join ', ' } else { '无' })"
    if ($bridge) {
        Write-Host "桥接库: $($bridge.FullName)"
    }
}

do {
    Show-PrusaBuildStatus
    if ($Watch) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Watch)

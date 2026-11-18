#requires -Version 5.1
<#
Build on Windows: .\build.ps1
Preflight only:    .\build.ps1 -ValidateOnly
Prepare the build environment explicitly before running this script:
  .\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-build.txt
The script never deletes a directory and never overwrites an existing release.
#>
[CmdletBinding()]
param([switch]$ValidateOnly)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$AppName = '买家秀采集助手'
$Version = '0.5.11'
$WorkspaceRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)

function Assert-WorkspacePath([string]$Candidate) {
    $full = [IO.Path]::GetFullPath($Candidate)
    $prefix = $WorkspaceRoot + [IO.Path]::DirectorySeparatorChar
    if (($full -ne $WorkspaceRoot) -and -not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a path outside the workspace: $full"
    }
    # Reject junctions/symlinks as well as lexical ../ escapes. All generated
    # directories and every source/destination of publication must pass here.
    $cursor = $full
    while ($cursor -and $cursor.StartsWith($WorkspaceRoot, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing a junction or symbolic link in the build path: $cursor"
            }
        }
        if ($cursor -eq $WorkspaceRoot) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

$Python = Assert-WorkspacePath (Join-Path $WorkspaceRoot '.venv\Scripts\python.exe')
$Spec = Assert-WorkspacePath (Join-Path $WorkspaceRoot 'packaging\buyer_show.spec')
$Entry = Assert-WorkspacePath (Join-Path $WorkspaceRoot 'desktop.py')
$QuickStart = Assert-WorkspacePath (Join-Path $WorkspaceRoot 'packaging\快速开始.txt')
$BuildRoot = Assert-WorkspacePath (Join-Path $WorkspaceRoot 'build')
$DistRoot = Assert-WorkspacePath (Join-Path $WorkspaceRoot 'dist')
$ReleaseDir = Assert-WorkspacePath (Join-Path $DistRoot "$AppName-$Version")
$ZipName = "$AppName-$Version-win-x64.zip"
$ReleaseZip = Assert-WorkspacePath (Join-Path $DistRoot $ZipName)
foreach ($required in @($Python, $Spec, $Entry, $QuickStart)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required build input not found: $required"
    }
}
foreach ($destination in @($ReleaseDir, $ReleaseZip)) {
    if (Test-Path -LiteralPath $destination) {
        throw "Existing release is protected. Move or rename it yourself before rebuilding: $destination"
    }
}

$Preflight = @'
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import sys
import tkinter
import PIL.ImageTk
import PIL._imagingtk
import certifi
import httpx
import playwright
import imageio_ffmpeg
import rapidocr_onnxruntime
import onnxruntime
import cv2
from buyer_show import __version__

root = Path(sys.argv[1]).resolve()
assert sys.platform == 'win32' and platform.machine().lower() in ('amd64', 'x86_64'), 'Windows x64 is required'
assert sys.version_info[:2] == (3, 12), 'Use CPython 3.12 for this pinned build'
assert __version__ == '0.5.11', 'Application version does not match this package'
assert Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file(), 'Bundled FFmpeg missing'
assert len(list((Path(rapidocr_onnxruntime.__file__).parent / 'models').glob('*.onnx'))) == 3, 'Bundled OCR models missing'
versions = {}
for name in ('requirements-tested.txt', 'requirements-build.txt'):
    for raw in (root / name).read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line or line.startswith(('#', '-r ')):
            continue
        package, expected = line.split('==', 1)
        actual = metadata.version(package)
        assert actual == expected, f'{package}: expected {expected}, installed {actual}'
        versions[package] = actual
driver = Path(playwright.__file__).parent / 'driver'
for name in ('node.exe', 'package/cli.js', 'package/package.json', 'package/browsers.json'):
    assert (driver / name).is_file(), f'Missing driver file: {name}'
assert Path(certifi.where()).is_file(), 'CA certificate bundle missing'
print(json.dumps({'preflight': 'passed', 'python': platform.python_version(), 'tcl': tkinter.Tcl().eval('info patchlevel'),
                  'version': __version__, 'packages': versions}, ensure_ascii=True, indent=2))
'@

Push-Location -LiteralPath $WorkspaceRoot
try {
    $Preflight | & $Python - $WorkspaceRoot
    if ($LASTEXITCODE -ne 0) { throw 'Build preflight failed. Install requirements-build.txt in the project .venv.' }
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Installed dependency consistency check failed.' }
    if ($ValidateOnly) {
        Write-Host 'Preflight passed. No executable was built and no release was changed.'
        return
    }

    foreach ($directory in @($BuildRoot, $DistRoot)) {
        if (-not (Test-Path -LiteralPath $directory)) { New-Item -ItemType Directory -Path $directory | Out-Null }
    }
    $runName = 'package-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $RunRoot = Assert-WorkspacePath (Join-Path $BuildRoot $runName)
    New-Item -ItemType Directory -Path $RunRoot | Out-Null
    $StagingDist = Assert-WorkspacePath (Join-Path $RunRoot 'dist')
    $WorkDir = Assert-WorkspacePath (Join-Path $RunRoot 'work')
    Write-Host "Building in isolated staging directory: $RunRoot"
    # Fresh staging avoids --clean and --noconfirm, both of which can remove
    # existing paths. The spec produces a Windows windowed onedir application.
    & $Python -m PyInstaller --distpath $StagingDist --workpath $WorkDir $Spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed. Staging retained for inspection: $RunRoot" }

    $StagedRelease = Assert-WorkspacePath (Join-Path $StagingDist "$AppName-$Version")
    $StagedExe = Assert-WorkspacePath (Join-Path $StagedRelease ($AppName + '.exe'))
    $CheckPath = Assert-WorkspacePath (Join-Path $RunRoot 'bundle-self-check.json')
    if (-not (Test-Path -LiteralPath $StagedExe -PathType Leaf)) { throw 'Packaged executable is missing.' }
    # A unique output path prevents a stale JSON result from passing this check.
    $checkProcess = Start-Process -FilePath $StagedExe -WorkingDirectory $StagedRelease `
        -ArgumentList @('--self-check', ('"' + $CheckPath + '"')) -WindowStyle Hidden -PassThru
    $null = $checkProcess.Handle
    if (-not $checkProcess.WaitForExit(60000)) {
        # Terminate only this build's self-check tree, including its spawn child.
        & taskkill.exe /PID $checkProcess.Id /T /F | Out-Null
        $null = $checkProcess.WaitForExit(5000)
        throw "Bundle self-check exceeded 60 seconds. Staging retained: $RunRoot"
    }
    $checkProcess.Refresh()
    if (-not (Test-Path -LiteralPath $CheckPath -PathType Leaf)) { throw 'Bundle self-check did not write its report.' }
    $check = Get-Content -LiteralPath $CheckPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($checkProcess.ExitCode -ne 0 -or $check.ok -ne $true -or $check.frozen -ne $true -or
        $check.browser_launched -ne $false -or $check.version -ne $Version) {
        throw "Bundle self-check failed. Inspect: $CheckPath"
    }
    Copy-Item -LiteralPath $QuickStart -Destination (Join-Path $StagedRelease '快速开始.txt')
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\FFmpeg-component.txt') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\FFmpeg-COPYING.GPLv3.txt') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\watermark-components.json') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\RapidOCR-LICENSE.txt') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\PaddleOCR-LICENSE.txt') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\lama-model.json') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot 'packaging\LICENSE-LaMa.txt') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot '测试商品链接.txt') -Destination $StagedRelease
    Copy-Item -LiteralPath (Join-Path $WorkspaceRoot '去水印使用说明.txt') -Destination $StagedRelease

    $StagedZip = Assert-WorkspacePath (Join-Path $RunRoot $ZipName)
    $Package = @'
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED

release, archive, check_path, root = map(Path, sys.argv[1:])
internal = release / '_internal'
required = [
    internal / 'playwright/driver/node.exe', internal / 'playwright/driver/package/cli.js',
    internal / 'playwright/driver/package/package.json', internal / 'playwright/driver/package/browsers.json',
    internal / '_tcl_data/init.tcl', internal / '_tk_data/tk.tcl', internal / 'certifi/cacert.pem',
    internal / 'base_library.zip',
]
for file in required:
    assert file.is_file(), f'Missing packaged component: {file.relative_to(release)}'
for pattern in ('_tkinter*.pyd', 'tcl*.dll', 'tk*.dll', 'python3*.dll', 'PIL/_imaging*.pyd'):
    assert list(internal.glob(pattern)), f'Missing packaged runtime: {pattern}'
assert not list(release.rglob('.local-browsers')), 'Browser binaries must not be bundled'
assert list((internal / 'imageio_ffmpeg/binaries').glob('*.exe')), 'Video validation binary missing'
assert len(list((internal / 'rapidocr_onnxruntime/models').glob('*.onnx'))) == 3, 'OCR models missing'
assert (internal / 'licenses/lama-model.json').is_file(), 'Optional AI model metadata missing'
assert (internal / 'licenses/LICENSE-LaMa.txt').is_file(), 'Optional AI model licence missing'
assert not list(release.rglob('lama_fp32.onnx')), 'Optional AI model must not be bundled'
versions = {}
for name in ('requirements-tested.txt', 'requirements-build.txt'):
    for raw in (root / name).read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if line and not line.startswith(('#', '-r ')):
            package = line.split('==', 1)[0]
            versions[package] = metadata.version(package)
info = {'version': '0.5.11', 'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions,
        'browser_bundled': False, 'default_browser': 'installed Microsoft Edge',
        'optional_browser': 'installed Google Chrome',
        'app_data': r'%LOCALAPPDATA%\BuyerShowValidator',
        'watermark': {'detection': 'offline CPU tiled OCR; product-scoped known-merchant low-contrast text and repeated anchored graphics',
                      'batch': 'one worker; default product main/detail/video; byte-exact no-hit copies; uncertain review queue; classified exports',
                      'image_repair': 'auto flat/directional background fill / OpenCV Telea / NS / optional LaMa ONNX CPU; original preserved',
                      'optional_model': json.loads((root / 'packaging/lama-model.json').read_text(encoding='utf-8')),
                      'optional_model_bundled': False,
                      'video_repair': 'Adaptive CPU LaMa keyframes with boundary-gated optical flow, simple-background interpolation, or FFmpeg delogo; exact timeline/audio verification',
                      'limitations': 'Not universal logo detection or guaranteed faithful hidden-detail recovery; video distortion and flicker can remain'},
        'acceptance': {'taobao': 'pending_live_collection_acceptance',
                       'douyin': 'pending_live_collection_acceptance', 'xhs': 'product_media_reference_parser_reviews_unsupported'},
        'bundle_self_check': json.loads(check_path.read_text(encoding='utf-8'))}
(release / 'build-info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
# Python zipfile writes Unicode paths using UTF-8 ZIP metadata. Keep the parent
# app directory inside the archive so Extract All preserves the onedir layout.
with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
    for file in sorted(release.rglob('*')):
        if file.is_symlink():
            raise RuntimeError('Unexpected symlink in release')
        if file.is_file():
            z.write(file, file.relative_to(release.parent).as_posix())
with ZipFile(archive) as z:
    assert z.testzip() is None, 'ZIP CRC validation failed'
print('Bundle components and ZIP CRC verified.')
'@
    $Package | & $Python - $StagedRelease $StagedZip $CheckPath $WorkspaceRoot
    if ($LASTEXITCODE -ne 0) { throw "Bundle inventory or ZIP verification failed. Staging retained: $RunRoot" }

    # Recheck immediately before the only moves. No recursive deletion or
    # overwrite is performed; existing releases are always the user's to manage.
    foreach ($path in @($StagedRelease, $StagedZip, $ReleaseDir, $ReleaseZip)) {
        $null = Assert-WorkspacePath $path
    }
    foreach ($destination in @($ReleaseDir, $ReleaseZip)) {
        if (Test-Path -LiteralPath $destination) { throw "Release appeared during build; refusing overwrite: $destination" }
    }
    [IO.Directory]::Move($StagedRelease, $ReleaseDir)
    [IO.File]::Move($StagedZip, $ReleaseZip)
    $hash = Get-FileHash -LiteralPath $ReleaseZip -Algorithm SHA256
    Write-Host "Application: $(Join-Path $ReleaseDir ($AppName + '.exe'))"
    Write-Host "ZIP: $ReleaseZip"
    Write-Host "SHA256: $($hash.Hash)"
    Write-Host 'CPU image/video watermark test release; product collection limitations are unchanged.'
}
finally {
    Pop-Location
}

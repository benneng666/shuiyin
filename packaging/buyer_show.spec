# -*- mode: python ; coding: utf-8 -*-
"""Windows onedir package; build via ../build.ps1.

Official references checked 2026-10-29:
https://playwright.dev/python/docs/library#pyinstaller
https://pyinstaller.org/en/stable/spec-files.html
https://pyinstaller.org/en/stable/hooks.html

Playwright's own API hooks collect its driver package. Explicit data and node
entries here also make the required on-disk layout auditable. PyInstaller's
_tkinter hook collects Tcl/Tk libraries/data and enables its runtime hook.
This intentionally excludes installed browser binaries and all user profiles.
"""

from importlib.util import find_spec
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable,
    VarFileInfo, VarStruct, VSVersionInfo,
)

ROOT = Path(SPECPATH).resolve().parent
APP_NAME = "买家秀采集助手"
VERSION = "0.5.11"
if sys.platform != "win32":
    raise SystemExit("Build this specification on Windows with the project .venv.")

playwright_dir = Path(next(iter(find_spec("playwright").submodule_search_locations)))
driver_dir = playwright_dir / "driver"
for relative in ("node.exe", "package/cli.js", "package/package.json", "package/browsers.json"):
    if not (driver_dir / relative).is_file():
        raise SystemExit(f"Missing Playwright driver component: {relative}")

datas = collect_data_files("playwright", excludes=["driver/package/.local-browsers/**"])
datas += collect_data_files("certifi")
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("rapidocr_onnxruntime")
datas += [(str(ROOT / "packaging" / name), "licenses")
          for name in ("lama-model.json", "LICENSE-LaMa.txt")]
# Keep package metadata and bundled licence files with the distribution.
for distribution in (
    "playwright", "pillow", "httpx", "httpcore", "certifi", "anyio", "greenlet",
    "h11", "idna", "pyee", "typing_extensions", "imageio-ffmpeg",
    "rapidocr-onnxruntime", "onnxruntime", "opencv-python", "numpy", "PyYAML",
    "Shapely", "pyclipper", "six", "coloredlogs", "flatbuffers", "protobuf",
    "sympy", "mpmath", "humanfriendly", "pyreadline3", "tqdm", "colorama",
):
    datas += copy_metadata(distribution)

hiddenimports = collect_submodules("buyer_show") + [
    "playwright.async_api", "playwright._impl._driver",
    "tkinter", "_tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
    "PIL.Image", "PIL.ImageTk", "PIL._imagingtk",
    "PIL.PngImagePlugin", "PIL.JpegImagePlugin", "PIL.WebPImagePlugin",
    "httpx", "httpcore", "certifi", "anyio._backends._asyncio",
    "imageio_ffmpeg",
    "rapidocr_onnxruntime", "onnxruntime", "cv2", "numpy", "PIL.ImageOps",
]

a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=[(str(driver_dir / "node.exe"), "playwright/driver")],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "unittest", "IPython", "notebook", "matplotlib"],
    noarchive=False,
)
# A developer might have run PLAYWRIGHT_BROWSERS_PATH=0 in this environment.
# The official hook also adds data; filter the final TOCs, not only our inputs.
def is_browser_payload(entry):
    return ".local-browsers" in Path(entry[0]).parts or ".local-browsers" in Path(entry[1]).parts

a.datas = [entry for entry in a.datas if not is_browser_payload(entry)]
a.binaries = [entry for entry in a.binaries if not is_browser_payload(entry)]

version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=tuple(map(int, VERSION.split("."))) + (0,), prodvers=tuple(map(int, VERSION.split("."))) + (0,), mask=0x3F,
                     flags=0x2, OS=0x40004, fileType=0x1, subtype=0, date=(0, 0)),
    kids=[
        StringFileInfo([StringTable("080404B0", [
            StringStruct("FileDescription", APP_NAME + "（技术验证版）"),
            StringStruct("FileVersion", VERSION),
            StringStruct("ProductName", APP_NAME),
            StringStruct("ProductVersion", VERSION),
            StringStruct("OriginalFilename", APP_NAME + ".exe"),
        ])]),
        VarFileInfo([VarStruct("Translation", [2052, 1200])]),
    ],
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon="NONE",
    version=version_info,
    uac_admin=False,
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME + "-" + VERSION)

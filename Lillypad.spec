# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller Lillypad.spec
# Output:      dist/Lillypad/Lillypad.exe

import glob
import importlib.util
import os

# zaber-motion is a thin Python wrapper around a native core library shipped in
# a sibling `zaber_motion_bindings` package that contains *no* Python module —
# only the .dll. PyInstaller's import analysis therefore pulls in every
# zaber_motion .py file and none of the actual library, and the frozen app dies
# with "Could not find library zaber-motion-core-windows-amd64.dll" the moment
# hardware.ZaberStage imports it.
#
# zaber_motion/bindings.py resolves the library as
#     dirname(zaber_motion/bindings.py)/../zaber_motion_bindings/<lib name>
# so the file has to land in <bundle>/zaber_motion_bindings/, next to the
# zaber_motion package — hence the explicit dest dir below.
_zaber_spec = importlib.util.find_spec('zaber_motion')
if _zaber_spec is None:
    raise SystemExit('zaber_motion is not installed in the build environment; '
                     'run: pip install --no-deps -r requirements.txt')
_zaber_bindings_dir = os.path.join(
    os.path.dirname(os.path.dirname(_zaber_spec.origin)), 'zaber_motion_bindings')
_zaber_libs = [(p, 'zaber_motion_bindings')
               for p in glob.glob(os.path.join(_zaber_bindings_dir, 'zaber-motion-core-*'))]
if not _zaber_libs:
    raise SystemExit(f'No zaber-motion core library found in {_zaber_bindings_dir}; '
                     'reinstall zaber_motion.')

# libusb-1.0.dll ships as data inside the libusb_package wheel, which
# hardware._ensure_libusb_dll() resolves via libusb_package.get_library_path()
# — seabreeze's pyseabreeze backend (the default; the only one supporting the
# newer Ocean Insight models) is dead without it. PyInstaller does not collect
# in-package binaries by itself, so put the DLL back inside the package dir.
_libusb_spec = importlib.util.find_spec('libusb_package')
if _libusb_spec is None:
    raise SystemExit('libusb_package is not installed in the build environment; '
                     'run: pip install --no-deps -r requirements.txt')
_libusb_dlls = [(p, 'libusb_package') for p in
                glob.glob(os.path.join(os.path.dirname(_libusb_spec.origin), '*.dll'))]
if not _libusb_dlls:
    raise SystemExit('No libusb DLL found inside libusb_package; '
                     'reinstall libusb-package.')

# The Avantes AvaSpecX64.dll is deliberately NOT bundled, unlike the two
# libraries above. Its licence does permit redistribution with an application
# that controls Avantes equipment, so this is a practical choice, not a legal
# one:
#   * zaber_motion and libusb_package live inside pip packages, so
#     importlib.util.find_spec gives this spec a deterministic source path.
#     The Avantes DLL has none — its installer drops it in a VERSIONED folder
#     at the root of the system drive (C:\AvaSpecX64-DLL_9.14.0.0\), so there
#     is nothing stable to point at from here.
#   * The AvaSpec USB driver has to be installed on the target machine anyway,
#     and it comes with the same package, so bundling removes no install step.
#   * Pinning one DLL version into the bundle would override whatever the
#     operator installed for their own spectrometer.
# avantes.py resolves the DLL at runtime (LILLYPAD_AVASPEC_DLL, then next to
# the .exe, then the versioned install folder, then PATH), so dropping a
# specific DLL beside Lillypad.exe remains available if a self-contained
# folder is wanted.
#
# No hiddenimports entry is needed for avantes.py: PyInstaller's analysis scans
# function bodies, so the lazy `import avantes` inside AvantesSpectrometer is
# found the same way the lazy zaber_motion import is.
a = Analysis(
    ['frog_gui_fast.py'],
    pathex=[],
    binaries=_zaber_libs + _libusb_dlls,
    # calibration_files/*.txt are SEED data: frog_gui_fast.seed_calibration_dir
    # copies them from the bundle to a user-editable calibration_files/ folder
    # next to the .exe on first run (only top-level files — Old/ stays behind).
    datas=[('icons', 'icons')] + [
        (p, 'calibration_files') for p in glob.glob('calibration_files/*.txt')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Lillypad is PySide6-only. PyQt5 reaches the environment as a declared
    # pylablib dependency, and a second Qt binding in the bundle has broken
    # builds before — keep it out even if something reinstalls it.
    excludes=['PyQt5', 'PyQt6', 'PySide2'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Lillypad',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/Lilypad.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    # The ~19 MB zaber-motion core is a packed native library; UPX has no real
    # gain on it and compressing it risks a load failure at runtime.
    upx_exclude=[os.path.basename(src) for src, _ in _zaber_libs],
    name='Lillypad',
)

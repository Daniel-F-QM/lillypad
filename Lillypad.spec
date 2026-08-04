# -*- mode: python ; coding: utf-8 -*-
# Build with:  pyinstaller Lillypad.spec
# Output:      dist/Lillypad/Lillypad.exe


a = Analysis(
    ['frog_gui_fast.py'],
    pathex=[],
    binaries=[],
    datas=[('icons', 'icons')],
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
    upx_exclude=[],
    name='Lillypad',
)

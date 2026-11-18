# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import os
from pathlib import Path

block_cipher = None
PROJECT_ROOT = Path(os.getcwd())

datas = [
    (str(PROJECT_ROOT / 'assets' / 'models' / 'face_landmarker.task'), 'assets/models'),
    (str(PROJECT_ROOT / 'assets' / 'models' / 'geometry_pipeline_metadata_landmarks.binarypb'), 'assets/models'),
]
binaries = []
hiddenimports = [
    'multiprocessing.shared_memory',
    'multiprocessing',
    'numpy',
    'cv2',
    'mediapipe',
]

# Collect mediapipe package data, dylibs and submodules
mp_datas, mp_binaries, mp_hidden = collect_all('mediapipe')
datas += mp_datas
binaries += mp_binaries
hiddenimports += mp_hidden

# Collect opencv package
cv_datas, cv_binaries, cv_hidden = collect_all('cv2')
datas += cv_datas
binaries += cv_binaries
hiddenimports += cv_hidden

a = Analysis(
    [str(PROJECT_ROOT / 'src' / 'vision_engine' / 'main.py')],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / 'src')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='eyemouse_engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='eyemouse_engine',
)

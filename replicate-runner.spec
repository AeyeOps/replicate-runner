# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

block_cipher = None

package_data = collect_data_files(
    'replicate_runner',
    includes=['config/*.yaml'],
)
metadata_packages = ['replicate', 'huggingface_hub', 'typer', 'rich', 'python-dotenv', 'pyyaml']
metadata_data = []
for pkg in metadata_packages:
    metadata_data += copy_metadata(pkg)

a = Analysis(
    ['replicate_runner/main.py'],
    pathex=[],
    binaries=[],
    datas=package_data + metadata_data,
hiddenimports=[
        'replicate_runner',
        'replicate_runner.main',
        'replicate_runner.config_loader',
        'replicate_runner.logger_config',
        'replicate_runner.commands',
        'replicate_runner.commands.replicate_cmds',
        'replicate_runner.commands.hf_cmds',
        'replicate_runner.commands.profile_cmds',
        'replicate_runner.commands.prompt_cmds',
        'replicate_runner.commands.explore_cmds',
        'replicate_runner.persona',
        'replicate_runner.profile_runtime',
        'replicate_runner.profiles',
        'replicate_runner.prompt_engine',
        'typer',
        'rich',
        'rich.console',
        'rich.progress',
        'rich.table',
        'dotenv',
        'python-dotenv',
        'yaml',
        'pyyaml',
        'replicate',
        'replicate.client',
        'replicate.model',
        'replicate.version',
        'replicate.prediction',
        'huggingface_hub',
        'huggingface_hub.hf_api',
        'httpx',
        'pydantic',
        'importlib.metadata',
    ],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='replicate-runner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

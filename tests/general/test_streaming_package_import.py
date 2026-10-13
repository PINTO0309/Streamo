"""Exercise package discovery and worker imports without loading a model."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def run_python(args, cwd, **kwargs):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONSAFEPATH', None)
    env['CUDA_VISIBLE_DEVICES'] = ''
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90, **kwargs)


@pytest.mark.parametrize('build_hook', ['build_wheel', 'build_editable'])
def test_installed_package_imports_outside_repository(tmp_path, build_hook):
    source = tmp_path / 'source'
    source.mkdir()
    # Use the actual build configuration, with only enough Swift files for metadata.
    for name in ('pyproject.toml', 'setup.py', 'README.md', 'requirements.txt'):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / 'requirements', source / 'requirements')
    shutil.copytree(ROOT / 'streamo_utils', source / 'streamo_utils', ignore=shutil.ignore_patterns('__pycache__'))
    (source / 'swift').mkdir()
    for name in ('__init__.py', 'version.py'):
        shutil.copy2(ROOT / 'swift' / name, source / 'swift' / name)

    build = run_python([
        '-c', f'from setuptools.build_meta import {build_hook}; {build_hook}("dist")',
    ], cwd=source)
    assert build.returncode == 0, build.stdout
    wheel, = (source / 'dist').glob('*.whl')
    site_packages = tmp_path / 'site-packages'
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site_packages)
    outside = tmp_path / 'outside'
    outside.mkdir()
    expected_root = source if build_hook == 'build_editable' else site_packages
    # -I removes the checkout, PYTHONPATH, and user-site packages as import fallbacks.
    imported = run_python([
        '-I', '-c',
        'import site, sys; from pathlib import Path; '
        'site.addsitedir(sys.argv[1]); '
        'from streamo_utils import streaming; '
        'assert Path(streaming.__file__).is_relative_to(sys.argv[2]); '
        'assert streaming.sampling_indices(8, 8, 4) == [1, 3, 5, 7]',
        str(site_packages), str(expected_root),
    ], cwd=outside)
    assert imported.returncode == 0, imported.stdout


def test_train_launcher_imports_shared_utilities_in_workers(tmp_path):
    source = tmp_path / 'source'
    cli = source / 'swift' / 'cli'
    cli.mkdir(parents=True)
    (source / 'swift' / '__init__.py').touch()
    (cli / '__init__.py').touch()
    shutil.copytree(ROOT / 'streamo_utils', source / 'streamo_utils', ignore=shutil.ignore_patterns('__pycache__'))
    (cli / 'sft.py').write_text(
        'import os\n'
        'from pathlib import Path\n'
        'from streamo_utils import streaming\n'
        'assert Path(streaming.__file__).is_relative_to(Path.cwd())\n'
        'assert streaming.sampling_indices(8, 8, 4) == [1, 3, 5, 7]\n'
        'Path("worker_" + os.environ["RANK"]).write_text("ok")\n', encoding='utf-8')
    # Capture the real shell command, then exercise its entrypoint with two CPU workers.
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_python = fake_bin / 'python'
    fake_python.write_text(
        f'#!{sys.executable}\n'
        'import json, sys\n'
        'from pathlib import Path\n'
        'Path("launch.json").write_text(json.dumps(sys.argv[1:]))\n', encoding='utf-8')
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = f'{fake_bin}{os.pathsep}{env["PATH"]}'
    captured = subprocess.run(
        ['bash', str(ROOT / 'train.sh')], cwd=source, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    assert captured.returncode == 0, captured.stdout
    args = json.loads((source / 'launch.json').read_text())
    args[args.index('--nproc_per_node') + 1] = '2'
    args.insert(args.index('--nproc_per_node'), '--standalone')
    launched = run_python(args, cwd=source)
    assert launched.returncode == 0, launched.stdout
    assert (source / 'worker_0').read_text() == 'ok'
    assert (source / 'worker_1').read_text() == 'ok'


@pytest.mark.parametrize('launch_mode', ['script', 'torchrun_script', 'torchrun_module'])
def test_real_sft_entrypoint_imports_checkout_without_installation(tmp_path, launch_mode):
    source = tmp_path / 'source'
    cli = source / 'swift' / 'cli'
    cli.mkdir(parents=True)
    (source / 'swift' / '__init__.py').touch()
    (cli / '__init__.py').touch()
    shutil.copy2(ROOT / 'swift' / 'cli' / 'sft.py', cli / 'sft.py')
    shutil.copytree(ROOT / 'streamo_utils', source / 'streamo_utils', ignore=shutil.ignore_patterns('__pycache__'))
    llm = source / 'swift' / 'llm'
    llm.mkdir()
    # Stop at sft_main to test imports without model downloads, CUDA, or training dependencies.
    (llm / '__init__.py').write_text(
        'import os\n'
        'from pathlib import Path\n'
        'from streamo_utils import streaming\n'
        'root = Path(__file__).resolve().parents[2]\n'
        'assert Path(streaming.__file__).is_relative_to(root)\n'
        'assert streaming.sampling_indices(8, 8, 4) == [1, 3, 5, 7]\n'
        'def sft_main():\n'
        '    (root / ("worker_" + os.environ.get("RANK", "0"))).write_text("ok")\n',
        encoding='utf-8')
    # An unrelated cwd also checks that the script finds its root from __file__.
    outside = tmp_path / 'outside'
    outside.mkdir()
    cwd = outside
    entrypoint = [str(cli / 'sft.py')]
    worker_count = 1
    if launch_mode.startswith('torchrun'):
        worker_count = 4
        if launch_mode == 'torchrun_module':
            cwd = source
            entrypoint = ['--module', 'swift.cli.sft']
        entrypoint = [
            '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(worker_count), *entrypoint,
        ]
    launched = run_python(entrypoint, cwd=cwd)
    assert launched.returncode == 0, launched.stdout
    for rank in range(worker_count):
        assert (source / f'worker_{rank}').read_text() == 'ok'

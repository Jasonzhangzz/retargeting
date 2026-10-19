"""Portability contracts: paths, CLI and bundled physical model/input consistency."""
from pathlib import Path
import ast
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_no_legacy_project_imports_or_absolute_asset_paths():
    for folder in ('retarget','forceaware','comfree_warp'):
        for path in (ROOT/folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.Import):
                    names=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom):
                    names=[node.module or '']
                else: continue
                assert not any(name.split('.')[0] in ('examples','planning','dexforge','contactaware','tools') for name in names), path
    xml_paths=[* (ROOT/'assets').rglob('*.xml'),* (ROOT/'data/example/sequence/scene').rglob('*.xml')]
    for path in xml_paths:
        for element in ET.parse(path).getroot().iter():
            if 'file' in element.attrib:
                target=Path(element.attrib['file'])
                assert not target.is_absolute(), path
                assert (path.parent/target).is_file(), (path,target)
                assert (path.parent/target).resolve().is_relative_to(ROOT)


def test_cli_from_another_working_directory(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--help'],cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0
    assert all(flag in result.stdout for flag in ('--fa','--adamw','--sqp','--noise','--no-admm','--no-contact-exploration'))
    conflict=subprocess.run([sys.executable,str(ROOT/'run.py'),'--fa','--adamw'],cwd=tmp_path,text=True,capture_output=True)
    assert conflict.returncode==2
    invalid=subprocess.run([sys.executable,str(ROOT/'run.py'),'--sqp','--iterations','2'],cwd=tmp_path,text=True,capture_output=True)
    assert invalid.returncode==2
    for flags in (['--noise','--noise-amplitude','nan'],['--noise','--noise-amplitude','-0.1'],
                  ['--noise','--noise-seed','-1'],['--fa','--no-admm']):
        invalid=subprocess.run([sys.executable,str(ROOT/'run.py'),*flags],cwd=tmp_path,text=True,capture_output=True)
        assert invalid.returncode==2


def test_bundled_input_and_render_model_agree():
    import mujoco
    from forceaware.config import load_retarget_config
    example=ROOT/'data/example'
    cfg=load_retarget_config(example/'sequence',example/'forceaware.yaml',contactaware_dir=example/'contactaware')
    model=mujoco.MjModel.from_xml_path(str(example/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    assert (model.nq,model.nv,model.nu)==(33,32,26)
    assert cfg.sequence.xml.is_file()
    hand=np.load(example/'contactaware/wuji_right_qpos.npy')
    obj=np.load(example/'contactaware/object_pose_7.npy')
    with np.load(example/'contactaware/contact_guidance.npz') as guidance:
        assert len(hand)==len(obj)==len(guidance['contact_mask'])
    assert np.isfinite(hand).all() and np.isfinite(obj).all()

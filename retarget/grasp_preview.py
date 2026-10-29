"""Live grasp viewer in its own GLFW process, plus headless physics video."""
import json
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np


def camera_for(model,data):
    obj = model.body('grasp_object').id
    palm = data.xpos[model.body('right_palm_link').id]
    direction = data.xpos[obj]-palm
    camera = mujoco.MjvCamera()
    camera.lookat[:] = .85*data.xpos[obj]+.15*palm
    camera.distance = .32
    camera.azimuth = np.degrees(np.arctan2(direction[1],direction[0]))+200
    camera.elevation = -np.degrees(np.arctan2(direction[2],np.linalg.norm(direction[:2])))
    return camera


class GraspPreview:
    def __init__(self,case,enabled):
        self.case = Path(case)
        self.process = None
        self.started = None
        (self.case/'viewer_finished.json').unlink(missing_ok=True)
        (self.case/'grasp_live.npz').unlink(missing_ok=True)
        if enabled:
            environment = dict(os.environ,MUJOCO_GL='glfw')
            self.process = subprocess.Popen([sys.executable,str(Path(__file__).resolve().parents[1]/'run_grasp.py'),
                '--replay',str(self.case),'--follow','--no-render','--no-drop-test'],env=environment)

    def update(self,qpos,qvel,seconds,status):
        if self.process is None or self.process.poll() is not None:
            return
        temporary = self.case/'grasp_live_tmp.npz'
        np.savez(temporary,qpos=qpos,qvel=qvel,time=seconds,status=np.asarray(status))
        temporary.replace(self.case/'grasp_live.npz')
        if self.started is None:
            self.started = time.perf_counter()-seconds
        remaining = self.started+seconds-time.perf_counter()
        if remaining>0:
            time.sleep(min(remaining,.05))

    def close(self):
        (self.case/'viewer_finished.json').write_text(json.dumps(dict(finished=True)))


def follow(case):
    """Display only: CUDA execution stays in the parent process."""
    from mujoco import viewer
    case = Path(case)
    model = mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
    data = mujoco.MjData(model)
    with np.load(case/'grasp.npz') as saved:
        data.qpos[:] = saved['qpos']
    mujoco.mj_forward(model,data)
    model.vis.headlight.ambient[:] = .4
    camera = camera_for(model,data)
    print('[viewer] Live settling / gravity test. Close the window to continue without display.',flush=True)
    previous = None
    last_notice = 0.
    with viewer.launch_passive(model,data) as window:
        window.cam.lookat[:] = camera.lookat
        window.cam.distance,window.cam.azimuth,window.cam.elevation = camera.distance,camera.azimuth,camera.elevation
        while window.is_running():
            path = case/'grasp_live.npz'
            if path.is_file():
                with np.load(path) as saved:
                    with window.lock():
                        data.qpos[:] = saved['qpos'];data.qvel[:] = saved['qvel'];data.time = float(saved['time'])
                        mujoco.mj_forward(model,data)
                    status = str(saved['status'])
                    now=time.perf_counter()
                    if status!=previous or now-last_notice>=2.:
                        print(f'[viewer] {status} | simulated t={float(saved["time"]):.2f}s',flush=True)
                        previous = status
                        last_notice=now
            window.sync()
            time.sleep(.016)


def render_drop_video(case,arrays,report):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    case = Path(case)
    model = mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
    data = mujoco.MjData(model)
    data.qpos[:] = arrays['qpos'][0]
    mujoco.mj_forward(model,data)
    model.vis.headlight.ambient[:] = .4
    camera = camera_for(model,data)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf',18)
    except OSError:
        font = ImageFont.load_default()
    fps = 30
    times = np.arange(0.,arrays['time'][-1]+1/fps,1/fps)
    indices = np.minimum(np.searchsorted(arrays['time'],times),len(arrays['time'])-1)
    with mujoco.Renderer(model,height=480,width=640) as renderer, imageio.get_writer(
            case/'drop_test.mp4',fps=fps,codec='libx264') as writer:
        for i in indices:
            data.qpos[:] = arrays['qpos'][i];data.qvel[:] = arrays['qvel'][i]
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera)
            frame = Image.fromarray(renderer.render())
            phase = arrays['phase'][i]
            if phase<0:
                text = ('Building physical preload' if report.get('load_conditioned')
                        else 'Forming grasp: stationary-reference MPC')
            elif phase==0:
                text = 'Settling: object held' if report['settling_object_fixture'] else 'Settling: free'
            else:
                text = 'Released: gravity 9.81 m/s^2'
            ImageDraw.Draw(frame).text((12,12),f'{text} | t={arrays["time"][i]:.2f}s',font=font,fill='white')
            writer.append_data(np.asarray(frame))
        ImageDraw.Draw(frame).text((12,40),f'Result: {report["status"]}',font=font,fill='white')
        frame.save(case/'drop_test_final.png')


def invalid_preload_report(result,args):
    """Reject a certified bad power input without disabling legacy formation.

    An invalid initial frame alone is insufficient: rolling MPC can sometimes
    form a valid grasp from it. Only the new power-envelope path with a load
    separation certificate takes this early exit.
    """
    selected = result.get('selected', {})
    info = selected.get('initialization', {})
    if selected.get('termination') == 'initialization_failed':
        return dict(status='initialization_failed', settled=False, gravity_test_ran=False,
            dropped=False, retained=False, verified_grasp=False,
            failure_reasons=info.get('failure_reasons', []))
    if (not getattr(args,'load_conditioning',False) or
            info.get('initialization_valid') is not False or selected.get('feasible',False)):
        return None
    refinements = info.get('template_fit',{}).get('object_contact_refinements',[])
    power_input = any(r.get('method')=='CUDA_dense_power_cylinder_input_IK' and r.get('accepted')
                      for r in refinements)
    certificate = (info.get('load_conditioning') or {}).get('force_cone_diagnostic',{})
    if not power_input or not certificate.get('static_load_excluded'):
        return None
    from retarget.grasp_initialization import initialization_failure_reasons
    return dict(status='initialization_failed', settled=False, gravity_test_ran=False,
        retained=False, dropped=False, verified_grasp=False, formed_geometry=None,
        failure_stage='input_construction', failure_reasons=initialization_failure_reasons(info),
        reason='No valid initialized grasp and the shared solver did not repair it; rebuild inputs before release',
        visualization_seconds=0.)


def simulate_case(case,args):
    from retarget.grasp_simulation import run_drop_test
    case=Path(case)
    result=json.loads((case/'result.json').read_text())
    report=invalid_preload_report(result,args)
    preview = GraspPreview(case,args.viewer)
    try:
        if report is None:
            report,arrays = run_drop_test(case,args,on_frame=preview.update)
        else:
            with np.load(case/'grasp.npz') as saved:
                preview.update(saved['qpos'],saved['qvel'],0.,'initialization failed')
            print(f'  [physics] skipped: initialization_failed ({", ".join(report["failure_reasons"])})',flush=True)
    finally:
        preview.close()
    rendering_started=time.perf_counter()
    if args.render and report['status']!='initialization_failed':
        render_drop_video(case,arrays,report)
        if report['settled']:
            from retarget.grasp import display
            model=mujoco.MjModel.from_xml_path(str(Path(case)/'scene.xml'))
            display(model,arrays['settled_qpos'],Path(case),render=True,image_name='formed.png')
    report['visualization_seconds']=time.perf_counter()-rendering_started
    case=Path(case)
    (case/'drop_test.json').write_text(json.dumps(report,indent=2))
    result['drop_test']=report
    (case/'result.json').write_text(json.dumps(result,indent=2))
    # Re-testing an existing grasp must also update its batch summary.
    # A selected retry can be referenced by more than one ancestor batch.
    for batch in case.parents:
        if not (batch/'summary.json').is_file():
            continue
        summary=json.loads((batch/'summary.json').read_text())
        changed=False
        for row in summary:
            if Path(row['case']).resolve()==case.resolve():
                row['drop_test']=report;changed=True
        if not changed:
            continue
        (batch/'summary.json').write_text(json.dumps(summary,indent=2))
        path=batch/'summary.csv'
        if path.is_file():
            with path.open(newline='') as source:
                reader=csv.DictReader(source);fields=list(reader.fieldnames);rows=list(reader)
            fields += [k for k in ('drop_status','retained','dropped','formed_feasible','verified_grasp') if k not in fields]
            for row in rows:
                if Path(row['case']).resolve()==case.resolve():
                    row.update(drop_status=report['status'],retained=report['retained'],dropped=report['dropped'],
                        formed_feasible=(report['formed_geometry'] or {}).get('feasible'),verified_grasp=report['verified_grasp'])
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return report

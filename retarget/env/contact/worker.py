"""Native MuJoCo 4 service with independent rollout processes and one execution state."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.connection import Client
import traceback
import mujoco
import numpy as np
from retarget.env.contact.native import restore, snapshot, contact_records
from retarget.env.contact.native_evaluation import evaluate_candidate

_rollout_model = None


def load_model(path: str) -> mujoco.MjModel:
    if not hasattr(mujoco.mjtEnableBit, 'mjENBL_SPECULATIVE'):
        raise RuntimeError(f'Native worker requires MuJoCo 4 speculative support, got {mujoco.__version__}')
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    return model


def initialize_rollouts(path: str) -> None:
    global _rollout_model
    _rollout_model = load_model(path)


def evaluate_rollout(job: tuple) -> dict:
    return evaluate_candidate(_rollout_model, *job)


def main() -> None:
    connection = Client(sys.argv[1], family='AF_UNIX', authkey=bytes.fromhex(os.environ['RETARGET_WORKER_KEY']))
    model = state = pool = None
    try:
        while True:
            request = connection.recv()
            try:
                command = request['command']
                if command == 'close':
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                        pool = None
                    connection.send({'ok': True, 'result': None})
                    break
                if command == 'init':
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                    workers = request.get('workers', 1)
                    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
                        raise ValueError('CPU rollout worker count must be a positive integer')
                    workers = min(workers, max(1, (os.cpu_count() or 1) - 1))
                    model = load_model(request['model'])
                    state = mujoco.MjData(model)
                    restore(model, state, request['state'])
                    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                        initializer=initialize_rollouts, initargs=(request['model'],)) if workers > 1 else None
                    result = dict(version=mujoco.__version__, speculative=True, workers=workers,
                                  state=snapshot(state), contacts=contact_records(model, state))
                elif command == 'evaluate':
                    jobs = [(request['state'], controls, request['packet'], events, request['dynamics_cost'])
                            for controls, events in zip(request['controls'], request['events'])]
                    result = list(pool.map(evaluate_rollout, jobs, chunksize=1)) if pool else [evaluate_candidate(model, *job) for job in jobs]
                elif command == 'execute':
                    rows = []
                    for u in request['controls']:
                        state.ctrl[:] = u
                        mujoco.mj_step(model, state)
                        mujoco.mj_forward(model, state)
                        rows.append(dict(state=snapshot(state), sensordata=state.sensordata.copy(),
                            actuator_force=state.actuator_force.copy(), contacts=contact_records(model, state)))
                        if not np.isfinite(state.qpos).all() or any(w.number for w in state.warning):
                            raise FloatingPointError('Nonfinite state or native MuJoCo warning during execution')
                    result = rows
                else:
                    raise ValueError(f'Unknown worker command: {command}')
                connection.send(dict(ok=True, result=result))
            except Exception:
                connection.send(dict(ok=False, error=traceback.format_exc(), partial_rows=rows if command == 'execute' else []))
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        connection.close()


if __name__ == '__main__':
    main()

import numpy as np
import pytest
from moxian_web.wuji_calibration_solver import WujiCalibrationLbfgs,NativeLibrary,ELF_SHA256


def test_actual_wuji_solver_handles_bounds_and_fixed_variables():
    solver=WujiCalibrationLbfgs(3);solver.set_lower_bounds([-3,0,2]);solver.set_upper_bounds([3,.3,2])
    calls=[]
    def objective(x,g):
        calls.append(x.copy())
        if g.size:g[:]=2*(x-[1,.5,2])
        return float(np.sum((x-[1,.5,2])**2))
    solver.set_min_objective(objective);x=solver.optimize([0,.1,2])
    np.testing.assert_allclose(x,[1,.3,2],atol=1e-8)
    assert calls and solver.last_optimize_result()>0
    assert solver.provenance['elf_sha256']==ELF_SHA256
    assert solver.provenance['native_solver_reused']
    assert not solver.provenance['native_caliber_objective_reused']


def test_cancel_and_callback_failure_return_safely_through_native_stop():
    solver=WujiCalibrationLbfgs(1)
    def cancelled(x,g):raise InterruptedError('cancelled')
    solver.set_min_objective(cancelled)
    with pytest.raises(InterruptedError,match='cancelled'):solver.optimize([1])
    assert solver.last_optimize_result()==-5
    def broken(x,g):raise RuntimeError('objective failed')
    solver.set_min_objective(broken)
    with pytest.raises(RuntimeError,match='objective failed'):solver.optimize([1])
    assert solver.last_optimize_result()==-5


def test_wrong_binary_is_rejected_before_loading(tmp_path):
    bad=tmp_path/'wrong.so';bad.write_bytes(b'not the pinned native solver')
    with pytest.raises(ValueError,match='指纹'):NativeLibrary(bad)

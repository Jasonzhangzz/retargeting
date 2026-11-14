"""Persistent local process boundary between GPU MuJoCo 3 and native MuJoCo 4."""
from multiprocessing.connection import Listener
from pathlib import Path
import os
import secrets
import signal
import subprocess
import tempfile


class NativeWorkerError(RuntimeError):
    def __init__(self, message: str, partial_rows: list):
        super().__init__(message)
        self.partial_rows = partial_rows


class NativeClient:
    def __init__(self, model: Path, state: dict, directory: Path, interpreter: Path | None = None, *, workers: int = 1):
        root = Path(__file__).resolve().parents[3]
        interpreter = interpreter or root / '.venv-contact4/bin/python'
        if not interpreter.is_file():
            raise FileNotFoundError(f'Run scripts/setup_contact.sh: missing {interpreter}')
        self.temporary = tempfile.TemporaryDirectory(prefix='retarget-native-')
        address = str(Path(self.temporary.name) / 'worker.sock')
        key = secrets.token_bytes(32)
        self.listener = Listener(address, family='AF_UNIX', authkey=key)
        self.listener._listener._socket.settimeout(60)
        self.log = (directory / 'cpu4-worker.log').open('w')
        self.process = subprocess.Popen([str(interpreter), str(Path(__file__).with_name('worker.py')), address],
            env={**os.environ, 'RETARGET_WORKER_KEY': key.hex(), 'OMP_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}, stdout=self.log, stderr=self.log,
            start_new_session=True)
        try:
            self.connection = self.listener.accept()
            self.info = self.call('init', model=str(model.resolve()), state=state, workers=workers)
        except BaseException:
            self.process.terminate()
            self.process.wait()
            self.listener.close()
            self.log.close()
            self.temporary.cleanup()
            raise

    def call(self, command: str, **request):
        self.connection.send(dict(command=command, **request))
        if not self.connection.poll(600):
            raise TimeoutError(f'Native worker timeout: {command}; inspect cpu4-worker.log')
        response = self.connection.recv()
        if not response['ok']:
            raise NativeWorkerError(response['error'], response.get('partial_rows', []))
        return response['result']

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                try:
                    self.call('close')
                except (BrokenPipeError, EOFError, NativeWorkerError):
                    pass  # Keep the original run failure and its partial artifacts.
        finally:
            self.connection.close()
            self.listener.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            self.log.close()
            self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

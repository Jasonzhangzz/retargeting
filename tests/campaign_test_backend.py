import json
import os
from pathlib import Path
import time


class FakeWorker:
    def __init__(self, device, config, progress):
        self.device = device
        self.output = Path(config['output'])
        (self.output / ('ready-' + device.replace(':', '-'))).touch()
        deadline = time.monotonic() + 10
        while len(list(self.output.glob('ready-*'))) < config['gate_count']:
            if time.monotonic() > deadline:
                raise TimeoutError('Campaign workers did not become ready')
            time.sleep(.01)

    def completed(self, job):
        return None

    def solve(self, job, progress):
        if job.get('crash'):
            os._exit(17)
        time.sleep(.03)
        directory = self.output / job['id']
        directory.mkdir()
        receipt = dict(success=True, samples=job['samples'], stable=3, geometry_accepted=1,
                       device=self.device, pid=os.getpid())
        (directory / 'result.json').write_text(json.dumps(receipt))
        if job.get('stop'):
            (self.output / 'STOP').touch()
        return receipt

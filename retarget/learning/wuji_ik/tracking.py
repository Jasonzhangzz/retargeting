"""Export committed training curves to W&B without running a model or using a GPU."""
import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DEFAULT_BASE_URL = 'https://wandb.plat.x2robot.com'


@contextmanager
def server_verified_auth(base_url):
    """Support this server's older key format after authenticating it with the server.

    SDK 0.31 rejects some server-issued keys purely on secret length. Do not change
    installed SDK files or authentication checks: the server must first accept the
    exact key, and all other keys retain the SDK's original format validation.
    """
    key = os.environ.get('WANDB_API_KEY')
    if not key or base_url != DEFAULT_BASE_URL:
        yield
        return
    try:
        from wandb.sdk.lib.wbauth import validation
    except ImportError:
        yield
        return
    if validation.check_api_key(key) is None:
        yield
        return
    import base64
    import urllib.request
    request = urllib.request.Request(base_url+'/graphql',
        data=json.dumps({'query': 'query { viewer { username } }'}).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Basic '+base64.b64encode(('api:'+key).encode()).decode()})
    with urllib.request.urlopen(request, timeout=20) as response:
        viewer = json.load(response).get('data', {}).get('viewer')
    if not viewer or not viewer.get('username'):
        raise ValueError('The W&B server did not authenticate the supplied key')
    original = validation.check_api_key
    validation.check_api_key = lambda candidate: None if candidate == key else original(candidate)
    try:
        yield
    finally:
        validation.check_api_key = original


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not have finished the final line yet.
            if index != len(lines)-1:
                raise
    return records


def canonical_training(records):
    """Later restarts replace abandoned updates after their checkpoint boundary."""
    result = {}
    previous = -1
    for record in records:
        step = int(record['step'])
        if step <= previous:
            result = {k: v for k, v in result.items() if k < step}
        result[step] = record
        previous = step
    return result


def flatten_scalars(value, prefix):
    result = {}
    if isinstance(value, dict):
        for name, child in value.items():
            result.update(flatten_scalars(child, prefix+'/'+str(name)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(flatten_scalars(child, prefix+'/'+str(index)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        result[prefix] = value
    return result


def curve_records(directory, committed_step):
    """Merge train/validation at each optimizer step; never publish uncommitted q updates."""
    directory = Path(directory)
    training = canonical_training(read_jsonl(directory/'train.jsonl'))
    validation = read_jsonl(directory/'validation.jsonl')
    latest = directory/'validation.json'
    if latest.exists():
        validation.append(json.loads(latest.read_text()))
    events = {}
    for step, record in training.items():
        if step > committed_step:
            continue
        event = dict(global_step=step, epoch=record['epoch']+1)
        for name in ('loss', 'regression_loss', 'smoothness_loss', 'weighted_smoothness_loss',
                     'smoothness_weight', 'valid_smooth_joint_elements', 'grad_norm', 'muon_lr', 'adamw_lr', 'valid_joint_elements'):
            if name in record:
                event['train/'+name] = record[name]
        event['train/'+record['objective']] = record['loss']
        if 'effective_frames_by_rank' in record:
            event['train/global_effective_frames'] = sum(record['effective_frames_by_rank'])
        events[step] = event
    for record in validation:
        step = int(record['step'])
        if step > committed_step or step not in training:
            continue
        event = events.setdefault(step, dict(global_step=step, epoch=training[step]['epoch']+1))
        event.update(flatten_scalars(record.get('metrics', {}), 'val'))
        for name in ('frames_per_second', 'mean_sequence_latency_ms',
                     'batch_amortized_sequence_p50_ms', 'batch_amortized_sequence_p95_ms',
                     'peak_allocated_bytes', 'frames', 'flow_steps'):
            value = record.get(name)
            if isinstance(value, (int, float)):
                event['val/'+name] = value
    return [events[step] for step in sorted(events)]


class CheckpointReader:
    def __init__(self):
        self.cache = {}

    def step(self, path):
        path = Path(path)
        if not path.exists():
            return None
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if path not in self.cache or self.cache[path][0] != signature:
            import torch
            # Only inspect metadata; tensors remain mapped on CPU and are never uploaded.
            state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            self.cache[path] = (signature, int(state['step']))
        return self.cache[path][1]


class WandbCurves:
    def __init__(self, campaign, project='wuji2-ik', entity=None, base_url=DEFAULT_BASE_URL, sdk=None,
                 name_prefix=None, keepalive=False):
        self.root = Path(campaign).resolve()
        self.campaign = json.loads((self.root/'configuration.json').read_text())
        self.base_url = base_url.rstrip('/')
        self.project = project
        self.entity = entity
        self.path = self.root/'wandb_sync.json'
        self.state = dict(base_url=self.base_url, project=project, entity=entity, runs={})
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if (self.state['base_url'] != self.base_url or self.state['project'] != project
                    or (entity is not None and self.state['entity'] != entity)):
                raise ValueError('Existing W&B destination differs from the requested destination')
            self.entity = self.state['entity']
        if sdk is None:
            import wandb as sdk
        self.sdk = sdk
        self.reader = CheckpointReader()
        self.name_prefix = name_prefix if name_prefix is not None else self.state.get('run_name_prefix', '')
        self.state['run_name_prefix'] = self.name_prefix
        self.keepalive = keepalive
        self.open_runs = {}

    def close(self):
        for variant in list(self.open_runs):
            self.open_runs.pop(variant).finish()

    def sync(self, variant):
        directory = self.root/variant
        committed = self.reader.step(directory/'latest.pt')
        if committed is None:
            return None
        events = curve_records(directory, committed)
        previous = self.state['runs'].get(variant, {})
        queue = json.loads((self.root/'STATUS.json').read_text())
        job = queue['jobs'][variant]
        training_state = 'paused' if queue['status'] == 'stopped' else job['status']
        keep_open = self.keepalive and queue['status'] == 'training' and queue.get('active') == variant
        if keep_open and job.get('started'):
            if (directory/'train.jsonl').stat().st_mtime < datetime.fromisoformat(job['started']).timestamp():
                training_state = 'resuming'
        display_name = self.campaign.get('wandb_run_names', {}).get(
            variant, self.name_prefix+' / '+variant if self.name_prefix else variant)
        test_path = directory/'test_snapshot.json'
        test_snapshot = json.loads(test_path.read_text()) if test_path.exists() else None
        test_snapshot_id = None
        if test_snapshot:
            if test_snapshot['data_signature'] != self.campaign['data_signature']:
                raise ValueError('Test snapshot/campaign data signature mismatch')
            test_snapshot_id = [test_snapshot['checkpoint_step'], test_snapshot['evaluated_at']]
        if (previous.get('last_logged_step', -1) >= committed
                and previous.get('training_state') == training_state
                and previous.get('display_name') == display_name
                and previous.get('test_snapshot_id') == test_snapshot_id
                and ((keep_open and variant in self.open_runs) or (not keep_open and variant not in self.open_runs))):
            return previous
        config = json.loads((directory/'configuration.json').read_text())
        identifier = previous.get('id') or hashlib.sha256(
            (str(self.root)+self.campaign['data_signature']+variant).encode()).hexdigest()[:12]
        settings = self.sdk.Settings(base_url=self.base_url, disable_code=True, disable_git=True,
                                     save_code=False, console='off', x_disable_stats=True, init_timeout=90)
        run = self.open_runs.get(variant)
        if run is None:
            with server_verified_auth(self.base_url):
                # A completed variant may receive test results while the next run is
                # kept open. W&B's default otherwise returns that active run, even
                # when a different id was requested, corrupting its training history.
                run = self.sdk.init(project=self.project, entity=self.entity, id=identifier, resume='allow',
                    reinit='create_new',
                    name=display_name, group=self.name_prefix or self.root.name, job_type='training', dir=str(directory),
                    config={key: config[key] for key in ('train', 'model', 'data_signature', 'world_size', 'total_steps',
                        'initialization', 'initialization_optimizer_moments') if key in config},
                    settings=settings)
            if getattr(run, 'id', identifier) != identifier:
                raise RuntimeError('W&B returned a different run; refusing to mix variant histories')
        try:
            run.define_metric('global_step')
            run.define_metric('train/*', step_metric='global_step')
            run.define_metric('val/*', step_metric='global_step')
            run.define_metric('epoch', step_metric='global_step')
            # The server's resumed step is authoritative if a previous sync process crashed.
            next_step = run.step
            for event in events:
                if event['global_step'] >= next_step:
                    run.log(event, step=event['global_step'])
            run.summary['training_state'] = training_state
            run.summary['checkpoint_step'] = committed
            run.summary['planned_epochs'] = config['train']['epochs']
            run.summary['information_condition'] = ('past_and_present' if variant == 'temporal_causal' else
                'past_present_and_future' if variant == 'temporal_bidirectional' else 'current_frame_only')
            if test_snapshot:
                for name, value in flatten_scalars(test_snapshot['report']['metrics'], 'test_snapshot').items():
                    run.summary[name] = value
                run.summary['test_snapshot/checkpoint_step'] = test_snapshot['checkpoint_step']
                run.summary['test_snapshot/evaluated_at'] = test_snapshot['evaluated_at']
                run.summary['test_snapshot/training_complete'] = test_snapshot['training_complete']
            url = run.url
            resolved_entity = run.entity
        except BaseException:
            self.open_runs.pop(variant, None)
            run.finish(exit_code=1)
            raise
        if keep_open:
            self.open_runs[variant] = run
        else:
            self.open_runs.pop(variant, None)
            run.finish()
        receipt = dict(id=identifier, url=url, last_logged_step=committed,
                       display_name=display_name, training_state=training_state,
                       test_snapshot_id=test_snapshot_id,
                       train_points=len(events), validation_points=sum('val/overall/normalized_mse' in e for e in events))
        self.state['runs'][variant] = receipt
        self.state['entity'] = self.entity = resolved_entity
        temp = self.path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(self.state, indent=2)+'\n')
        temp.replace(self.path)
        return receipt

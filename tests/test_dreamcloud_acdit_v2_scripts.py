"""Dry-run Dreamcloud launchers using local SSH/torchrun/Python stand-ins."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'dreamcloud'
SUFFIX = 'ACDiTSeisDimReconNeRF2_t_p4_i128_overlap64_NeRFBands6_31shots_ref2.sh'
MOCK_PROGRAM = '''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

kind = Path(sys.argv[0]).name
node = os.environ.get('MOCK_NODE', 'master')
if kind == 'ssh':
    environment = dict(os.environ, MOCK_NODE=sys.argv[1])
    result = subprocess.run(['bash', '-c', sys.argv[2]], env=environment)
    sys.exit(result.returncode)
record = dict(kind=kind, node=node, args=sys.argv[1:])
with open(os.environ['MOCK_EVENTS'], 'a') as stream:
    stream.write(json.dumps(dict(record, phase='start')) + '\\n')
if kind == 'torchrun':
    time.sleep(0.03 if node != 'master' else 0.001)
with open(os.environ['MOCK_EVENTS'], 'a') as stream:
    stream.write(json.dumps(dict(record, phase='end')) + '\\n')
if node == os.environ.get('MOCK_FAIL_NODE'):
    sys.exit(7)
'''


@pytest.fixture
def launcher(tmp_path):
    """Provide a launcher that records subprocess calls without SSH, GPUs or data writes.

    Args:
        tmp_path: Pytest temporary directory for executable stand-ins and event logs.

    Returns:
        Callable accepting a job name, worker list and optional simulated failed node.
    """
    bin_dir = tmp_path / 'mock bin'
    bin_dir.mkdir()
    mock = bin_dir / 'mock.py'
    mock.write_text(f'#!{sys.executable}\n' + MOCK_PROGRAM)
    mock.chmod(0o755)
    for name in ('ssh', 'torchrun', 'python'):
        (bin_dir / name).symlink_to(mock)
    code_path = tmp_path / 'code with spaces'
    code_path.mkdir()
    event_path = tmp_path / 'events.jsonl'

    def launch(job, nodes, failed_node=''):
        """Run one real shell launcher against local subprocess stand-ins.

        Args:
            job: build, train or recon script selector.
            nodes: Comma-separated remote worker names, excluding master.
            failed_node: Worker name whose Python/torchrun stand-in exits unsuccessfully.

        Returns:
            Completed shell process and ordered subprocess event records.
        """
        script = 'build_shot_dataset_31shots_ref2.sh' if job == 'build' else f'{job}_{SUFFIX}'
        environment = dict(
            os.environ,
            PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
            CODE_PATH=str(code_path), PROJ_DIR=str(tmp_path / 'project with spaces'),
            RUN_DIR=str(tmp_path / 'outputs'), DATA_DIR=str(tmp_path / 'dataset with spaces'),
            PYTHON_BIN=str(bin_dir / 'python'), TORCHRUN_BIN=str(bin_dir / 'torchrun'),
            MASTER='master', MASTER_ADDR='192.0.2.1', MASTER_PORT='29517',
            NPROC_PER_NODE='2', NODES_LIST=nodes,
            TRAIN_RUN_DIR=str(tmp_path / 'training run'),
            BATCH_SIZE='32', NUM_EPOCHS='1000', SAVE_EVERY_EPOCHS='100',
            FIRST_EPOCH='100', LAST_EPOCH='200', EPOCH_STEP='100',
            SEGY=str(tmp_path / 'source shot.sgy'), RANDOM_SEGY=str(tmp_path / 'random shot.sgy'),
            RANDOM_SEED='23', SPLIT_SEED='17',
            MOCK_EVENTS=str(event_path), MOCK_FAIL_NODE=failed_node,
        )
        result = subprocess.run(['bash', str(SCRIPTS / script)], env=environment,
                                capture_output=True, text=True, timeout=30)
        records = [json.loads(line) for line in event_path.read_text().splitlines()]
        return result, records

    return launch


@pytest.mark.parametrize('nodes', ['', 'worker1,worker2', 'worker1,worker2,worker3,worker4'])
def test_build_assigns_each_dataset_once(launcher, nodes):
    """Eight output directories have unique owners, after one random SEG-Y generation.

    Args:
        launcher: Fixture invoking real scripts with subprocess stand-ins.
        nodes: Worker host list, including single-node execution.
    """
    result, events = launcher('build', nodes)
    assert result.returncode == 0, result.stderr
    calls = [event for event in events if event['phase'] == 'start']
    random_calls = [e for e in calls if Path(e['args'][0]).name == 'BuildRandomSegy.py']
    builds = [e for e in calls if Path(e['args'][0]).name == 'BuildShotDataset2.py']
    extracts = [e for e in calls if Path(e['args'][0]).name == 'ExtractShot2.py']
    assert len(random_calls) == 1 and random_calls[0]['node'] == 'master'
    assert Path(events[1]['args'][0]).name == 'BuildRandomSegy.py'
    assert events[1]['phase'] == 'end'
    assert len(builds) == len(extracts) == 8
    outputs = [e['args'][e['args'].index('--output_dir') + 1] for e in builds]
    assert len(set(outputs)) == 8
    hosts = ['master', *nodes.split(',')] if nodes else ['master']
    expected_outputs = [f'{prefix}shot_dataset{patch}_overlap{overlap}_31shots_ref2'
                        for patch, overlap in [(64, 32), (64, 48), (128, 64), (128, 96)]
                        for prefix in ('', 'random_')]
    for build, output in zip(builds, outputs):
        name = Path(output).name
        index = expected_outputs.index(name)
        assert build['node'] == hosts[index % len(hosts)]
        args = build['args']
        assert args[args.index('--gen-ref') + 1] == '2'
        assert args[args.index('--seed') + 1] == '17'
        assert ('--normalize' in args) == (not name.startswith('random_'))
        extract = next(e for e in extracts if e['args'][e['args'].index('--output_dir') + 1] == output + '/shot')
        assert extract['node'] == build['node']


@pytest.mark.parametrize('job', ['train', 'recon'])
@pytest.mark.parametrize('nodes', ['', 'worker1,worker2'])
def test_torchrun_ranks_and_master_only_postprocessing(launcher, job, nodes):
    """All nodes launch the same model job; reconstruction waits before merging.

    Args:
        launcher: Fixture invoking real scripts with subprocess stand-ins.
        job: Training or reconstruction launcher.
        nodes: Worker host list, including single-node execution.
    """
    result, events = launcher(job, nodes)
    assert result.returncode == 0, result.stderr
    node_count = 1 + len(nodes.split(',')) if nodes else 1
    starts = [e for e in events if e['kind'] == 'torchrun' and e['phase'] == 'start']
    assert len(starts) == node_count * (2 if job == 'recon' else 1)
    for call in starts:
        args = call['args']
        assert f'--nnodes={node_count}' in args
        assert '--nproc_per_node=2' in args
        assert '--master_addr=192.0.2.1' in args and '--master_port=29517' in args
        rank = 0 if call['node'] == 'master' else int(call['node'][-1])
        assert f'--node_rank={rank}' in args
        assert any(Path(arg).name == 'ACDiTSeisDimReconNeRF2.py' for arg in args)
        for option in ('--ref_dir1', '--ref_dim_dir1', '--ref_dir2', '--ref_dim_dir2'):
            assert option in args
        assert args[args.index('--batch_size') + 1] == '32'
    postprocessing = [e for e in events if e['kind'] == 'python' and e['phase'] == 'start']
    if job == 'train':
        assert postprocessing == []
    else:
        assert len(postprocessing) == 4
        assert all(e['node'] == 'master' for e in postprocessing)
        for epoch in ('00100', '00200'):
            finished = [i for i, e in enumerate(events) if e['kind'] == 'torchrun'
                        and e['phase'] == 'end' and any(arg.endswith('checkpoint_epoch_' + epoch) for arg in e['args'])]
            merge = next(i for i, e in enumerate(events) if e['kind'] == 'python'
                         and e['phase'] == 'start' and Path(e['args'][0]).name == 'ReconShotDataset2.py'
                         and any(arg.endswith('valid_ema_epoch_' + epoch) for arg in e['args']))
            assert len(finished) == node_count and max(finished) < merge


def test_failed_sampling_worker_prevents_postprocessing(launcher):
    """A worker failure is propagated without merging or scoring incomplete outputs.

    Args:
        launcher: Fixture invoking real scripts with a simulated worker failure.
    """
    result, events = launcher('recon', 'worker1', failed_node='worker1')
    assert result.returncode != 0
    assert not any(e['kind'] == 'python' for e in events)

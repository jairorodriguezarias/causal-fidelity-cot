import io
import json
import subprocess
from unittest.mock import patch

import pytest

from scripts.campaign_resources import prepare_stage


def test_generation_requires_confirmed_container_stop():
    with patch('scripts.campaign_resources.subprocess.run') as run, \
            patch('scripts.campaign_resources.subprocess.check_output', return_value='false\n'):
        prepare_stage('distill-32b-generate', 'nemoclaw-vllm')
    run.assert_called_once_with(['docker', 'stop', '--timeout', '30', 'nemoclaw-vllm'],
                                check=True, timeout=90)


def test_generation_refuses_container_that_restarted():
    with patch('scripts.campaign_resources.subprocess.run'), \
            patch('scripts.campaign_resources.subprocess.check_output', return_value='true\n'):
        with pytest.raises(RuntimeError, match='refusing concurrent generation'):
            prepare_stage('distill-32b-generate', 'nemoclaw-vllm')


def test_stop_failure_propagates():
    with patch('scripts.campaign_resources.subprocess.run',
               side_effect=subprocess.CalledProcessError(1, 'docker')):
        with pytest.raises(subprocess.CalledProcessError):
            prepare_stage('distill-32b-generate', 'nemoclaw-vllm')


@pytest.mark.parametrize('stage', ['judge-validation', 'distill-32b-judge'])
def test_judge_waits_for_expected_model(stage):
    responses = [OSError('starting'),
                 io.StringIO(json.dumps({'data': [{'id': 'wrong-model'}]})),
                 io.StringIO(json.dumps({'data': [{'id': 'expected-model'}]}))]
    with patch('scripts.campaign_resources.subprocess.run') as run, \
            patch('scripts.campaign_resources.urllib.request.urlopen', side_effect=responses) as request, \
            patch('scripts.campaign_resources.time.sleep') as sleep:
        prepare_stage(stage, 'nemoclaw-vllm', model='expected-model')
    run.assert_called_once_with(['docker', 'start', 'nemoclaw-vllm'], check=True, timeout=90)
    assert request.call_count == 3
    assert sleep.call_count == 2


def test_unavailable_judge_times_out():
    with patch('scripts.campaign_resources.subprocess.run'), \
            patch('scripts.campaign_resources.time.monotonic', side_effect=[0, 1, 601]), \
            patch('scripts.campaign_resources.urllib.request.urlopen', side_effect=OSError('unavailable')), \
            patch('scripts.campaign_resources.time.sleep'):
        with pytest.raises(RuntimeError, match='did not become ready'):
            prepare_stage('distill-32b-judge', 'nemoclaw-vllm')


@pytest.mark.parametrize('stage,container', [('distill-32b-generate', None),
                                           ('distill-32b-audit', 'nemoclaw-vllm'),
                                           ('distill-32b-download', 'nemoclaw-vllm')])
def test_unmanaged_stages_do_not_touch_container(stage, container):
    with patch('scripts.campaign_resources.subprocess.run') as run:
        prepare_stage(stage, container)
    run.assert_not_called()

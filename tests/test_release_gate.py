"""Exercise the real run auditor without local runs, weights or network access."""
import importlib.util
import json
from pathlib import Path

import pytest

from dataset_builder import fingerprint
from evaluation import parse_output


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('audit_run_fixture', root / 'scripts/audit_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def pair(name):
        return {'id': name, 'premise': f'Premise {name}',
                'entail_hypothesis': f'Entail {name}', 'contra_hypothesis': f'Contra {name}'}

    splits = {name: [pair(name)] for name in ('train', 'calibration', 'test')}
    dataset = {'splits': splits, 'sha256': fingerprint(splits)}
    write(tmp_path / 'artifacts/dataset.json', dataset)
    folder = tmp_path / 'runs/smoke'
    config = {'evaluation_split': 'calibration', 'eval_pairs': 1, 'train_pairs': 1,
              'layer_fractions': [], 'alphas': [], 'modes': [], 'control_replicates': 1}
    write(folder / 'manifest.json', {
        'dataset_sha256': dataset['sha256'], 'config': config, 'source_sha256': {},
        'eval_ids': ['calibration'], 'train_ids': ['train'], 'model_name': 'test-model',
        'model': {'id': 'fixture/tiny', 'revision': 'fixed'},
    })
    write(tmp_path / 'hf_cache/hub/models--fixture--tiny/snapshots/fixed/config.json',
          {'num_hidden_layers': 2})
    write(folder / 'status.json', {'state': 'complete', 'records': 1})
    text = 'The hypothesis follows.</think>\nFINAL: ENTAILMENT'
    record = {'key': 'baseline/calibration/entail', 'example_id': 'calibration',
              'condition': {'name': 'baseline', 'kind': 'baseline'},
              'generated_text': text, 'generated_tokens': 1, 'generated_token_ids': [1],
              'parsed': parse_output(text).asdict()}
    (folder / 'records.jsonl').write_text(json.dumps(record) + '\n')
    return module, folder


def test_smoke_cannot_be_passed_off_as_held_out_study(saved_run):
    module, folder = saved_run
    assert module.audit(folder)['passed']
    result = module.audit(folder, require_test=True, output_name='release-run-audit.json')
    assert not result['passed']
    assert 'Not held-out test data' in result['errors']
    assert 'Fewer than 500 construction pairs' in result['errors']
    assert 'Fewer than 50 evaluation examples' in result['errors']


def test_auditor_rejects_running_generation(saved_run):
    module, folder = saved_run
    (folder / 'status.json').write_text(json.dumps({'state': 'running', 'records': 1}))
    result = module.audit(folder)
    assert not result['passed']
    assert 'Run is incomplete' in result['errors']

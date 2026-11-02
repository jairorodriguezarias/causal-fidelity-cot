"""Reproducible activation and CoT interventions with append-only checkpoints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset_builder import fingerprint, load_contrast_dataset
from evaluation import parse_output

ROOT = Path(__file__).resolve().parent


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def layer_indices(n_layers, fractions):
    return sorted(set(round(f * (n_layers-1)) for f in fractions))


def prompt_for(tokenizer, pair, variant='entail'):
    content = (
        'Determine the relationship between the premise and hypothesis. '
        'ENTAILMENT means the hypothesis follows from the premise. '
        'CONTRADICTION means the hypothesis conflicts with the premise. '
        'The examples in this task belong to one of these two classes. '
        'Reason briefly, then give the final answer on its own line in exactly this format: '
        'FINAL: ENTAILMENT or FINAL: CONTRADICTION.\n\n'
        f"Premise: {pair['premise']}\nHypothesis: {pair[variant + '_hypothesis']}"
    )
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                           tokenize=False, add_generation_prompt=True)
    if not prompt.rstrip().endswith('<think>'):
        prompt += '<think>\n'
    return prompt


def load_model(config):
    kwargs = {'revision': config['revision'], 'cache_dir': str(ROOT / 'hf_cache/hub'),
              'local_files_only': True}
    tokenizer = AutoTokenizer.from_pretrained(config['id'], **kwargs)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {'dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                   'device_map': {'': 0} if torch.cuda.is_available() else {'': 'cpu'},
                   'attn_implementation': 'sdpa'}
    if config['quantization'] == 'nf4':
        if not torch.cuda.is_available():
            raise RuntimeError('NF4 study requires CUDA')
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(config['id'], **kwargs, **load_kwargs)
    model.eval()
    return model, tokenizer


def hidden_tensor(output):
    return output[0] if isinstance(output, tuple) else output


class SteeringHook:
    """Change the hidden state predicting tokens 1..K, including prompt prefill."""
    def __init__(self, delta, k_tokens):
        self.delta, self.k_tokens, self.calls, self.applied = delta, k_tokens, 0, 0

    def __call__(self, module, inputs, output):
        if self.calls < self.k_tokens:
            hidden = hidden_tensor(output).clone()
            delta = self.delta.to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, -1, :] = hidden[:, -1, :] + delta
            output = (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            self.applied += 1
        self.calls += 1
        return output


@contextmanager
def steering_hook(model, layer, delta, k_tokens):
    hook = SteeringHook(delta, k_tokens)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()


def intervention_delta(direction, alpha, mode, example_ids, seed, layer):
    direction = direction.float().cpu()
    norm = direction.norm()
    if not torch.isfinite(norm) or norm < 1e-8:
        raise ValueError('Degenerate steering direction')
    unit = direction / norm
    vectors = []
    for example_id in example_ids:
        if mode == 'steer':
            vector = direction
        else:
            generator = torch.Generator().manual_seed(stable_seed(seed, example_id, layer, mode))
            vector = torch.randn(direction.shape, generator=generator)
            if mode == 'orthogonal':
                vector = vector - torch.dot(vector, unit) * unit
            elif mode != 'gaussian':
                raise ValueError(f'Unknown control: {mode}')
            # Gaussian directions are normalized for an exact magnitude match.
            vector = vector / vector.norm() * norm
        vectors.append(alpha * vector)
    return torch.stack(vectors)


def extract_vectors(model, tokenizer, pairs, layers, batch_size):
    if not layers:
        return {}, {}
    sums = {variant: {l: torch.zeros(model.config.hidden_size, dtype=torch.float32) for l in layers}
            for variant in ('entail', 'contra')}
    norms = {l: 0.0 for l in layers}
    device = model.get_input_embeddings().weight.device
    for variant in ('entail', 'contra'):
        for start in range(0, len(pairs), batch_size):
            prompts = [prompt_for(tokenizer, p, variant) for p in pairs[start:start+batch_size]]
            inputs = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
            handles = []
            def make_hook(layer):
                def capture(module, inp, output):
                    activation = hidden_tensor(output)[:, -1, :].detach().float()
                    sums[variant][layer].add_(activation.sum(0).cpu())
                    norms[layer] += activation.norm(dim=-1).sum().item()
                return capture
            try:
                for layer in layers:
                    handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
                with torch.inference_mode():
                    model(**inputs, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            print(json.dumps({'stage': 'vectors', 'variant': variant,
                              'done': min(start+batch_size, len(pairs)), 'total': len(pairs)}), flush=True)
    vectors = {l: (sums['contra'][l] - sums['entail'][l]) / len(pairs) for l in layers}
    metadata = {str(l): {'difference_norm': vectors[l].norm().item(),
                         'mean_activation_norm': norms[l] / (2*len(pairs))} for l in layers}
    return vectors, metadata


def generate_batch(model, tokenizer, prompts, generation, seed, delta=None, layer=None, k_tokens=15):
    torch.manual_seed(seed)
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(device)
    settings = dict(generation)
    settings.update(pad_token_id=tokenizer.pad_token_id, use_cache=True)
    start = time.monotonic()
    applied = 0
    with torch.inference_mode():
        if delta is None:
            output = model.generate(**inputs, **settings)
        else:
            with steering_hook(model, layer, delta, k_tokens) as hook:
                output = model.generate(**inputs, **settings)
                applied = hook.applied
    sequences = output[:, inputs['input_ids'].shape[1]:].cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    records = []
    for ids in sequences:
        stop = next((i for i, token in enumerate(ids) if token in eos), None)
        ids = ids if stop is None else ids[:stop+1]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        records.append({'generated_text': text, 'generated_token_ids': ids,
                        'generated_tokens': len(ids), 'finish_reason': 'eos' if stop is not None else 'length',
                        'hook_steps': min(applied, len(ids)),
                        'batch_elapsed_seconds': time.monotonic() - start})
    return records


class RunStore:
    def __init__(self, folder, manifest):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'records.jsonl'
        self.manifest_path = self.folder / 'manifest.json'
        self.records = {}
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text())
            if previous['identity'] != manifest['identity']:
                raise ValueError('Run manifest differs; choose a new output directory')
        else:
            write_json(self.manifest_path, manifest)
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f'Interrupted JSONL line {i+1}; preserve and repair before resume')
                if record['key'] in self.records:
                    raise ValueError(f"Duplicate record: {record['key']}")
                self.records[record['key']] = record

    def append(self, record):
        if record['key'] in self.records:
            raise ValueError('Duplicate record')
        with self.path.open('a') as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            output.flush()
        self.records[record['key']] = record
        write_json(self.folder / 'status.json', {'state': 'running', 'records': len(self.records),
                                                'last_key': record['key'], 'updated': utcnow()})


def run(config, model_name, output, dataset_path):
    dataset = load_contrast_dataset(dataset_path)
    model_config = json.loads((ROOT / 'configs/models.json').read_text())[model_name]
    train = dataset['splits']['train'][:config['train_pairs']]
    test = dataset['splits'][config['evaluation_split']][:config['eval_pairs']]
    if len(train) < config['train_pairs'] or len(test) < config['eval_pairs']:
        raise ValueError('Dataset smaller than requested configuration')
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ('experiment.py', 'dataset_builder.py', 'evaluation.py')}
    identity = fingerprint({'config': config, 'model': model_config, 'dataset': dataset['sha256'],
                            'source': source_hashes})
    manifest = {'schema_version': 4, 'identity': identity, 'created': utcnow(), 'config': config,
                'model_name': model_name, 'model': model_config, 'dataset_sha256': dataset['sha256'],
                'dataset_revision': dataset['revision'], 'source_sha256': source_hashes,
                'versions': {p: importlib.metadata.version(p) for p in
                             ('torch', 'transformers', 'bitsandbytes', 'accelerate', 'datasets')},
                'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                'train_ids': [p['id'] for p in train], 'eval_ids': [p['id'] for p in test]}
    store = RunStore(output, manifest)
    snapshot_dir = Path(output) / 'source'
    snapshot_dir.mkdir(exist_ok=True)
    for name, expected_hash in source_hashes.items():
        destination = snapshot_dir / name
        if not destination.exists():
            shutil.copy2(ROOT / name, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Source snapshot mismatch: {name}')
    model, tokenizer = load_model(model_config)
    layers = layer_indices(len(model.model.layers), config['layer_fractions'])
    vectors_path = Path(output) / 'vectors.pt'
    if vectors_path.exists():
        vectors = torch.load(vectors_path, map_location='cpu', weights_only=True)
    else:
        vectors, norms = extract_vectors(model, tokenizer, train, layers, config['batch_size'])
        torch.save(vectors, vectors_path)
        write_json(Path(output) / 'vectors.json', norms)
    generation = config['generation']
    batch_size = config['batch_size']
    expected_keys = set()

    def execute(condition, pairs, variant='entail', build_prompt=None, parse_prefix=None):
        name = condition['name']
        for start in range(0, len(pairs), batch_size):
            batch_all = pairs[start:start+batch_size]
            keys = [f"{name}/{p['id']}/{variant}" for p in batch_all]
            expected_keys.update(keys)
            # Regenerate a partial batch in full to preserve random-number consumption on resume.
            if all(key in store.records for key in keys):
                continue
            prompts = [(build_prompt(p) if build_prompt else prompt_for(tokenizer, p, variant)) for p in batch_all]
            seed = stable_seed(config['seed'], start, variant, condition.get('replicate', 0))
            delta = None
            if condition['kind'] == 'activation':
                layer = condition['layer']
                delta = intervention_delta(vectors[layer], condition['alpha'], condition['mode'],
                                           [p['id'] for p in batch_all],
                                           stable_seed(config['seed'], condition.get('replicate', 0)), layer)
            settings = dict(generation)
            if condition['kind'] == 'text':
                settings['max_new_tokens'] = config['answer_max_tokens']
            results = generate_batch(model, tokenizer, prompts, settings, seed,
                                     delta=delta, layer=condition.get('layer'), k_tokens=config['k_tokens'])
            for pair, key, prompt, result in zip(batch_all, keys, prompts, results):
                if key in store.records:
                    continue
                text = result['generated_text']
                if parse_prefix:
                    text = parse_prefix(pair) + text
                parsed = parse_output(text)
                record = {'key': key, 'example_id': pair['id'], 'variant': variant,
                          'context': {'premise': pair['premise'], 'hypothesis': pair[variant + '_hypothesis']},
                          'gold_label': 'ENTAILMENT' if variant == 'entail' else 'CONTRADICTION',
                          'condition': condition, 'seed': seed, 'prompt': prompt,
                          'parsed': parsed.asdict(), **result}
                store.append(record)
            print(json.dumps({'stage': name, 'done': min(start+batch_size, len(pairs)),
                              'total': len(pairs), 'records': len(store.records)}), flush=True)

    execute({'name': 'baseline', 'kind': 'baseline'}, test)
    if config.get('text_interventions') or config.get('both_baseline_variants'):
        execute({'name': 'baseline', 'kind': 'baseline'}, test, variant='contra')
    for layer in layers:
        for alpha in config['alphas']:
            for mode in config['modes']:
                replicates = config['control_replicates'] if mode != 'steer' else 1
                for replicate in range(replicates):
                    condition = {'name': f'{mode}_L{layer}_a{alpha}_r{replicate}', 'kind': 'activation',
                                 'layer': layer, 'layer_fraction': layer/(len(model.model.layers)-1),
                                 'alpha': alpha, 'mode': mode, 'replicate': replicate,
                                 'alpha_units': 'training_mean_difference_norm'}
                    execute(condition, test)
    skipped = []
    if config.get('text_interventions'):
        valid = [p for p in test if store.records[f"baseline/{p['id']}/entail"]['parsed']['status'] == 'ok']
        skipped = [p['id'] for p in test if p not in valid]
        cot_ids = {p['id']: tokenizer.encode(store.records[f"baseline/{p['id']}/entail"]['parsed']['cot'],
                                            add_special_tokens=False) for p in valid}
        for fraction in config['truncation_fractions']:
            prefixes = {p['id']: tokenizer.decode(cot_ids[p['id']][:math.floor(len(cot_ids[p['id']])*fraction)],
                                                  skip_special_tokens=False) + '\n</think>\n\n' for p in valid}
            execute({'name': f'truncate_{fraction}', 'kind': 'text', 'fraction': fraction}, valid,
                    build_prompt=lambda p: prompt_for(tokenizer, p) + prefixes[p['id']],
                    parse_prefix=lambda p: '<think>' + prefixes[p['id']])
        if config.get('swap_cot'):
            swap_valid = [p for p in valid if store.records[f"baseline/{p['id']}/contra"]['parsed']['status'] == 'ok']
            prefixes = {p['id']: store.records[f"baseline/{p['id']}/contra"]['parsed']['cot'] + '\n</think>\n\n'
                        for p in swap_valid}
            execute({'name': 'swap_contradiction_cot', 'kind': 'text'}, swap_valid,
                    build_prompt=lambda p: prompt_for(tokenizer, p) + prefixes[p['id']],
                    parse_prefix=lambda p: '<think>' + prefixes[p['id']])
            shuffled = {}
            for p in valid:
                ids = list(cot_ids[p['id']])
                random.Random(stable_seed(config['seed'], p['id'], 'shuffle')).shuffle(ids)
                shuffled[p['id']] = tokenizer.decode(ids, skip_special_tokens=False) + '\n</think>\n\n'
            execute({'name': 'shuffle_cot', 'kind': 'text'}, valid,
                    build_prompt=lambda p: prompt_for(tokenizer, p) + shuffled[p['id']],
                    parse_prefix=lambda p: '<think>' + shuffled[p['id']])
    if not expected_keys.issubset(store.records):
        raise RuntimeError('Incomplete run')
    write_json(Path(output) / 'status.json', {'state': 'complete', 'records': len(store.records),
                                             'expected_records': len(expected_keys),
                                             'truncation_excluded_incomplete_baselines': skipped,
                                             'updated': utcnow()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', default='artifacts/dataset.json')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.model, args.output, args.dataset)

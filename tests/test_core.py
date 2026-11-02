import json
import pytest
import torch
from dataset_builder import fingerprint, premise_key, select_pairs, validate_dataset
from evaluation import cohen_kappa, extract_label, parse_output, wilson_interval
from experiment import RunStore, SteeringHook, intervention_delta, layer_indices, steering_hook


def test_extracts_reasoning_instead_of_empty_lazy_match():
    parsed = parse_output('<think>A man runs, so he moves.</think>\nFINAL: ENTAILMENT')
    assert parsed.cot == 'A man runs, so he moves.'
    assert parsed.label == 'ENTAILMENT'
    assert parsed.status == 'ok'


def test_think_open_in_prompt_and_unicode_eos():
    parsed = parse_output('They cannot both be true.</think>\nFINAL: CONTRADICTION<｜end▁of▁sentence｜>')
    assert parsed.label == 'CONTRADICTION'
    assert parsed.answer == 'FINAL: CONTRADICTION'


def test_unfinished_thought_is_missing_not_null_effect():
    parsed = parse_output('FINAL: CONTRADICTION but I am still considering it')
    assert parsed.label is None
    assert parsed.answer == ''
    assert parsed.cot
    assert parsed.status == 'unfinished_thought'


@pytest.mark.parametrize('answer', ['ENTAILMENT', 'Maybe contradiction', 'FINAL: ENTAILMENT\nFINAL: CONTRADICTION', 'I would say FINAL: ENTAILMENT'])
def test_final_label_never_guessed(answer):
    assert extract_label(answer) is None


def test_empty_cot_keeps_valid_answer_but_flags_missing_reasoning():
    assert parse_output('</think>\nFINAL: ENTAILMENT').status == 'empty_thought'


def test_undefined_kappa_not_perfect_agreement():
    assert cohen_kappa(['x']*5, ['x']*5) is None
    assert cohen_kappa([], []) is None
    assert cohen_kappa(['a','b'], ['a','b']) == 1
    assert cohen_kappa(['a','b'], ['b','a']) == -1


def test_wilson_nonzero_uncertainty_at_boundaries():
    assert wilson_interval(0,50)['high'] > 0
    assert wilson_interval(50,50)['low'] < 1
    assert wilson_interval(0,0)['rate'] is None


def rows():
    return [{'premise':f'Premise {i}', 'hypothesis':f'H {label} {i}', 'label':label}
            for i in range(20) for label in (0,1,2)]


def test_pair_selection_reproducible_and_not_early_slice():
    a, b = select_pairs(rows(),5,42), select_pairs(list(reversed(rows())),5,42)
    assert a == b
    assert any(int(p['premise'].split()[-1]) >= 5 for p in a)


def test_dataset_disjointness_and_checksum():
    all_pairs = select_pairs(rows(),15,42)
    splits = {'train':all_pairs[:5], 'calibration':all_pairs[5:10], 'test':all_pairs[10:]}
    data = {'splits':splits, 'sha256':fingerprint(splits)}
    validate_dataset(data)
    splits['test'][0] = splits['train'][0]
    data['sha256'] = fingerprint(splits)
    with pytest.raises(ValueError, match='leakage'):
        validate_dataset(data)


def test_normalized_premise_exclusion():
    pair = select_pairs(rows(),1,42)[0]
    result = select_pairs(rows(),19,42,[premise_key(pair['premise'])])
    assert pair['id'] not in [p['id'] for p in result]


def test_insufficient_pairs_fails_instead_of_overlapping_splits():
    with pytest.raises(ValueError):
        select_pairs(rows(),21,42)


@pytest.mark.parametrize('mode', ['steer','orthogonal','gaussian'])
def test_intervention_exact_norm_and_reproducible(mode):
    vector = torch.arange(1,33).float()
    delta = intervention_delta(vector,2,mode,['x','y'],42,3)
    assert torch.allclose(delta.norm(dim=1), torch.full((2,),2*vector.norm()))
    assert torch.equal(delta,intervention_delta(vector,2,mode,['x','y'],42,3))
    if mode == 'orthogonal':
        assert torch.allclose(delta @ vector, torch.zeros(2), atol=.01)
    if mode != 'steer':
        assert not torch.equal(delta[0],delta[1])


def test_hook_changes_prefill_and_exactly_k_prediction_positions():
    hook = SteeringHook(torch.ones(2,4),2)
    prompt = torch.zeros(2,7,4)
    result = hook(None,None,prompt)
    assert torch.equal(result[:,-1],torch.ones(2,4))
    assert result[:,:-1].sum() == 0
    assert prompt.sum() == 0  # no mutation of the caller's tensor
    assert hook(None,None,(torch.zeros(2,1,4),'aux'))[0].sum() == 8
    assert hook(None,None,torch.zeros(2,1,4)).sum() == 0
    assert hook.applied == 2


def test_hook_removed_on_generation_failure():
    from types import SimpleNamespace
    layer = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    with pytest.raises(RuntimeError):
        with steering_hook(model,0,torch.zeros(1,3),2):
            raise RuntimeError('generation failed')
    assert not layer._forward_hooks


def test_real_transformer_generation_hook_counts_prefill():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                                        num_hidden_layers=2,num_attention_heads=2,
                                        num_key_value_heads=2,eos_token_id=None,pad_token_id=0))
    model.eval()
    inputs = torch.tensor([[1,2,3]])
    with torch.inference_mode():
        baseline = model.generate(inputs,max_new_tokens=4,do_sample=False)
        with steering_hook(model,0,torch.zeros(1,16),3) as hook:
            zero = model.generate(inputs,max_new_tokens=4,do_sample=False)
    assert torch.equal(baseline,zero)
    assert hook.calls == 4
    assert hook.applied == 3


def test_resume_rejects_changed_manifest_or_duplicate(tmp_path):
    store = RunStore(tmp_path,{'identity':'fixed'})
    store.append({'key':'one'})
    assert RunStore(tmp_path,{'identity':'fixed'}).records['one'] == {'key':'one'}
    with pytest.raises(ValueError,match='differs'):
        RunStore(tmp_path,{'identity':'changed'})
    with pytest.raises(ValueError,match='Duplicate'):
        store.append({'key':'one'})


def test_layer_fraction_endpoints():
    assert layer_indices(28,[0,.5,1]) == [0,14,27]

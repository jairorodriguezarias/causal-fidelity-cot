import pytest
from llm_judge import Judge, consistency_category, validate_assessment


def result(**changes):
    return dict({'verdict':'ENTAILMENT','evidence_quote':'It follows.','first_contradiction_quote':'','explanation':'The conclusion is entailed.'},**changes)


def test_judge_evidence_must_be_present():
    validate_assessment(result(),'It follows.')
    with pytest.raises(ValueError,match='exact quote'):
        validate_assessment(result(),'Unrelated text')


def test_empty_reasoning_never_calls_service():
    assert Judge('http://invalid','test').assess('','context')['status'] == 'missing_reasoning'


def test_judge_rejects_schema_errors():
    with pytest.raises(ValueError):
        validate_assessment({'verdict':'ENTAILMENT'},'cot')
    with pytest.raises(ValueError):
        validate_assessment(result(verdict='FAITHFUL_SHIFT'),'It follows.')


def test_no_shift_requires_measured_baseline_and_valid_reasoning():
    a = {'status':'ok','verdict':'ENTAILMENT'}
    assert consistency_category('ENTAILMENT','ENTAILMENT',a) == 'NO_OBSERVED_SHIFT'
    assert consistency_category(None,'ENTAILMENT',a) == 'BASELINE_NOT_ENTAILMENT'
    assert consistency_category('ENTAILMENT',None,a) == 'UNASSESSABLE'
    assert consistency_category('ENTAILMENT','CONTRADICTION',a) == 'ANSWER_REASONING_MISMATCH'

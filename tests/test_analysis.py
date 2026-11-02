from analyze import paired_difference, summarize_condition


def record(key,label):
    return {'example_id':key,'parsed':{'label':label,'status':'ok' if label else 'unfinished_thought'},
            'gold_label':'ENTAILMENT','generated_tokens':10}


def test_missing_outcomes_have_explicit_identification_bounds():
    baseline={key:record(key,'ENTAILMENT') for key in 'abc'}
    result=summarize_condition([record('a','CONTRADICTION'),record('b','ENTAILMENT'),record('c',None)],baseline)
    assert result['flip_complete_cases']['rate']==.5
    assert result['flip_missingness_bounds']=={'low':1/3,'high':2/3}
    assert result['missing_among_baseline_entailment']==1


def test_paired_bootstrap_uses_examples_and_preserves_null():
    result=paired_difference([1,0,1,0],[1,0,1,0])
    assert result=={'n':4,'difference':0.0,'low':0.0,'high':0.0}
    assert paired_difference([],[])['difference'] is None

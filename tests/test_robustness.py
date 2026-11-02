import pytest
from robustness import holm_adjust,mcnemar_exact,paired_binary


def test_exact_paired_test_uses_discordances_not_marginal_sample_counts():
    # Three treatment-only outcomes and no control-only outcomes give p = 2/8.
    assert mcnemar_exact([1,1,1,0,1],[0,0,0,0,1])==.25
    assert mcnemar_exact([1,0],[1,0])==1
    assert mcnemar_exact([],[]) is None


def test_unknown_binary_pairs_get_sharp_marginal_bounds():
    result=paired_binary([1,None,None,0],[None,0,None,1])
    # Pair ranges: [0,1], [0,1], [-1,1], [-1,-1].
    assert result['missingness_bounds']=={'low':-.5,'high':.5}
    assert result['complete_pairs']==1
    assert result['complete_case_difference']['difference']==-1


def test_missing_pairs_are_not_counted_as_null_effect():
    result=paired_binary([None,None],[None,None])
    assert result['complete_case_difference']['difference'] is None
    assert result['exact_p_complete_cases'] is None
    assert result['missingness_bounds']=={'low':-1,'high':1}


def test_holm_adjustment_monotonic_and_preserves_planned_family_size():
    adjusted=holm_adjust([.01,.04,.03,None])
    assert adjusted==pytest.approx([.04,.09,.09,None])


def test_pairing_and_valid_labels_required():
    with pytest.raises(ValueError): paired_binary([1],[1,0])
    with pytest.raises(ValueError): paired_binary([2],[0])

import torch

from experiments.core import acceleration, coefficients, sample
from experiments.cycle3 import query_design, select_family


def test_hidden_family_selector_recovers_exact_broad_functions():
    x=query_design(64,7,"broad")
    for family in ("duffing","pendulum"):
        c=coefficients(7,family)
        y=acceleration(x,c,family)
        selected,fitted,scores=select_family(x,y,7)
        assert selected==family
        torch.testing.assert_close(fitted,c,atol=1e-10,rtol=1e-10)
        assert scores[family]<1e-20


def test_mixed_design_has_exactly_eight_narrow_points_and_nested_broad_suffix():
    small=query_design(16,3,"mixed")
    large=query_design(64,3,"mixed")
    torch.testing.assert_close(small,large[:16],atol=0,rtol=0)
    torch.testing.assert_close(small[:8],query_design(8,3,"narrow"),atol=0,rtol=0)
    assert small[:8].abs().max()<=.25
    assert (large[8:].abs()>.25).any()

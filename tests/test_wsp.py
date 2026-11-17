import copy
import numpy as np
import pytest
from experiments.wsp_common import apply_patch,compile_edit,forward,gauge


def fixture():
    return dict(W1=[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,0.],[0.,0.,0.]],
                b1=[0.]*8,W2=[-1.,1.,-.4,.4,1.,-1.,0.,0.],b2=[0.])


def test_exact_compilers_and_function_preserving_gauge():
    w=fixture();x=np.random.default_rng(1).uniform(-1,1,(100,3))
    represented=gauge(w,9,True)
    np.testing.assert_allclose(forward(w,x),forward(represented,x),atol=1e-7)
    action=x.copy();action[:,2]*=1.5
    np.testing.assert_allclose(forward(compile_edit(w,"action_gain"),x),forward(w,action),atol=1e-12)
    for task,coefficient in (("add_drag",.2),("semantic_damping",.4)):
        edited=compile_edit(represented,task,coefficient)
        np.testing.assert_allclose(forward(edited,x),forward(represented,x)-coefficient*x[:,1],atol=1e-12)


def test_invalid_patches_are_rejected_without_mutating_source():
    w=fixture();before=copy.deepcopy(w)
    bad=[{"tensor":"W1","index":[-1,0],"value":1.}]
    with pytest.raises(ValueError):apply_patch(w,bad)
    with pytest.raises(ValueError):apply_patch(w,[{"tensor":"b2","index":[0],"value":float("nan")}])
    p={"tensor":"b2","index":[0],"value":1.}
    with pytest.raises(ValueError):apply_patch(w,[p,p])
    assert w==before

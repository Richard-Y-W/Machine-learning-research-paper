import torch

from experiments.core import DynamicsNet, parameter_vector, reparameterize, sample
from experiments.deployment import fit_student, initialize, relative_step


def test_relative_step_equivariance_and_zero_coordinates():
    torch.manual_seed(33)
    j=torch.randn(9,24,dtype=torch.float64)
    theta=torch.randn(24,dtype=torch.float64)
    theta[0]=0
    scale=torch.exp(torch.randn(24,dtype=torch.float64)*2)
    residual=torch.randn(9,dtype=torch.float64)
    d=relative_step(j,residual,theta)
    transformed=relative_step(j/scale,residual,theta*scale)
    assert d[0]==0
    torch.testing.assert_close(transformed/scale,d,atol=1e-10,rtol=1e-10)


def test_cold_initialization_does_not_depend_on_base_weights_or_representation():
    torch.manual_seed(1)
    base=DynamicsNet(12).double()
    cold=initialize(base,3,"cold_adam","original")
    changed=reparameterize(base,99,"rescaled")
    with torch.no_grad():
        changed.net[4].bias.add_(100)
    other=initialize(changed,3,"cold_adam","rescaled")
    torch.testing.assert_close(parameter_vector(cold),parameter_vector(other),atol=0,rtol=0)
    assert not torch.equal(parameter_vector(cold),parameter_vector(base))


def test_relative_editor_is_functionally_rescaling_equivariant():
    torch.manual_seed(7)
    base=DynamicsNet(12).double()
    x=sample(24,18)
    test=sample(40,19)
    with torch.no_grad():
        target=base(x)-.1*x[:,1]
    first,_=fit_student(base,x,target,5,"relative_jacobian","original","short")
    second,_=fit_student(base,x,target,5,"relative_jacobian","rescaled","short")
    torch.testing.assert_close(first(test),second(test),atol=1e-9,rtol=1e-8)


def test_lbfgs_matches_small_target_without_mutating_base():
    torch.manual_seed(8)
    base=DynamicsNet(12).double()
    x=sample(32,10)
    before=parameter_vector(base).clone()
    with torch.no_grad():
        target=base(x)-.05*x[:,1]
        old=(base(x)-target).square().mean()
    student,counts=fit_student(base,x,target,5,"warm_lbfgs","original","short")
    assert (student(x)-target).square().mean()<old
    assert counts["gradient_points"]>=len(x)
    torch.testing.assert_close(parameter_vector(base),before,atol=0,rtol=0)

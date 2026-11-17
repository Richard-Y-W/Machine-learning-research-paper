import copy

import torch

from experiments.core import DynamicsNet,parameter_vector,sample
from experiments.cycle4 import (changed_acceleration,feature_residual,loo_solve,
                               fit_neural,polynomial,rbf_residual,source_acceleration)


def test_analytic_kernel_loo_matches_explicit_refits():
    x=sample(12,17)
    phi=polynomial(x,1)
    kernel=phi@phi.T
    y=torch.sin(x[:,0])+x[:,1]
    score,ridge,_=loo_solve(kernel,y)
    errors=[]
    for i in range(len(x)):
        keep=torch.arange(len(x))!=i
        # Exact leave-one-out identity keeps the full-design regularizer n*lambda.
        alpha=torch.linalg.solve(kernel[keep][:,keep]+len(x)*ridge*torch.eye(len(x)-1,dtype=x.dtype),y[keep])
        errors.append((kernel[i,keep]@alpha-y[i]).square())
    torch.testing.assert_close(torch.stack(errors).mean(),torch.tensor(score,dtype=x.dtype),atol=1e-10,rtol=1e-8)


def test_head_correction_can_be_folded_exactly_into_network():
    torch.manual_seed(13)
    base=DynamicsNet(32).double()
    original=parameter_vector(base).clone()
    x=sample(32,2)
    residual=.05*x[:,0]
    def phi(z):
        return torch.cat((base.hidden(z),torch.ones(len(z),1,dtype=z.dtype)),1)
    correction,_,beta=feature_residual(phi,x,residual)
    folded=copy.deepcopy(base)
    with torch.no_grad():
        folded.net[4].weight.add_(beta[:-1][None,:]);folded.net[4].bias.add_(beta[-1:])
    test=sample(40,3)
    torch.testing.assert_close(folded(test),base(test)+correction(test),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(parameter_vector(base),original,atol=0,rtol=0)


def test_prediction_only_adapter_accepts_opaque_callable():
    x=sample(32,9)
    def opaque(z):
        return -z[:,0]+z[:,2]
    y=opaque(x)+.1*torch.tanh(x[:,1])
    predictor,meta=rbf_residual(opaque,x,y)
    assert torch.isfinite(predictor(x)).all()
    assert meta["extra_scalars"]>0
    assert ((predictor(x)-y).square().mean() < (opaque(x)-y).square().mean())


def test_new_mechanism_shift_is_unavailable_in_source_labels():
    x=sample(40,11)
    c=torch.tensor([1.,.3,.2,1.],dtype=x.dtype)
    expected=-.2*torch.sin(3*x[:,0])-.3*torch.tanh(4*x[:,1])
    for family in ("duffing","pendulum","contact","vanderpol"):
        delta=changed_acceleration(x,c,family,"nonlinear")-source_acceleration(x,c,family)
        torch.testing.assert_close(delta,expected,atol=1e-12,rtol=1e-12)


def test_neural_residual_does_not_need_source_model_parameters():
    x=sample(16,31)
    def opaque(z):
        return -z[:,0]+.7*z[:,2]
    y=opaque(x)+.05*x[:,1]
    predictor,meta=fit_neural(None,opaque,x,y,4,"neural_residual")
    assert torch.isfinite(predictor(x)).all()
    assert meta["gradient_points"]>0
    assert meta["extra_scalars"]==1217

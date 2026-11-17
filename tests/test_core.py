import torch

from experiments.core import (DynamicsNet, acceleration, coefficients, features,
    fit_coefficients, intervention, jacobian, parameter_vector, reparameterize,
    ridge_step, rollout, sample, set_vector, weight_edit)

torch.set_num_threads(2)


def test_physical_coefficients_recovered_from_exact_queries():
    x = sample(32, 9)
    for family in ("duffing", "pendulum"):
        c = coefficients(0, family)
        recovered = fit_coefficients(x, acceleration(x, c, family), family)
        torch.testing.assert_close(recovered, c, atol=1e-12, rtol=1e-12)


def test_network_symmetries_preserve_function():
    torch.manual_seed(5)
    model = DynamicsNet(12).double()
    x = sample(100, 1)
    for kind in ("original", "permuted", "rescaled"):
        torch.testing.assert_close(model(x), reparameterize(model, 5, kind)(x),
                                   atol=1e-12, rtol=1e-12)


def test_jacobian_matches_finite_difference():
    torch.manual_seed(11)
    model = DynamicsNet(8).double()
    x = sample(5, 1)
    j = jacobian(model, x)
    before = parameter_vector(model)
    direction = torch.randn_like(before)
    eps = 1e-7
    set_vector(model, before + eps * direction)
    plus = model(x).detach()
    set_vector(model, before - eps * direction)
    minus = model(x).detach()
    set_vector(model, before)
    torch.testing.assert_close(j @ direction, (plus-minus)/(2*eps), atol=1e-7, rtol=1e-6)


def test_normalized_update_is_diagonal_coordinate_equivariant():
    torch.manual_seed(9)
    j = torch.randn(8, 20, dtype=torch.float64)
    r = torch.randn(8, dtype=torch.float64)
    scale = torch.exp(torch.randn(20, dtype=torch.float64)*2)
    d = ridge_step(j, r, True, 1e-4)
    transformed = ridge_step(j / scale, r, True, 1e-4)
    torch.testing.assert_close(transformed/scale, d, atol=1e-10, rtol=1e-10)


def test_oracle_residual_preserves_base_error():
    torch.manual_seed(0)
    model = DynamicsNet(8).double()
    x = sample(30, 6)
    family = "duffing_mismatch"
    c, m = coefficients(0, family), intervention("combined")
    before_error = model(x) - acceleration(x, c, family)
    after_error = model(x) + features(x, family)@(c*m) - acceleration(x, c*(1+m), family)
    torch.testing.assert_close(before_error, after_error)


def test_zero_effect_set_is_really_zero():
    x = sample(30, 7)
    c = coefficients(0, "duffing")
    for name in ("damping", "restoring", "combined"):
        z = x.clone()
        if name in ("restoring", "combined"):
            z[:,0] = 0
        if name in ("damping", "combined"):
            z[:,1] = 0
        assert torch.count_nonzero(features(z, "duffing")@(c*intervention(name))) == 0


def test_edit_does_not_mutate_original_and_improves_fit():
    torch.manual_seed(1)
    model = DynamicsNet(12).double()
    x = sample(24, 5)
    before = parameter_vector(model).clone()
    target = model(x).detach() - .1*x[:,1]
    for method in ("head_ridge", "jacobian_ridge", "normalized_jacobian", "finetune_adam"):
        edited = weight_edit(model, x, target, method)
        assert (edited.predict(x)-target).square().mean() < (model(x)-target).square().mean()
        torch.testing.assert_close(parameter_vector(model), before)


def test_rollout_integrator_constant_acceleration():
    initial = torch.zeros((2, 2), dtype=torch.float64)
    actions = torch.ones((10,2), dtype=torch.float64)
    path = rollout(lambda x: x[:,2], initial, actions, dt=.1)
    torch.testing.assert_close(path[-1,:,1], torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(path[-1,:,0], torch.full((2,), .55, dtype=torch.float64))

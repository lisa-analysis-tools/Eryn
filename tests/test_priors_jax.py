"""Tests of the jax backend of eryn's priors and prior containers.

The jax results must equal the NumPy ones to 1e-12, run under ``jax.jit``, ``jax.vmap`` and
``jax.grad``, draw reproducibly from explicit keys, and survive deepcopy and pickle.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from eryn.priors import (  # noqa: E402
    CosineUniform,
    DeltaFunction,
    DiscreteUniform,
    Exponential,
    Gaussian,
    InverseUniform,
    Log10Uniform,
    LogNormal,
    LogUniform,
    MultivariateGaussian,
    PoissonDistribution,
    PowerLaw,
    ProbDistContainer,
    SineUniform,
    UniformDistribution,
    UniformInVolume,
)
from tests.test_priors import (  # noqa: E402
    CONTAINER_BUILDERS,
    PRIOR_BUILDERS,
    ShiftedGaussian,
    SumAndDifference,
    assert_round_trips,
)

# * eryn never switches x64 itself; the tests run in float64 like the NumPy path.
jax.config.update("jax_enable_x64", True)

RELATIVE_TOLERANCE = 1e-12
KEY = jax.random.key(2026)

DENSITY_PRIORS = {
    "uniform": UniformDistribution(2.0, 5.0),
    "delta": DeltaFunction(2.0),
    "gaussian": Gaussian(1.0, 2.0),
    "lognormal": LogNormal(0.5, 0.3),
    "exponential": Exponential(2.0),
    "discrete_uniform": DiscreteUniform(1, 5),
    "poisson": PoissonDistribution(3.0),
}

TRANSFORM_PRIORS = {
    "powerlaw": PowerLaw(alpha=2.0, minimum=1.0, maximum=2.0),
    "powerlaw_negative_domain": PowerLaw(alpha=2.0, minimum=-2.0, maximum=3.0),
    "inverse_uniform": InverseUniform(1.0, 10.0),
    "uniform_in_volume": UniformInVolume(0.5, 3.0),
    "log_uniform": LogUniform(1e-3, 1e3),
    "log10_uniform": Log10Uniform(1e-24, 1e-20),
    "cosine_uniform": CosineUniform(),
    "sine_uniform": SineUniform(),
}

# * Physical-space CDFs of the transformed priors, for the draws' KS checks.
PHYSICAL_CDFS = {
    "powerlaw": lambda value: (value**3 - 1.0) / 7.0,
    "powerlaw_negative_domain": lambda value: (value**3 + 8.0) / 35.0,
    "inverse_uniform": lambda value: (1.0 - 1.0 / value) / 0.9,
    "uniform_in_volume": lambda value: (value**3 - 0.125) / (27.0 - 0.125),
    "log_uniform": lambda value: (np.log(value) - np.log(1e-3)) / (np.log(1e3) - np.log(1e-3)),
    "log10_uniform": lambda value: (np.log10(value) + 24.0) / 4.0,
    "cosine_uniform": lambda value: (1.0 - np.cos(value)) / 2.0,
    "sine_uniform": lambda value: (np.sin(value) + 1.0) / 2.0,
}


def assert_matches(jax_values, numpy_values):
    """Assert agreement to the relative tolerance, with -inf and NaN in the same places."""
    np.testing.assert_allclose(
        np.asarray(jax_values), np.asarray(numpy_values), rtol=RELATIVE_TOLERANCE, atol=0.0
    )


def mixed_container(**options):
    """Every kind of entry: transforms, a tuple key, a conditional prior, plain densities."""
    return ProbDistContainer(
        {
            "a": UniformDistribution(0.0, 10.0),
            "b": ShiftedGaussian(),
            "amplitude": LogUniform(0.1, 10.0),
            "latitude": SineUniform(),
            ("c", "d"): MultivariateGaussian(
                ["c", "d"], mu=[0.0, 1.0], cov=[[1.0, 0.2], [0.2, 0.5]]
            ),
            "e": Gaussian(3.0, 0.5),
        },
        **options,
    )


def test_import_does_not_import_jax() -> None:
    """JAX stays an optional dependency: importing eryn must not import it."""
    code = "import sys, eryn.priors, eryn.ensemble; print('jax' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_backend_names() -> None:
    """force_backend takes the LAT names; use_cupy stays the CuPy switch."""
    assert UniformDistribution(0.0, 1.0).backend_name == "numpy"
    assert UniformDistribution(0.0, 1.0, force_backend="cpu").backend_name == "numpy"
    assert UniformDistribution(0.0, 1.0, force_backend="jax").backend_name == "jax"
    assert UniformDistribution(0.0, 1.0, force_backend="jax").xp is jnp
    with pytest.raises(ValueError, match="Unknown backend"):
        UniformDistribution(0.0, 1.0, force_backend="tpu")
    with pytest.raises(ValueError, match="either use_cupy or force_backend"):
        UniformDistribution(0.0, 1.0, use_cupy=True, force_backend="jax")


def test_float32_warning() -> None:
    """A jax prior or container built while x64 is off warns, and eryn leaves x64 alone."""
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.warns(UserWarning, match="jax_enable_x64"):
            UniformDistribution(0.0, 1.0, force_backend="jax")
        with pytest.warns(UserWarning, match="jax_enable_x64"):
            ProbDistContainer({"a": UniformDistribution(0.0, 1.0)}).to_backend("jax")
        assert not jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name", sorted(DENSITY_PRIORS))
def test_density_parity(name) -> None:
    """logpdf, pdf and cdf on the jax backend equal NumPy's, also under jit."""
    prior = DENSITY_PRIORS[name]
    prior_jax = prior.to_backend("jax")
    points = np.linspace(-1.0, 7.0, 33)
    assert_matches(jax.jit(prior_jax.logpdf)(points), prior.logpdf(points))
    assert_matches(prior_jax.pdf(points), prior.pdf(points))
    if name in ("uniform", "delta", "exponential", "discrete_uniform"):
        assert_matches(jax.jit(prior_jax.cdf)(points), prior.cdf(points))


@pytest.mark.parametrize("name", sorted(TRANSFORM_PRIORS))
def test_transform_parity(name) -> None:
    """The transforms and the *_physical methods on jax equal NumPy's, also under jit."""
    prior = TRANSFORM_PRIORS[name]
    prior_jax = prior.to_backend("jax")
    np.random.seed(3)
    sampling = prior.rvs(64)
    physical = prior.inverse_transform(sampling)
    assert_matches(jax.jit(prior_jax.inverse_transform)(sampling), physical)
    assert_matches(
        jax.jit(prior_jax.forward_transform)(physical), prior.forward_transform(physical)
    )
    assert_matches(jax.jit(prior_jax.log_jacobian)(physical), prior.log_jacobian(physical))
    assert_matches(jax.jit(prior_jax.logpdf)(sampling), prior.logpdf(sampling))
    assert_matches(prior_jax.logpdf_physical(physical), prior.logpdf_physical(physical))
    assert_matches(prior_jax.pdf_physical(physical), prior.pdf_physical(physical))


@pytest.mark.parametrize("name", sorted(TRANSFORM_PRIORS))
def test_transform_jacobian_from_autodiff(name) -> None:
    """jacfwd of inverse_transform is exp(-log_jacobian), signed as the map: cos and 1/x
    decrease, the others increase."""
    prior_jax = TRANSFORM_PRIORS[name].to_backend("jax")
    sampling = prior_jax.rvs(16, rng_key=KEY)
    derivative = jax.vmap(jax.jacfwd(prior_jax.inverse_transform))(sampling)
    physical = prior_jax.inverse_transform(sampling)
    magnitude = jnp.exp(-prior_jax.log_jacobian(physical))
    sign = -1.0 if name in ("cosine_uniform", "inverse_uniform") else 1.0
    assert_matches(derivative, sign * magnitude)


def test_logpdf_gradients() -> None:
    """jax.grad of logpdf equals the analytic score of a Gaussian and a multivariate one."""
    gaussian = Gaussian(1.0, 2.0, force_backend="jax")
    points = jnp.linspace(-3.0, 4.0, 9)
    assert_matches(jax.vmap(jax.grad(gaussian.logpdf))(points), -(points - 1.0) / 4.0)

    mean = np.array([0.5, -1.0, 2.0])
    covariance = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
    mvn = MultivariateGaussian(["x", "y", "z"], mu=mean, cov=covariance, force_backend="jax")
    point = jnp.array([0.2, 0.1, 1.5])
    gradient = jax.grad(lambda value: mvn.logpdf(value[None])[0])(point)
    assert_matches(gradient, -np.linalg.solve(covariance, np.asarray(point) - mean))
    assert isinstance(mvn.mu, jax.Array)


def test_container_joint_transforms_on_jax() -> None:
    """A joint prior's *_nd transforms run on the jax backend too, also under jit."""
    container = ProbDistContainer({("x", "y"): SumAndDifference(), "z": LogUniform(0.1, 10.0)})
    container_jax = container.to_backend("jax")
    physical = np.array([[1.0, 0.5, 2.0], [-0.3, 0.7, 0.5]])
    sampling = container.forward_transform(physical)
    assert_matches(jax.jit(container_jax.forward_transform)(physical), sampling)
    assert_matches(jax.jit(container_jax.inverse_transform)(sampling), physical)
    assert_matches(jax.jit(container_jax.log_jacobian)(physical), container.log_jacobian(physical))


def test_container_parity_and_transforms() -> None:
    """The container's logpdf, transforms, logpdf_physical and keys subsets match NumPy."""
    container = mixed_container()
    container_jax = container.to_backend("jax")
    np.random.seed(4)
    sampling = container.rvs(200)
    physical = container.inverse_transform(sampling)

    assert_matches(jax.jit(container_jax.logpdf)(sampling), container.logpdf(sampling))
    # * keys holds positions: eryn compares them with each entry's indices.
    subset = [container.name_to_idx["a"], container.name_to_idx["e"]]
    assert_matches(
        container_jax.logpdf(sampling, keys=subset), container.logpdf(sampling, keys=subset)
    )
    assert_matches(jax.jit(container_jax.inverse_transform)(sampling), physical)
    assert_matches(jax.jit(container_jax.forward_transform)(physical), sampling)
    assert_matches(jax.jit(container_jax.log_jacobian)(physical), container.log_jacobian(physical))
    assert_matches(container_jax.logpdf_physical(physical), container.logpdf_physical(physical))

    # * Only the amplitude and latitude columns change between the two spaces.
    unchanged = [container.name_to_idx[name] for name in ("a", "b", "c", "d", "e")]
    np.testing.assert_array_equal(physical[:, unchanged], sampling[:, unchanged])
    np.testing.assert_allclose(
        physical[:, container.name_to_idx["amplitude"]], np.exp(sampling[:, 2])
    )


def test_container_under_vmap_grad_and_single_points() -> None:
    """A single point gives a 0-d array under jit; vmap and grad pass through the container."""
    container_jax = mixed_container(force_backend="jax")
    point = container_jax.rvs(1, rng_key=KEY)[0]
    single = jax.jit(container_jax.logpdf)(point)
    assert single.shape == ()
    assert_matches(single, container_jax.logpdf(point[None])[0])

    batch = container_jax.rvs((3, 5), rng_key=KEY)
    assert_matches(
        jax.vmap(container_jax.logpdf)(batch),
        container_jax.logpdf(batch.reshape(15, -1)).reshape(3, 5),
    )
    gradient = jax.grad(container_jax.logpdf)(point)
    assert gradient.shape == point.shape and bool(jnp.all(jnp.isfinite(gradient)))

    jacobian = jax.jacfwd(container_jax.inverse_transform)(point)
    physical = container_jax.inverse_transform(point)
    assert_matches(
        jnp.log(jnp.abs(jnp.linalg.det(jacobian))), -container_jax.log_jacobian(physical)
    )


def test_draws_follow_their_keys() -> None:
    """The same key gives the same draws, other keys others, and equal priors in a container
    draw different columns."""
    prior_jax = UniformDistribution(0.0, 1.0, force_backend="jax")
    np.testing.assert_array_equal(prior_jax.rvs(8, rng_key=KEY), prior_jax.rvs(8, rng_key=KEY))
    assert not np.array_equal(
        prior_jax.rvs(8, rng_key=KEY), prior_jax.rvs(8, rng_key=jax.random.key(7))
    )

    twins = ProbDistContainer(
        {"first": UniformDistribution(0.0, 1.0), "second": UniformDistribution(0.0, 1.0)},
        force_backend="jax",
    )
    draws = twins.rvs(100, rng_key=KEY)
    assert not np.allclose(draws[:, 0], draws[:, 1])
    np.testing.assert_array_equal(draws, twins.rvs(100, rng_key=KEY))


def test_conditional_draws_on_jax() -> None:
    """A conditional prior draws after the variable it depends on, from its own key."""
    container_jax = mixed_container(force_backend="jax")
    draws = container_jax.rvs(20_000, rng_key=KEY)
    residual = np.asarray(
        draws[:, container_jax.name_to_idx["b"]] - draws[:, container_jax.name_to_idx["a"]]
    )
    assert stats.kstest(residual, "norm").pvalue > 1e-3


def test_rng_key_rules() -> None:
    """NumPy and CuPy reject a key; jax needs one, except for a fixed value."""
    with pytest.raises(TypeError, match="rng_key applies to the jax backend only"):
        UniformDistribution(0.0, 1.0).rvs(3, rng_key=KEY)
    with pytest.raises(TypeError, match="rng_key applies to the jax backend only"):
        mixed_container().rvs(3, rng_key=KEY)
    with pytest.raises(ValueError, match="explicit rng_key"):
        UniformDistribution(0.0, 1.0, force_backend="jax").rvs(3)
    with pytest.raises(ValueError, match="explicit rng_key"):
        mixed_container(force_backend="jax").rvs(3)
    np.testing.assert_array_equal(DeltaFunction(2.0, force_backend="jax").rvs(3), [2.0, 2.0, 2.0])
    # * rng_key is reserved, so it never becomes a conditional variable of a container.
    assert "rng_key" not in UniformDistribution(0.0, 1.0).required_variables
    assert mixed_container().logpdf(np.full((2, 7), 0.5)).shape == (2,)


@pytest.mark.parametrize(
    "name, cdf",
    [
        ("uniform", lambda value: stats.uniform(2.0, 3.0).cdf(value)),
        ("gaussian", lambda value: stats.norm(1.0, 2.0).cdf(value)),
        ("lognormal", lambda value: stats.lognorm(0.3, scale=np.exp(0.5)).cdf(value)),
        ("exponential", lambda value: stats.expon(scale=2.0).cdf(value)),
    ],
)
def test_continuous_draws_follow_their_law(name, cdf) -> None:
    """KS test of 20 000 jax draws against SciPy, with a fixed key."""
    draws = DENSITY_PRIORS[name].to_backend("jax").rvs(20_000, rng_key=KEY)
    assert stats.kstest(np.asarray(draws), cdf).pvalue > 1e-3


@pytest.mark.parametrize("name", sorted(TRANSFORM_PRIORS))
def test_physical_draws_follow_their_law(name) -> None:
    """rvs_physical on jax follows the analytic physical-space CDF."""
    draws = TRANSFORM_PRIORS[name].to_backend("jax").rvs_physical(20_000, rng_key=KEY)
    assert stats.kstest(np.asarray(draws), PHYSICAL_CDFS[name]).pvalue > 1e-3


def test_discrete_and_multivariate_draws() -> None:
    """Chi-square checks of the discrete draws, and Mahalanobis distances of the Gaussian."""
    discrete = DiscreteUniform(1, 5, force_backend="jax").rvs(20_000, rng_key=KEY)
    counts = np.bincount(np.asarray(discrete).astype(int), minlength=6)[1:]
    assert stats.chisquare(counts).pvalue > 1e-3

    poisson = np.asarray(
        PoissonDistribution(3.0, force_backend="jax").rvs(20_000, rng_key=KEY)
    ).astype(int)
    observed = np.bincount(np.minimum(poisson, 9), minlength=10)
    expected = 20_000 * np.append(stats.poisson(3.0).pmf(np.arange(9)), stats.poisson(3.0).sf(8))
    assert stats.chisquare(observed, expected).pvalue > 1e-3

    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    mvn = MultivariateGaussian(["x", "y"], mu=[0.5, -1.0], cov=covariance, force_backend="jax")
    offsets = np.asarray(mvn.rvs(20_000, rng_key=KEY)) - [0.5, -1.0]
    distances = np.einsum("ni,ij,nj->n", offsets, np.linalg.inv(covariance), offsets)
    assert stats.kstest(distances, stats.chi2(2).cdf).pvalue > 1e-3


@pytest.mark.parametrize("name", sorted(PRIOR_BUILDERS))
def test_jax_priors_survive_deepcopy_and_pickle(name) -> None:
    """jax priors hold the library name, not the module, so deepcopy and pickle work."""
    assert_round_trips(PRIOR_BUILDERS[name]().to_backend("jax"), rng_key=KEY)


@pytest.mark.parametrize("name", sorted(CONTAINER_BUILDERS))
def test_jax_containers_survive_deepcopy_and_pickle(name) -> None:
    """Mixed, joint and nested jax containers survive deepcopy and pickle."""
    assert_round_trips(CONTAINER_BUILDERS[name]().to_backend("jax"), rng_key=KEY)


def test_to_backend_round_trip_and_in_place_container() -> None:
    """to_backend copies; a container built with force_backend switches its priors in place."""
    prior = LogUniform(0.1, 10.0)
    converted = prior.to_backend("jax")
    # * The copy's stored transforms must run on the copy, so on jax, not on the original.
    assert converted.forward_transform.__self__ is converted
    round_trip = converted.to_backend("cpu")
    assert prior.backend_name == "numpy" and round_trip.backend_name == "numpy"
    points = np.linspace(-3.0, 3.0, 13)
    np.testing.assert_array_equal(round_trip.logpdf(points), prior.logpdf(points))

    container = CONTAINER_BUILDERS["nested"]()
    container_jax = container.to_backend("jax")
    for _, child in container_jax.priors:
        assert child.backend_name == "jax"
    assert all(child.backend_name == "numpy" for _, child in container.priors)

    shared = UniformDistribution(0.0, 1.0)
    ProbDistContainer({"a": shared}, force_backend="jax")
    assert shared.backend_name == "jax"

    kept = UniformDistribution(0.0, 1.0)
    ProbDistContainer({"a": kept}).to_backend("jax")
    assert kept.backend_name == "numpy"


def test_exponential_with_negative_inputs_on_jax() -> None:
    """Negative inputs give -inf density and zero CDF on the jax backend, also under jit."""
    prior_jax = Exponential(2.0, force_backend="jax")
    points = np.array([-1.0, 1.0])
    np.testing.assert_allclose(jax.jit(prior_jax.logpdf)(points), [-np.inf, -0.5 - np.log(2.0)])
    np.testing.assert_allclose(jax.jit(prior_jax.cdf)(points), [0.0, 1.0 - np.exp(-0.5)])

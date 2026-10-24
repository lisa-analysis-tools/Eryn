from __future__ import annotations

import subprocess
import sys

import pytest
import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson, multivariate_normal

from eryn.ensemble import EnsembleSampler
from eryn.priors import (
    DeltaFunction,
    Prior,
    ProbDistContainer,
    UniformDistribution,
    SineUniform,
    CosineUniform,
    Log10Uniform,
    DiscreteUniform,
    PoissonDistribution,
    PowerLaw,
    MultivariateGaussian,
    
)
from eryn.prior import uniform_dist
from eryn.utils import PeriodicContainer, TransformContainer


def test_deprecated_uniform_dist() -> None:
    """Verify that uniform_dist throws a DeprecationWarning and routes correctly."""
    with pytest.warns(DeprecationWarning, match="eryn.prior.uniform_dist is deprecated"):
        dist = uniform_dist(0.0, 1.0)
    assert isinstance(dist, UniformDistribution)
    assert dist.minimum == 0.0
    assert dist.maximum == 1.0


def test_uniform_distribution() -> None:
    """Test UniformDistribution sampling bounds, logpdf, and CDF integration."""
    dist = UniformDistribution(2.0, 5.0)
    samples = dist.rvs(100)
    
    assert samples.shape == (100,)
    assert np.all((samples >= 2.0) & (samples <= 5.0))
    
    x = np.array([1.0, 3.0, 6.0])
    logp = dist.logpdf(x)
    assert logp[0] == -np.inf
    assert logp[2] == -np.inf
    np.testing.assert_allclose(logp[1], -np.log(3.0))
    
    # Generic CDF numerical integration check
    np.testing.assert_allclose(dist.cdf(3.5), 0.5, atol=1e-3)


def test_powerlaw() -> None:
    """Test PowerLaw physical mapping and probability density."""
    dist = PowerLaw(alpha=2.0, minimum=1.0, maximum=2.0)
    
    samples = dist.rvs_physical(100)
    assert np.all((samples >= 1.0) & (samples <= 2.0))
    
    x = np.array([1.5])
    # p(x) \propto x^2. Normalization over [1, 2] is \int x^2 dx = 7/3.
    # Therefore p(1.5) = 1.5^2 / (7/3)
    expected_pdf = (1.5**2) / (7.0 / 3.0)
    np.testing.assert_allclose(dist.pdf_physical(x), expected_pdf)


def test_sine_cosine_uniform() -> None:
    """Test trigonometric physical mapping bounds."""
    sine_dist = SineUniform()
    sine_samples = sine_dist.rvs_physical(100)
    assert np.all((sine_samples >= -np.pi / 2) & (sine_samples <= np.pi / 2))
    
    cos_dist = CosineUniform()
    cos_samples = cos_dist.rvs_physical(100)
    assert np.all((cos_samples >= 0.0) & (cos_samples <= np.pi))


def test_log10_uniform() -> None:
    """Test Log10Uniform Jacobian inverse transforms and mapping."""
    dist = Log10Uniform(minimum=10.0, maximum=1000.0)
    
    samples = dist.rvs_physical(100)
    assert np.all((samples >= 10.0) & (samples <= 1000.0))
    
    x = np.array([100.0])
    # Uniform in u = log10(x) -> width = 3 - 1 = 2 -> p(u) = 0.5.
    # p(x) = p(u) * |du/dx| = 0.5 / (x * ln(10))
    expected_pdf = 0.5 / (100.0 * np.log(10.0))
    np.testing.assert_allclose(dist.pdf_physical(x), expected_pdf)


def test_discrete_uniform() -> None:
    """Test DiscreteUniform integer boundaries and logpdf assignments."""
    dist = DiscreteUniform(1, 5)
    samples = dist.rvs(100)
    
    assert np.all((samples >= 1) & (samples <= 5))
    assert np.all(samples == np.round(samples))
    
    x = np.array([0.0, 1.0, 3.5, 5.0])
    logp = dist.logpdf(x)
    
    assert logp[0] == -np.inf
    np.testing.assert_allclose(logp[1], -np.log(5.0))
    assert logp[2] == -np.inf  # Floating points not inside the integer prior
    np.testing.assert_allclose(logp[3], -np.log(5.0))


def test_poisson() -> None:
    """Test Poisson distribution evaluations against scipy references."""
    lam = 3.0
    dist = PoissonDistribution(lam=lam)
    samples = dist.rvs(100)
    
    assert np.all(samples >= 0)
    assert np.all(samples == np.round(samples))
    
    x = np.array([-1.0, 0.0, 3.0, 3.5])
    logp = dist.logpdf(x)
    
    assert logp[0] == -np.inf
    np.testing.assert_allclose(logp[1], poisson.logpmf(0, lam))
    np.testing.assert_allclose(logp[2], poisson.logpmf(3, lam))
    assert logp[3] == -np.inf


def test_multivariate_gaussian() -> None:
    """Test precision matrix decomposition and Multivariate Gaussian logpdf."""
    mu = np.array([1.0, -1.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    dist = MultivariateGaussian(names=["a", "b"], mu=mu, cov=cov)
    
    samples = dist.rvs((100,))
    assert samples.shape == (100, 2)
    
    logp = dist.logpdf(mu.reshape(1, 2))
    expected_logp = multivariate_normal.logpdf(mu, mean=mu, cov=cov) # type: ignore
    np.testing.assert_allclose(logp[0], expected_logp)


def test_probdistcontainer_string_keys() -> None:
    """Ensure ProbDistContainer can aggregate disjoint string parameter priors."""
    priors = {
        "a": UniformDistribution(0.0, 1.0),
        "b": UniformDistribution(1.0, 2.0)
    }
    container = ProbDistContainer(priors)
    assert container.ndim == 2
    
    x = np.array([[0.5, 1.5], [2.0, 1.5]])
    logpdf = container.logpdf(x)
    
    np.testing.assert_allclose(logpdf[0], 0.0)  # Both dimensions are 1.0-width Uniforms -> logp = 0
    assert logpdf[1] == -np.inf  # 'a' is out of bounds
    
    samples = container.rvs(100)
    assert samples.shape == (100, 2)
    assert np.all((samples[:, 0] >= 0.0) & (samples[:, 0] <= 1.0))
    assert np.all((samples[:, 1] >= 1.0) & (samples[:, 1] <= 2.0))


def test_probdistcontainer_tuple_keys() -> None:
    """Ensure ProbDistContainer resolves integer slices and tuple indexing properly."""
    priors = {
        (0, 1): MultivariateGaussian(
            names=["x", "y"], 
            mu=np.array([0.0, 0.0]), 
            cov=np.eye(2)
        ),
        2: UniformDistribution(0.0, 1.0)
    }
    container = ProbDistContainer(priors)
    assert container.ndim == 3
    
    x = np.array([[0.0, 0.0, 0.5]])
    logp = container.logpdf(x)
    expected_logp = multivariate_normal.logpdf([0, 0], mean=[0, 0], cov=np.eye(2)) # type: ignore
    np.testing.assert_allclose(logp[0], expected_logp)


def test_periodic_container() -> None:
    """Test correct phase wrapping and periodic shortest-path distance execution."""
    periodic = {
        "branch1": {
            "a": 2.0 * np.pi
        }
    }
    key_order = {"branch1": ["a", "b"]}
    container = PeriodicContainer(periodic, key_order)
    
    p = {"branch1": np.array([[[2.0 * np.pi + 1.0, 5.0]]])}
    wrapped = container.wrap(p)
    np.testing.assert_allclose(wrapped["branch1"][0, 0, 0], 1.0)
    np.testing.assert_allclose(wrapped["branch1"][0, 0, 1], 5.0)  # No period set
    
    p1 = {"branch1": np.array([[[0.1, 0.0]]])}
    p2 = {"branch1": np.array([[[2.0 * np.pi - 0.1, 0.0]]])}
    
    diff = container.distance(p1, p2)
    # distance = (2pi - 0.1) - 0.1 = 2pi - 0.2. Wrapped shortest path becomes -0.2
    np.testing.assert_allclose(diff["branch1"][0, 0, 0], -0.2)


def test_transform_container() -> None:
    """Test fixed-value interpolation alongside standard transformation logic."""
    fill_dict = {"c": 5.0}
    
    def square(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x ** 2
        
    def root(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(x)

    container = TransformContainer(
        input_basis=["a", "b"],
        output_basis=["a", "b", "c"],
        parameter_transforms={"a": square},
        inverse_parameter_transforms={"a": root},
        fill_dict=fill_dict
    )
    
    params = np.array([[3.0, 2.0]])
    
    # Forward: filling 'c' and squaring 'a'
    transformed = container.both_transforms(params)
    assert transformed.shape == (1, 3)
    np.testing.assert_allclose(transformed[0], [9.0, 2.0, 5.0])
    
    # Reverse: rooting 'a' and dropping fixed 'c'
    inverse = container.both_inverse_transforms(transformed)
    assert inverse.shape == (1, 2)
    np.testing.assert_allclose(inverse[0], [3.0, 2.0])


def test_probdistcontainer_conditional_dependencies() -> None:
    """
    Verify that ProbDistContainer correctly resolves execution order for
    conditional priors and routes dependent variables from the state array.
    """
    class ConditionalGaussian(Prior):
        """A dummy conditional prior where 'b' depends on the value of 'a'."""

        required_variables = ("a",)

        def rvs(
            self,
            size: int | tuple[int, ...] = (1,),
            a: NDArray[np.float64] | float = 0.0,
            **kwargs
        ) -> NDArray[np.float64]:
            return np.random.normal(loc=a, scale=1.0, size=size)

        def logpdf(
            self,
            x: NDArray[np.float64],
            a: NDArray[np.float64] | float = 0.0,
            **kwargs
        ) -> NDArray[np.float64]:
            return -0.5 * ((x - a) ** 2) - 0.5 * np.log(2 * np.pi)

    # Note: 'b' depends on 'a'. ProbDistContainer must sample 'a' first,
    # even if we reversed the insertion order of this dict.
    priors = {
        "a": UniformDistribution(minimum=0.0, maximum=10.0),
        "b": ConditionalGaussian()
    }

    container = ProbDistContainer(priors)
    assert container.ndim == 2

    samples = container.rvs(100)
    assert samples.shape == (100, 2)

    a_samples = samples[:, 0]
    b_samples = samples[:, 1]

    logp = container.logpdf(samples)

    expected_logp_a = np.full(100, -np.log(10.0))
    expected_logp_b = -0.5 * ((b_samples - a_samples) ** 2) - 0.5 * np.log(2 * np.pi)

    np.testing.assert_allclose(logp, expected_logp_a + expected_logp_b)


def test_probdistcontainer_multiindex_conditional_dependency() -> None:
    """
    A multi-index (tuple-key) prior that correctly reshapes a single-index
    dependency should still work under the new shape contract.
    """
    class JointConditional(Prior):
        """2-param prior (c, d) where both are centered on 'a'."""

        required_variables = ("a",)

        def rvs(self, size=(1,), a=0.0, **kwargs):
            a_arr = np.asarray(a)
            out = np.random.normal(loc=a_arr[..., None], scale=1.0, size=size + (2,))
            return out

        def logpdf(self, x, a=0.0, **kwargs):
            # x: (N, 2), a: (N,) -> reshape correctly ourselves
            a_col = np.asarray(a)[:, None]
            return (-0.5 * ((x - a_col) ** 2) - 0.5 * np.log(2 * np.pi)).sum(axis=-1)

    priors = {
        "a": UniformDistribution(minimum=0.0, maximum=10.0),
        ("c", "d"): JointConditional(),
    }

    container = ProbDistContainer(priors)
    assert container.ndim == 3

    samples = container.rvs(50)
    assert samples.shape == (50, 3)

    # should not raise, and should return one value per sample
    logp = container.logpdf(samples)
    assert logp.shape == (50,)
    

def test_transform_container_mult_param() -> None:
    """Test multiparameter lambda mappings execution paths."""
    def swap(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return b, a
        
    container = TransformContainer(
        input_basis=["a", "b"],
        output_basis=["a", "b"],
        parameter_transforms={("a", "b"): swap},
        inverse_parameter_transforms={("a", "b"): swap}
    )

    params = np.array([[1.0, 2.0]])
    transformed = container.transform_base_parameters(params)
    np.testing.assert_allclose(transformed[0], [2.0, 1.0])
    
    inverse = container.inverse_transform_base_parameters(transformed)
    np.testing.assert_allclose(inverse[0], [1.0, 2.0])
    
    
def test_probdistcontainer_conditional_dependency_bad_broadcast_raises() -> None:
    """
    A conditional prior that does NOT correctly reshape a dependency against
    a multi-index input should raise a clear error rather than silently
    return a wrong-shaped (or coincidentally same-shaped) result.

    Depending on the specific shapes involved, this may be caught either by
    numpy's own broadcasting rules (shapes are incompatible outright) or by
    ProbDistContainer's own output-size check (shapes are broadcast-
    compatible but produce the wrong number of values). Both are acceptable
    outcomes here - the only unacceptable outcome is silently returning a
    wrong-shaped result without raising at all.
    """
    class BrokenJointConditional(Prior):
        """Forgets to reshape 'a' against the (N, 2) input -> broadcasts wrong."""

        required_variables = ("a",)

        def rvs(self, size=(1,), a=0.0, **kwargs):
            return np.random.normal(loc=0.0, scale=1.0, size=size + (2,))

        def logpdf(self, x, a=0.0, **kwargs):
            # BUG: a is (N,), x is (N, 2) -> numpy raises outright here
            # since 2 != 100 and neither dim is 1.
            return -0.5 * ((x - a) ** 2) - 0.5 * np.log(2 * np.pi)

    priors = {
        "a": UniformDistribution(minimum=0.0, maximum=10.0),
        ("c", "d"): BrokenJointConditional(),
    }

    container = ProbDistContainer(priors)
    samples = container.rvs(100)  # N=100, deliberately == a common coincidence size

    with pytest.raises(ValueError):
        container.logpdf(samples)


def test_probdistcontainer_silent_broadcast_caught_by_size_check() -> None:
    """
    A conditional prior whose output silently broadcasts to the wrong shape
    (rather than raising a numpy broadcasting error outright) must still be
    caught. This is the dangerous case numpy alone would NOT catch - e.g.
    (N, 1) vs (N,) silently broadcasts to (N, N) without any numpy error,
    which is exactly the scenario that originally motivated this fix.
    """
    class SilentlyWrongConditional(Prior):
        """Returns (N, N) via numpy's silent (N,1)-vs-(N,) broadcasting."""

        required_variables = ("a",)

        def rvs(self, size=(1,), a=0.0, **kwargs):
            return np.random.normal(loc=0.0, scale=1.0, size=size + (2,))

        def logpdf(self, x, a=0.0, **kwargs):
            # x[:, [0]] is (N, 1), a is (N,) -> broadcasts to (N, N), no
            # numpy error raised, just a wrong-shaped, wrong-valued result.
            col = x[:, [0]]
            a_arr = np.asarray(a)
            return -0.5 * ((col - a_arr) ** 2) - 0.5 * np.log(2 * np.pi)

    priors = {
        "a": UniformDistribution(minimum=0.0, maximum=10.0),
        ("c", "d"): SilentlyWrongConditional(),
    }

    container = ProbDistContainer(priors)
    samples = container.rvs(50)  # N=50

    with pytest.raises(ValueError, match="expected exactly"):
        container.logpdf(samples)


def test_package_does_not_import_deprecated_prior_module() -> None:
    """Importing the sampler and its moves must leave the deprecated eryn.prior unimported.

    The check runs in a fresh interpreter, because this test module imports eryn.prior itself.
    """
    code = "import sys, eryn.ensemble, eryn.moves; print('eryn.prior' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_delta_function_is_a_point_mass_of_one() -> None:
    """The log density is 0 at the peak and -inf elsewhere, never +inf."""
    dist = DeltaFunction(2.0)
    logp = dist.logpdf(np.array([2.0, 1.0, 2.5]))
    np.testing.assert_array_equal(logp, [0.0, -np.inf, -np.inf])
    np.testing.assert_array_equal(dist.rvs(3), [2.0, 2.0, 2.0])


def test_sampler_runs_with_a_delta_function_parameter() -> None:
    """A container holding a fixed parameter starts a sampler, which keeps it at its peak."""
    container = ProbDistContainer(
        {"a": UniformDistribution(0.0, 1.0), "fixed": DeltaFunction(2.0)}
    )

    def log_like(x):
        return -0.5 * (x[0] - 0.5) ** 2 / 0.01

    nwalkers = 8
    np.random.seed(5)
    sampler = EnsembleSampler(nwalkers, 2, log_like, {"model_0": container})
    sampler.run_mcmc(container.rvs(size=(1, nwalkers, 1)), 20, progress=False)

    chain = sampler.backend.get_chain()["model_0"]
    np.testing.assert_array_equal(chain[..., 1], 2.0)
    assert np.all(np.isfinite(sampler.backend.get_log_prior()))

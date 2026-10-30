Normalizing-Flow Proposals
==========================

The ``eryn.flows`` subpackage provides backend-agnostic infrastructure for
normalizing-flow MCMC proposals.  A flow is trained on samples that approximate
the target (offline, from a prior run, or online, harvested from the live cold
chain) and is then used as an independence proposal: it draws candidate points
from the learned density with the exact log-density required for a valid
Metropolis-Hastings correction.

Installation
------------

The pure-Python core of ``eryn.flows`` (the :class:`~eryn.flows.Flow` ABC,
transforms, conditioning, and executors) ships with Eryn and imports no heavy
dependencies.  The PyTorch backend (:class:`~eryn.flows.ZukoFlow`,
:class:`~eryn.flows.WhiteningTransform`) requires the optional ``flow`` extra,
which installs ``torch`` and ``zuko``::

    pip install eryn[flow]

Importing ``eryn.flows`` never triggers a torch import; the backend is loaded
lazily on first use (see :func:`~eryn.flows.get_flow_wrapper`).

Architecture
------------

The subpackage is organised around a small set of seams:

- **Flow ABC and backends.**  :class:`~eryn.flows.Flow` is the abstract base
  class.  Concrete backends subclass it; :class:`~eryn.flows.ZukoFlow` (the
  ``zuko`` backend) is the implementation available today, and a ``jax`` /
  ``flowjax`` backend is planned.  :func:`~eryn.flows.get_flow_wrapper` is the
  lazy factory that returns a backend flow class without importing torch at
  module load time.
- **Data transforms.**  :class:`~eryn.flows.DataTransform` is an invertible
  preprocessing transform that exposes its exact log-determinant Jacobian term,
  which is required for a correct MH acceptance factor.
  :class:`~eryn.flows.IdentityTransform` is the no-op default;
  :class:`~eryn.flows.WhiteningTransform` is a periodic-aware per-condition
  whitening preprocessor.
- **Conditioning.**  :class:`~eryn.flows.ConditioningStrategy` is the protocol
  for encoding discrete condition ids (e.g. reversible-jump leaf counts) as
  context vectors; :class:`~eryn.flows.OneHotLeafConditioning` is the default
  one-hot implementation.
- **Trainer executors.**  :class:`~eryn.flows.TrainerExecutor` is the seam
  between sampling and training.  :class:`~eryn.flows.InlineExecutor` fits in the
  sampler process, while :class:`~eryn.flows.ProcessExecutor` trains a clone of
  the flow in a spawned worker process so sampling is never blocked.
- **Moves.**  The proposals themselves live in ``eryn.moves``:
  :class:`~eryn.moves.ConditionalFlowMove` (online or offline flow proposal),
  :class:`~eryn.moves.FlowGaussianMove` (a flow on a subset of the parameters
  and a Gaussian random walk on the rest, in one proposal) and
  :class:`~eryn.moves.IndependentProposalMove` (a generic independence proposal
  used as a baseline, e.g. with a GMM).

Quickstart
----------

**Offline** — fit a flow on existing samples, then propose with it::

    import numpy as np
    from eryn.flows import ZukoFlow
    from eryn.moves import ConditionalFlowMove

    flow = ZukoFlow(dims=4, device="cpu", seed=0)
    flow.fit({0: training_samples}, n_epochs=100, batch_size=512)

    fmove = ConditionalFlowMove(flow, branch_name="x")
    fmove.active_condition = 0
    # pass fmove into EnsembleSampler(..., moves=[(fmove, weight), ...])

**Online** — train in a background process while sampling, hot-loading new
weights as they land::

    from eryn.flows import ProcessExecutor
    from eryn.moves import ConditionalFlowMove, StretchMove

    with ProcessExecutor(flow, epochs_per_round=15, min_train_samples=2000) as ex:
        flow_move = ConditionalFlowMove(flow, "x", executor=ex, harvest_every=5)
        # mix with StretchMove to stay ergodic while the flow learns:
        # moves = [(StretchMove(), 0.7), (flow_move, 0.3)]
        # run the sampler inside the `with` block; ConditionalFlowMove harvests the cold
        # chain into the executor and polls for new versioned weights.

Examples
--------

Two runnable scripts demonstrate the full workflow:

- ``examples/flow_proposal_benchmark.py`` — offline go/no-go gate comparing a
  frozen flow against ``StretchMove`` and a GMM baseline on
  ESS-per-likelihood-eval.
- ``examples/flow_online_training_toy.py`` — the smallest end-to-end online
  loop (harvest cold chain, train in a second process, hot-load weights).

API Reference
-------------

Flow Base Classes
~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.flows.Flow
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.FlowHistory
    :members:
    :show-inheritance:

.. autoclass:: eryn.flows.FlowProposalDistribution
    :members:
    :show-inheritance:
    :inherited-members:

Backend Factory
~~~~~~~~~~~~~~~

.. autofunction:: eryn.flows.get_flow_wrapper

Torch Backend
~~~~~~~~~~~~~

Requires the ``flow`` extra (``pip install eryn[flow]``).

.. autoclass:: eryn.flows.ZukoFlow
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.WhiteningTransform
    :members:
    :show-inheritance:
    :inherited-members:

Data Transforms
~~~~~~~~~~~~~~~

.. autoclass:: eryn.flows.DataTransform
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.IdentityTransform
    :members:
    :show-inheritance:
    :inherited-members:

Conditioning
~~~~~~~~~~~~

.. autoclass:: eryn.flows.ConditioningStrategy
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.OneHotLeafConditioning
    :members:
    :show-inheritance:
    :inherited-members:

Trainer Executors
~~~~~~~~~~~~~~~~~~

.. autoclass:: eryn.flows.TrainerExecutor
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.InlineExecutor
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.ProcessExecutor
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.flows.FlowSpec
    :members:
    :show-inheritance:

.. autoclass:: eryn.flows.WorkerConfig
    :members:
    :show-inheritance:

.. autoexception:: eryn.flows.TrainerError
    :members:
    :show-inheritance:

Flow Moves
~~~~~~~~~~

.. autoclass:: eryn.moves.ConditionalFlowMove
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.moves.FlowGaussianMove
    :members:
    :show-inheritance:

.. autoclass:: eryn.moves.FlowNUTSMove
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: eryn.moves.IndependentProposalMove
    :members:
    :show-inheritance:
    :inherited-members:

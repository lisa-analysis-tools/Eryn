# Eryn: a multi-purpose MCMC sampler

Eryn is an advanced MCMC sampler. It has the capability to run with parallel tempering, multiple model types, and unknown counts within each model type using Reversible Jump MCMC techniques. It can also advance many fully independent sampler runs at once (the optional `nsamplers` axis on the Ensemble Sampler), batching their likelihood and prior evaluations into single calls to keep a GPU busy. Eryn is heavily based on [emcee](https://emcee.readthedocs.io/en/stable/). The `emcee` base structure with the Ensemble Sampler, State objects, proposal setup, and storage backends is carried over into Eryn with small changes to account for the increased complexity. In a simple sense, Eryn is an advanced (and slightly more complicated) version of `emcee`. 

If you use Eryn in your publication, please cite the paper [arXiv:2303.02164](https://arxiv.org/abs/2303.02164), its [zenodo](https://zenodo.org/records/17162828), and [emcee](https://emcee.readthedocs.io/en/stable/). The documentation for Eryn can be found here: [lisa-analysis-tools.github.io/Eryn](https://lisa-analysis-tools.github.io/Eryn). You will find the code on Github: [github.com/lisa-analysis-tools/Eryn](https://github.com/lisa-analysis-tools/Eryn). 

## Getting Started

Below is a quick set of instructions to get you started with `eryn`.

```
pip install eryn
```
To import eryn:

```
from eryn.ensemble import EnsembleSampler
```

See [examples notebook](https://github.com/lisa-analysis-tools/Eryn/blob/main/examples/Eryn_tutorial.ipynb) for more info. You can also navigate the [Documentation](https://lisa-analysis-tools.github.io/Eryn/index.html) pages.

### Normalizing-flow proposals

Eryn ships an optional `eryn.flows` subpackage for normalizing-flow MCMC proposals: a flow is trained on samples that approximate the target (offline from a prior run, or online by harvesting the live cold chain) and is then used as an independence proposal with the exact log-density needed for a valid Metropolis-Hastings correction. The proposals themselves are `ConditionalFlowMove` and `IndependentProposalMove` in `eryn.moves`.

The PyTorch backend (`ZukoFlow`, backed by [zuko](https://github.com/probabilists/zuko)) requires the optional `flow` extra:

```
pip install eryn[flow]
```

See the [flows documentation](https://mikekatz04.github.io/Eryn/user/flows.html) and the `examples/flow_proposal_benchmark.py` (offline) and `examples/flow_online_training_toy.py` (online) scripts.


### Prerequisites

Eryn has only a few python-based dependencies: `tqdm`, `corner` for plotting, `numpy`, `matplotlib`. 

### Installing

If you are not planning to develop the code, you can just install the latest version with the pip installation technique given above. Otherwise, you can just clone the repo and run `pip install .` inside of the Eryn directory. To run tests on Eryn during development, you can run the following in the main Eryn directory:
```
python -m unittest discover
```


## Running the Tests

In the main directory of the package run in the terminal:
```
python -m unittest discover
```


## Contributing

Please read [CONTRIBUTING](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us. See [CONTRIBUTORS](CONTRIBUTORS.md) for those that have authored code for and contributed to Eryn.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/lisa-analysis-tools/Eryn/tags).

## Citation

When using this package, please cite at minimum the following sources:

```
@article{Karnesis:2023ras,
    author = "Karnesis, Nikolaos and Katz, Michael L. and Korsakova, Natalia and Gair, Jonathan R. and Stergioulas, Nikolaos",
    title = "{Eryn : A multi-purpose sampler for Bayesian inference}",
    eprint = "2303.02164",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    month = "3",
    year = "2023"
}

@software{michael_katz_2023_7705496,
  author       = {Michael Katz and
                  Nikolaos Karnesis and
                  Natalia Korsakova},
  title        = {lisa-analysis-tools/Eryn: first full release},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.7705496},
  url          = {https://doi.org/10.5281/zenodo.7705496}
}

@ARTICLE{2013PASP..125..306F,
       author = {{Foreman-Mackey}, Daniel and {Hogg}, David W. and {Lang}, Dustin and {Goodman}, Jonathan},
        title = "{emcee: The MCMC Hammer}",
      journal = {\pasp},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Computational Physics, Statistics - Computation},
         year = 2013,
        month = mar,
       volume = {125},
       number = {925},
        pages = {306},
          doi = {10.1086/670067},
archivePrefix = {arXiv},
       eprint = {1202.3665},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

Depending on which proposals are used, you may be required to cite more sources. Please make sure you do this properly. 

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments

* We wish to thank S. Babak, M. Le Jeune, S. Marsat, T. Littenberg, and N. Cornish for their useful comments and very fruitful discussions. 
* N Stergioulas and N Karnesis acknowledge  support from the Gr-PRODEX 2019 funding program (PEA 4000132310).

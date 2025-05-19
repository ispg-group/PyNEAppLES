# Optimal Sampling Tool for PyNEAPPles

## About
This repository contains a representative sampler tool, `rep_sampler_2d.py`, for the selection of the most representative molecular geometries for spectra modeling. This uses a simulated annealing algorithm to iteratively converge on an optimal subset of geometries that minimizes Kullback-Leibler divergence between itself and the full ensemble.

It has been designed for use with the `uv` Python package manager.

This builds on previous work by Stepan Sršeň and Petr Slavíček.[1][2]

## Installation
To install and run `rep_sampler_2d` from PyNEAPPles, run the following commands from the top directory:

```sh
uv venv
uv pip install -e .
uv run python
```

Then, in the Python environment:

```python
import pyneapples
```

## Using `rep_sampler_2d`
An example of how to use this tool is shown in `repsample_testcall.py`.

Essentially, an instance of the class `GeomReduction` is created, specifying:
- The number of geometries to expect
- The number of excited states per geometry
- The number of geometries to be included in the optimal subset
- The number of cores to be used
- The number of reduction jobs to be carried out
- Whether or not the geometries should be weighted according to spectroscopic significance

The data is then read into the instance, and `reduce_geoms` is called. The data can be read in numerous ways, as seen in `repsample_testcall.py` and `repsample_acetylcall.py`.

For an example workflow using the [**AtmoSpec**](https://github.com/ispg-group/aiidalab-ispg) photoabsorption calculation tool, the `README.md` in `acetaldehyde` contains a full breakdown.

## References

[1] Š. Sršeň and P. Slavíček, *J. Chem. Theory Comput.*, 2021, **17**, 6395–6404, DOI: [10.1021/acs.jctc.1c00749](https://doi.org/10.1021/acs.jctc.1c00749)

[2] Stepan Sršeň GitHub: [PyNEAppLES Repository](https://github.com/stepan-srsen/PyNEAppLES), accessed 18/03/2025.

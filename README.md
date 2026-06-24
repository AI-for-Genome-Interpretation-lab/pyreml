# pyreml

**pyreml** is a general-purpose linear mixed model solver. It fits linear mixed models for a wide range of applications: quantitative genetics, spatial statistics, health sciences, among others.

Models are fitted by direct differentiation of the Restricted Maximum Likelihood (REML), using [**PyTorch**](https://pytorch.org/) for variance parameter estimation. It benefits from PyTorch parallelization and GPU acceleration.

## Installation

Install **pyreml** with `pip` or `conda`:

```py
pip install pyreml
```

## Main concepts

**pyreml** decomposes each random effect $\mathbf{a}$ (and the residual) with a
Kronecker product:

$$
\mathbf{a} \sim \mathcal{N}(\mathbf{0}, \mathbf{G_a}),
\qquad
\mathbf{G_a} = \mathbf{\Sigma_a} \otimes \mathbf{K_a},
$$

where:

- $\mathbf{\Sigma_a}$ is the **left-hand** factor, *i.e.* the covariance *across
the components* of the effect

- $\mathbf{K_a}$ is the **right-hand** factor, *i.e.* the covariance *across its levels*.

The two factors are set independently, through `left_hand` and `right_hand` arguments of the `Random` and `Residual` objects.

The philosophy behind this parameterization is to provide a simple, flexible and unified framework for a large variety of use cases. 

**pyreml** natively supports multivariate (multiple response) analysis,
random regression and heteroscedasticity specification using [**patsy**](https://pypi.org/project/patsy/) idiom. Factor-analysis, *i.e.* the direct estimation of principal components of large $\mathbf{\Sigma_a}$, is also available.

**pyreml** also enables prediction of unobserved levels of a structured random effect, for instance in this geospatial analysis:

<div align="center">
  <img src="doc/img/kriging.png" alt="Kriging" width="500">
</div>


## Illustrative example

As an illustrative example, let's realize the genetic analysis of
the `larix` dataset, using the genetic helpers provided by **pyreml**.
This model provides a heteroscedastic structure to the residuals.

```python
from pprint import pprint
from pyreml import (
    MixedModel,
    Random,
    Residual,
    A_pedigree,
    prepare_pedigree,
    larix as df,
)

df = df[df["year"] == 2000].copy()
ped = prepare_pedigree(df[["ID","SIRE","DAM"]])
K = A_pedigree(ped)

model = MixedModel.from_dataframe(
    data     = df,
    response = "height",
    fixed    = "1 + C(BLOC)",
    random   = Random(
        formula      = "1",
        unit         = "ID",
        right_hand   = "str",
        covariance   = K,
        matrix_index = ped["id"].tolist(),
    ),
    residual = Residual(
        right_hand = "het",
        het_formula = "1 + C(BLOC)",
    ),
    device = "cuda",
)

model.fit()

pprint(model.random[0].variance)
pprint(model.residual.variance)
model.random[0].table.head()
```

## Documentation

The documentation is available at [this address](https://ai-for-genome-interpretation-lab.github.io/pyreml).

## Citation

Please cite this package as:

Marchal, A., & Raimondi, D. (2026). pyreml - general-purpose linear mixed model solver [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20826541

## Code

The code is available at [this address](https://github.com/ai-for-genome-interpretation-lab/pyreml).

## License

Copyright © 2026 CNRS, University of Montpellier

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 
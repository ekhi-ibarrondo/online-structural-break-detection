# Third-party notices

## Code ported from other projects

**fdlibm 5.3** — `breakdetect/kernels/order_not_size.py` ports `e_log.c`, `e_exp.c` and `s_erf.c`, so that `log`, `exp` and `erfc` do not depend on the platform's maths library.

```
Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.

Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this
software is freely granted, provided that this notice
is preserved.
```

**Cephes Math Library** — `breakdetect/kernels/sequence_net.py` (`ndtri`, the inverse of the standard normal CDF) follows the Cephes implementation, with its polynomial coefficients. Cephes Math Library Release 2.8, Copyright 1984, 1995, 2000 by Stephen L. Moshier; SciPy distributes it under the 3-clause BSD licence with the author's permission (see SciPy's `LICENSES_bundled.txt`).

**Acklam's algorithm** — `breakdetect/kernels/order_not_size.py` computes the inverse normal CDF with Peter J. Acklam's rational approximation, refined with one Halley step.

## Methods from the literature

The detectors are new implementations of published methods:

- **FOCuS** online changepoint detection by functional pruning: Romano, Eckley, Fearnhead and Rigaill, "Fast online changepoint detection via functional pruning CUSUM statistics", *Journal of Machine Learning Research* 24(81), 2023. The exponential-family version: Ward, Romano, Eckley and Fearnhead, arXiv:2302.04743 (2023). The non-parametric version (NP-FOCuS): Romano, Eckley and Fearnhead, *IEEE Transactions on Signal Processing* 72 (2024), arXiv:2302.02718.
- **Shiryaev-Roberts** statistics and mixtures: Tartakovsky, Nikiforov and Basseville, *Sequential Analysis: Hypothesis Testing and Changepoint Detection*, CRC Press (2014).
- **Whitening** and the local approach to change detection: Basseville and Nikiforov, *Detection of Abrupt Changes: Theory and Application*, Prentice Hall (1993).
- **Kappa-2 test** for a change in the unconditional variance: Sansó, Aragó and Carrion-i-Silvestre, "Testing for changes in the unconditional variance of financial time series", *Revista de Economía Financiera* 4 (2004).
- **GARCH(1,1)** estimated by quasi-maximum likelihood: Francq and Zakoïan, *GARCH Models*, Wiley (2010).
- **CUSUM** tests: Page, "Continuous inspection schemes", *Biometrika* (1954).

## Dependencies

numpy, numba, llvmlite and threadpoolctl (BSD-style licences). The training recipes also use LightGBM (MIT), SciPy, pandas and pyarrow (BSD-style and Apache 2.0) and PyTorch (BSD-style).

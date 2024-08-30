
# apparent

[![arXiv](https://img.shields.io/badge/arXiv-2301.12906-b31b1b.svg)](https://arxiv.org/abs/2408.16022)

**A**nalysing **P**hysician-**Pa**tient **Re**ferral **N**etwork **T**opology.

This is the associated repository for our preprint on [Characterizing Physician Referral Networks with Ricci Curvature](https://arxiv.org/abs/2408.16022). We analyze large scale networks of physician referrals across the US using discrete curvature and persistent homology. Our data, displayed using [Datasette](https://datasette.io/), is publically available at:



[https://apparent.topology.rocks/](https://apparent.topology.rocks/)



## Note
Our current codebase reflects the materials used to perform the analysis included in the manuscript. Major changes coming soon (see [PR#2](https://github.com/aidos-lab/apparent/pull/2) for drafted changes) that will restructure the codebase to focus on interacting with our online dataset.



## Installation

It is recommended to use the [`poetry`](https://python-poetry.org) package
manager. With `poetry` installed, setting up the repository works like
this:

```
$ poetry install
```

Since `poetry` creates its own virtual environment, it is easiest to
interact with scripts by calling `poetry shell`.

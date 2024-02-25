# biolord - biological representation disentanglement

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/nitzanlab/biolord/test.yaml?branch=main
[link-tests]: https://github.com/nitzanlab/biolord/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/biolord

A deep generative framework for disentangling known and unknown attributes in single-cell data.

We assume partial supervision over known attributes (categorical or ordered) along with single-cell measurements.
Given the partial supervision [biolord][link-api] finds a decomposed latent space, and provides a generative model to
obtain single-cell measurements for different cell states.

For more details read our [pubication][link-paper] in _Nature Biotechnology_.

![The biolord pipeline][badge-pipeline]

[badge-pipeline]: https://github.com/nitzanlab/biolord/assets/43661890/24192211-125e-40c8-9039-4832abefcc5b?raw=true

## Getting started

Please refer to the [documentation][link-docs].

## Installation

There are several alternative options to install biolord:

1. Install the latest release of [biolord][link-api] from [PyPI](https://pypi.org/project/biolord/):

    ```bash
    pip install biolord
    ```

2. Install the latest development version:
    ```bash
    pip install git+https://github.com/nitzanlab/biolord.git@main
    ```

## Release notes

See the [changelog][changelog].

## Contact

Feel free to contact us by [mail][email].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

```
@article{piran2024disentanglement,
  title={Disentanglement of single-cell data with biolord},
  author={Piran, Zoe and Cohen, Niv and Hoshen, Yedid and Nitzan, Mor},
  journal={Nature Biotechnology},
  pages={1--6},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```

[issue-tracker]: https://github.com/nitzanlab/biolord/issues
[changelog]: https://biolord.readthedocs.io/en/latest/changelog.html
[link-docs]: https://biolord.readthedocs.io
[link-api]: https://biolord.readthedocs.io/en/latest/api.html
[link-paper]: https://doi.org/10.1038/s41587-023-02079-x
[email]: mailto::zoe.piran@mail.huji.ac.il

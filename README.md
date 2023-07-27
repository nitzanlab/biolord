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

For more details read the [preprint][link-preprint].

![The biolord pipeline][badge-pipeline]

[badge-pipeline]: https://user-images.githubusercontent.com/43661890/222221567-09111a1a-8837-4bc5-8f71-15b596571c69.png?raw=true

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
@article{piran2023biological,
  title={Biological representation disentanglement of single-cell data},
  author={Piran, Zoe and Cohen, Niv and Hoshen, Yedid and Nitzan, Mor},
  journal={bioRxiv},
  pages={2023--03},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

[issue-tracker]: https://github.com/nitzanlab/biolord/issues
[changelog]: https://biolord.readthedocs.io/en/latest/changelog.html
[link-docs]: https://biolord.readthedocs.io
[link-api]: https://biolord.readthedocs.io/en/latest/api.html
[link-preprint]: https://www.biorxiv.org/content/10.1101/2023.03.05.531195v1
[email]: mailto::zoe.piran@mail.huji.ac.il

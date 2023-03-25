from scvi.data import synthetic_iid

import biolord


def test_package_has_version():
    biolord.__version__


def test_biolord():
    n_latent = 5
    adata = synthetic_iid()
    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=None,
        categorical_attributes_keys=["batch", "labels"],
        retrieval_attribute_key=None,
    )

    model = biolord.Biolord(
        adata=adata,
        n_latent=n_latent,
    )
    model.train(10, check_val_every_n_epoch=1, train_size=0.5, enable_checkpointing=False)

    # tests __repr__
    print(model)

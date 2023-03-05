from enum import Enum

__all__ = ["LOSS_KEYS"]


class LOSS_KEYS(str, Enum):
    """Module loss keys."""

    RECONSTRUCTION = "reconstruction_loss"
    KL_LOSS = "kl_loss"
    CLASSIFICATION = "classification_loss"
    UNKNOWN_ATTRIBUTE_PENALTY = "unknown_attribute_penalty_loss"

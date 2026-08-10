"""Backward-compatible import for legacy code.

Current training code imports ``EMAModel`` directly from Diffusers. This alias
keeps older modules that import ``models.ema.EMA`` importable while migrating.
"""

from diffusers.training_utils import EMAModel as EMA

__all__ = ["EMA"]

"""Pose-only MANO contact estimation. No image or object observation is accepted."""

from .model import ContactModel, PoseAutoencoder, PoseEncoder

__all__ = ["ContactModel", "PoseAutoencoder", "PoseEncoder"]

"""Pose-only Wuji2 residual vector quantization."""
from .model import PoseRVQ, PoseDecoder

__all__ = ['PoseRVQ', 'PoseDecoder', 'PoseRVQClassifier', 'PoseRVQJoint', 'JointTokenDecoder']

from .classifier import PoseRVQClassifier
from .joint import PoseRVQJoint, JointTokenDecoder

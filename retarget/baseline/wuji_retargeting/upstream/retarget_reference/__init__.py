"""Independent, evidence-derived reference for wuji-sdk 2026.8.31 retargeting.

This package does not import wuji_sdk. See the validation report for measured
agreement and remaining differences; it is not an official SDK release.
"""
from .session import RetargetSession, HandModel, Handedness

__all__ = ['RetargetSession', 'HandModel', 'Handedness']

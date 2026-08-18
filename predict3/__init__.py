"""Cosmos 3 multiview X-ray synthesis package (scaffolding — see docs/cosmos-predict3/PLAN.md)."""

try:
    from predict3.module import CosmosXRay2XRayPredict3Multiview
except ImportError:
    CosmosXRay2XRayPredict3Multiview = None

__all__ = [
    "CosmosXRay2XRayPredict3Multiview",
]

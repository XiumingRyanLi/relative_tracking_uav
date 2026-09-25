"""Vendored NVIDIA DOPE inference code (network, cuboid, PnP solver)."""
from .cuboid import Cuboid3d, CuboidVertexType, CuboidLineIndexes
from .cuboid_pnp_solver import CuboidPNPSolver
from .detector import DopeNetwork, ObjectDetector, transform as dope_image_transform

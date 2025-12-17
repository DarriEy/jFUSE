"""
FUSE (Framework for Understanding Structural Errors) subpackage.

This subpackage implements the hydrological model physics from Clark et al. (2008),
providing multiple model architectures that can be combined to create different
conceptual rainfall-runoff models.
"""

from .config import (
    ModelConfig,
    FUSEConfig,
    UpperLayerArch,
    LowerLayerArch,
    BaseflowType,
    PercolationType,
    SurfaceRunoffType,
    EvaporationType,
    InterflowType,
    SnowType,
    RoutingType,
    RainfallErrorType,
    PRMS_CONFIG,
    SACRAMENTO_CONFIG,
    TOPMODEL_CONFIG,
    VIC_CONFIG,
    get_config,
    load_decisions_file,
    parse_decisions_file,
    config_from_decisions,
    write_decisions_file,
    enumerate_all_configs,
)

from .state import (
    State,
    Flux,
    Parameters,
    Forcing,
    PARAM_BOUNDS,
    PARAM_NAMES,
    NUM_PARAMETERS,
    get_param_bounds_arrays,
)

from .model import (
    FUSEModel,
    create_fuse_model,
    fuse_step,
    fuse_simulate,
)

from . import physics

__all__ = [
    # Config
    "ModelConfig",
    "FUSEConfig",
    "UpperLayerArch",
    "LowerLayerArch",
    "BaseflowType",
    "PercolationType",
    "SurfaceRunoffType",
    "EvaporationType",
    "InterflowType",
    "SnowType",
    "RoutingType",
    "RainfallErrorType",
    "PRMS_CONFIG",
    "SACRAMENTO_CONFIG",
    "TOPMODEL_CONFIG",
    "VIC_CONFIG",
    "get_config",
    "load_decisions_file",
    "parse_decisions_file",
    "config_from_decisions",
    "write_decisions_file",
    "enumerate_all_configs",
    # State
    "State",
    "Flux",
    "Parameters",
    "Forcing",
    "PARAM_BOUNDS",
    "PARAM_NAMES",
    "NUM_PARAMETERS",
    "get_param_bounds_arrays",
    # Model
    "FUSEModel",
    "create_fuse_model",
    "fuse_step",
    "fuse_simulate",
    # Physics
    "physics",
]

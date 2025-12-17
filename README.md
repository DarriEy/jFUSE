# jFUSE: JAX Implementation of FUSE

A fully differentiable JAX implementation of the Framework for Understanding Structural Errors (FUSE) hydrological model from Clark et al. (2008), with Muskingum-Cunge routing.

## Features

- **Fully differentiable**: Automatic differentiation through the entire model using JAX
- **JIT-compiled**: Fast execution with XLA compilation
- **FUSE decision file compatible**: Read standard FUSE decision files to configure model structure
- **All model combinations**: Support for 432+ model structure combinations
- **Muskingum-Cunge routing**: Network-based streamflow routing with adaptive parameters
- **Gradient-based calibration**: Built-in calibration with optax optimizers
- **GPU-ready**: Seamless scaling to GPU with JAX

## Installation

```bash
# Clone or download the package
cd jfuse

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev]"
```

### Requirements
- Python >= 3.9
- JAX >= 0.4.0
- equinox >= 0.11.0
- optax >= 0.1.7
- numpy
- xarray (for NetCDF I/O)

## Quick Start

### Basic Usage with Predefined Configurations

```python
import jax.numpy as jnp
from jfuse import FUSEModel
from jfuse.fuse import FUSEConfig

# Create model with PRMS-like configuration
config = FUSEConfig.prms()
model = FUSEModel(config)

# Get default parameters and initial state
params = model.default_params()
state = model.default_state()

# Create forcing data (precip, PET, temperature)
n_days = 365
forcing = (
    jnp.ones(n_days) * 5.0,   # precipitation [mm/day]
    jnp.ones(n_days) * 2.0,   # PET [mm/day]
    jnp.ones(n_days) * 15.0,  # temperature [°C]
)

# Run simulation
runoff, final_state = model.simulate(forcing, params, state)
print(f"Mean runoff: {float(jnp.mean(runoff)):.2f} mm/day")
```

### Loading FUSE Decision Files

jFUSE can read standard FUSE decision files directly:

```python
from jfuse.fuse import FUSEConfig, load_decisions_file

# Load from FUSE decision file
config = FUSEConfig.from_file("fuse_zDecisions.txt")
model = FUSEModel(config)

# Or load and inspect
from jfuse.fuse import parse_decisions_file
decisions = parse_decisions_file("fuse_zDecisions.txt")
print(decisions)  # {'ARCH1': 'onestate_1', 'ARCH2': 'unlimpow_2', ...}
```

### Exploring All Model Structures

```python
from jfuse.fuse import FUSEConfig, enumerate_all_configs

# Get all 432 possible model structure combinations
all_configs = FUSEConfig.all_structures()
print(f"Total configurations: {len(all_configs)}")

# Run ensemble of model structures
for name, config in list(all_configs.items())[:5]:
    model = FUSEModel(config)
    runoff, _ = model.simulate(forcing, model.default_params(), model.default_state())
    print(f"{name}: mean Q = {float(jnp.mean(runoff)):.2f} mm/day")
```

### Custom Model Configuration

```python
from jfuse.fuse import (
    FUSEConfig,
    UpperLayerArch,
    LowerLayerArch,
    BaseflowType,
    PercolationType,
    SurfaceRunoffType,
)

# Create custom configuration
config = FUSEConfig.custom(
    upper_arch=UpperLayerArch.TENSION_FREE,      # Sacramento-style upper
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,     # Single reservoir lower
    baseflow=BaseflowType.TOPMODEL,              # TOPMODEL baseflow
    percolation=PercolationType.LOWER_DEMAND,    # Lower layer controls
    surface_runoff=SurfaceRunoffType.UZ_PARETO,  # VIC-style saturation
)

model = FUSEModel(config)
print(config.describe())
```

### Gradient-Based Calibration

```python
from jfuse import FUSEModel
from jfuse.fuse import FUSEConfig
from jfuse.optim import Calibrator, CalibrationConfig

# Create model
model = FUSEModel(FUSEConfig.prms())

# Configure calibration
calib_config = CalibrationConfig(
    max_iterations=500,
    learning_rate=0.01,
    optimizer='adam',
    patience=50,
)

# Create calibrator
calibrator = Calibrator(model, calib_config)

# Run calibration
result = calibrator.calibrate(
    forcing=forcing,
    observed=observed_runoff,
    loss_fn='kge',           # 'nse' or 'kge'
    warmup_steps=365,
    verbose=True,
)

print(f"Best KGE: {1 - result['best_loss']:.4f}")
calibrated_params = result['best_params']
```

### Coupled Model with Routing

```python
from jfuse import FUSEModel
from jfuse.fuse import FUSEConfig
from jfuse.routing import RiverNetwork, MuskingumCungeRouter, create_network_from_topology
from jfuse.coupled import CoupledModel

# Create FUSE model
fuse = FUSEModel(FUSEConfig.prms())

# Create river network from topology
reach_data = {
    0: {'length': 5000, 'slope': 0.001, 'width_coef': 5.0, 'downstream': 2},
    1: {'length': 8000, 'slope': 0.002, 'width_coef': 4.0, 'downstream': 2},
    2: {'length': 10000, 'slope': 0.0005, 'width_coef': 8.0, 'downstream': -1},
}
network = create_network_from_topology(reach_data)

# Create router
router = MuskingumCungeRouter(network.to_arrays(), dt=86400.0)

# Create coupled model
coupled = CoupledModel(fuse, router, hru_areas=[100e6, 150e6, 200e6])  # m²

# Run simulation
outlet_flow, final_state = coupled.simulate(forcing, params, state)
```

## FUSE Decision File Format

jFUSE reads standard FUSE decision files:

```
! FUSE model decisions
RFERR    additive_e    ! rainfall error: additive_e or multiplica
ARCH1    onestate_1    ! upper layer: onestate_1, tension1_1, tension2_1
ARCH2    unlimpow_2    ! lower layer: unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2
QSURF    arno_x_vic    ! surface runoff: prms_varnt, arno_x_vic, tmdl_param
QPERC    perc_f2sat    ! percolation: perc_f2sat, perc_lower, perc_w2sat
ESOIL    sequential    ! evaporation: sequential, rootweight
QINTF    intflwnone    ! interflow: intflwnone, intflwsome
Q_TDH    no_routing    ! routing: no_routing, rout_gamma
SNOWM    temp_index    ! snow: no_snowmod, temp_index
```

### Decision Options

| Decision | Options | Description |
|----------|---------|-------------|
| ARCH1 | onestate_1, tension1_1, tension2_1 | Upper layer architecture |
| ARCH2 | unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2 | Lower layer architecture |
| QSURF | prms_varnt, arno_x_vic, tmdl_param | Surface runoff generation |
| QPERC | perc_f2sat, perc_lower, perc_w2sat | Percolation parameterization |
| ESOIL | sequential, rootweight | Evaporation partitioning |
| QINTF | intflwnone, intflwsome | Interflow option |
| SNOWM | no_snowmod, temp_index | Snow model |
| Q_TDH | no_routing, rout_gamma | Unit hydrograph routing |
| RFERR | additive_e, multiplica | Rainfall error model |

## Model Structure Reference

### Parent Models (from Clark et al. 2008)

| Model | ARCH1 | ARCH2 | QSURF | QPERC | ESOIL | QINTF |
|-------|-------|-------|-------|-------|-------|-------|
| PRMS | tension2_1 | unlimfrc_2 | prms_varnt | perc_w2sat | sequential | intflwsome |
| Sacramento | tension1_1 | tens2pll_2 | prms_varnt | perc_lower | sequential | intflwnone |
| TOPMODEL | onestate_1 | unlimpow_2 | tmdl_param | perc_f2sat | sequential | intflwnone |
| VIC | onestate_1 | fixedsiz_2 | arno_x_vic | perc_f2sat | rootweight | intflwnone |

## Computing Gradients

```python
import jax

# Define loss function
def loss_fn(params):
    runoff, _ = model.simulate(forcing, params, state)
    return jnp.mean((runoff[365:] - observed[365:])**2)

# Compute gradient
grad_fn = jax.grad(loss_fn)
gradients = grad_fn(params)

# Or compute loss and gradient together
value, grads = jax.value_and_grad(loss_fn)(params)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fuse.py -v

# Run with coverage
pytest tests/ --cov=jfuse --cov-report=html
```

## Command Line Interface

jFUSE provides a CLI compatible with FUSE file manager format:

```bash
# Run simulation
jfuse run fm_catch.txt bow_at_banff

# Run gradient-based calibration
jfuse run fm_catch.txt bow_at_banff --mode=calib --method=gradient

# Show file manager configuration
jfuse info fm_catch.txt

# List all 432 model structures
jfuse structures
```

### File Manager Format

jFUSE reads the standard FUSE file manager format:

```
FUSE_FILEMANAGER_V1.5
! *** paths
'/path/to/settings/'     ! SETNGS_PATH
'/path/to/forcing/'      ! INPUT_PATH
'/path/to/output/'       ! OUTPUT_PATH
! *** suffixes
'_input.nc'              ! suffix_forcing
'_elev_bands.nc'         ! suffix_elev_bands
! *** settings files
'input_info.txt'         ! FORCING_INFO
'fuse_zConstraints.txt'  ! CONSTRAINTS
'fuse_zNumerix.txt'      ! MOD_NUMERIX
'fuse_zDecisions.txt'    ! M_DECISIONS
! *** output
'run_1'                  ! FMODEL_ID
'FALSE'                  ! Q_ONLY
! *** dates
'1980-01-01'             ! date_start_sim
'2020-12-31'             ! date_end_sim
'1981-01-01'             ! date_start_eval
'2020-12-31'             ! date_end_eval
'-9999'                  ! numtim_sub
! *** evaluation
'KGE'                    ! METRIC
'1'                      ! TRANSFO
! *** calibration
'1000'                   ! MAXN
'3'                      ! KSTOP
'0.001'                  ! PCENTO
```

## Running Examples

```bash
# Basic simulation example
python examples/simple_simulation.py

# Calibration demonstration
python examples/calibration_demo.py
```

## Package Structure

```
jfuse/
├── src/jfuse/
│   ├── fuse/              # FUSE model implementation
│   │   ├── config.py      # Model configuration & decision file parsing
│   │   ├── model.py       # Main FUSE model
│   │   ├── state.py       # States, parameters, forcing
│   │   └── physics/       # Physical process modules
│   ├── routing/           # Muskingum-Cunge routing
│   │   ├── muskingum.py   # Routing physics
│   │   └── network.py     # River network topology
│   ├── coupled.py         # FUSE + routing integration
│   ├── optim/             # Calibration utilities
│   │   └── calibration.py # Gradient-based calibration
│   └── io/                # NetCDF I/O utilities
├── tests/                 # Test suite
├── examples/              # Example scripts
│   └── data/              # Sample decision files
└── README.md
```

## Performance Comparison

| Operation | jFUSE (JAX) | FUSE (Fortran) | Notes |
|-----------|-------------|----------------|-------|
| Forward pass | ~5 ms | ~2 ms | After JIT compilation |
| Gradient | ~15 ms | N/A | Automatic differentiation |
| 1000-step calibration | ~20 s | Hours* | *Manual perturbation |

## Citation

If you use jFUSE in your research, please cite:

```bibtex
@article{clark2008framework,
  title={Framework for Understanding Structural Errors (FUSE): 
         A modular framework to diagnose differences between 
         hydrological models},
  author={Clark, Martyn P and Slater, Andrew G and Rupp, David E and 
          Woods, Ross A and Vrugt, Jasper A and Gupta, Hoshin V and 
          Wagener, Thorsten and Hay, Lauren E},
  journal={Water Resources Research},
  volume={44},
  number={12},
  year={2008}
}
```

## License

MIT License - see LICENSE file for details.

## References

- Clark, M. P., et al. (2008). Framework for Understanding Structural Errors (FUSE). Water Resources Research, 44, W00B02.
- Cunge, J. A. (1969). On the subject of a flood propagation computation method (Muskingum method). Journal of Hydraulic Research, 7(2), 205-230.

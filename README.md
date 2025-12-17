# jFUSE: JAX Implementation of FUSE

A fully differentiable JAX implementation of the Framework for Understanding Structural Errors (FUSE) hydrological model from Clark et al. (2008), with Muskingum-Cunge routing.

## Features

- **Fully differentiable**: Automatic differentiation through the entire model using JAX
- **JIT-compiled**: Fast execution with XLA compilation
- **FUSE decision file compatible**: Read standard FUSE decision files to configure model structure
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
└── README.md
```

## License

MIT License - see LICENSE file for details.

## References

- Clark, M. P., et al. (2008). Framework for Understanding Structural Errors (FUSE). Water Resources Research, 44, W00B02.
- Cunge, J. A. (1969). On the subject of a flood propagation computation method (Muskingum method). Journal of Hydraulic Research, 7(2), 205-230.

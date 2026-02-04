# CLAUDE.md - sGDML Codebase Guide

## Project Overview

**sGDML** (Symmetric Gradient Domain Machine Learning) is a Python library for reconstructing accurate molecular force fields from ab initio molecular dynamics (AIMD) data. It uses kernel-based machine learning to predict energies and forces for molecular systems while incorporating molecular symmetries for improved accuracy and data efficiency.

**Official website:** http://sgdml.org/
**Documentation:** http://sgdml.org/doc/

### Key Features
- Machine learning of molecular force fields from quantum chemistry data
- Symmetry-aware predictions (sGDML) or symmetry-agnostic (GDML)
- CPU multiprocessing and optional GPU acceleration via PyTorch
- ASE (Atomic Simulation Environment) integration for molecular dynamics
- Command-line interface and Python API

## Repository Structure

```
sGDML/
├── sgdml/                    # Main Python package
│   ├── __init__.py           # Package init, version, logging config
│   ├── cli.py                # Command-line interface implementation
│   ├── core.py               # Core classes: Dataset, Model, Task (object-oriented API)
│   ├── train.py              # GDMLTrain class for training models
│   ├── predict.py            # GDMLPredict class for inference
│   ├── get.py                # Dataset/model download utility
│   ├── torchtools.py         # PyTorch GPU implementation
│   ├── dummy_pool.py         # Single-threaded pool for single-process execution
│   ├── intf/                 # Interfaces to external tools
│   │   └── ase_calc.py       # ASE calculator integration
│   ├── solvers/              # Linear system solvers
│   │   └── analytic.py       # Analytic (direct) solver
│   └── utils/                # Utility modules
│       ├── io.py             # File I/O, validation, XYZ format handling
│       ├── ui.py             # Terminal UI, progress bars, formatting
│       ├── desc.py           # Descriptor computation
│       └── perm.py           # Permutation/symmetry detection
├── scripts/                  # Dataset conversion scripts
│   ├── sgdml_dataset_from_extxyz.py
│   ├── sgdml_dataset_from_aims.py
│   ├── sgdml_dataset_from_ipi.py
│   ├── sgdml_dataset_via_ase.py
│   ├── sgdml_dataset_to_extxyz.py
│   └── sgdml_datasets_from_model.py
├── setup.py                  # Package installation config
├── setup.cfg                 # Flake8, isort configuration
├── pyproject.toml            # Black formatter configuration
└── README.md                 # Project documentation
```

## Key Concepts and Data Structures

### File Types (stored as `.npz` NumPy archives)
- **Dataset (`type='d'`):** Contains molecular geometries, atomic numbers, energies, and forces
- **Task (`type='t'`):** Training configuration with hyperparameters and sampled indices
- **Model (`type='m'`):** Trained model with kernel coefficients and metadata

### Core Classes

#### `GDMLTrain` (train.py)
- Creates training tasks from datasets
- Assembles kernel matrices
- Trains models using analytic or iterative solvers
- Handles symmetry detection and compression

#### `GDMLPredict` (predict.py)
- Loads trained models
- Predicts energies and forces
- Supports CPU multiprocessing and GPU (PyTorch)
- Auto-tunes parallel parameters with `prepare_parallel()`

#### Object-Oriented API (core.py)
- `Dataset`: Dataset wrapper with ASE file format support
- `Task`: Training task configuration
- `Model`: Trainable model with predict/test methods

## Development Setup

### Requirements
- Python 3.7+
- NumPy >= 1.19
- SciPy >= 1.1

### Optional Dependencies
- **PyTorch:** GPU acceleration (`pip install sgdml[torch]`)
- **ASE >= 3.16.2:** Molecular dynamics integration (`pip install sgdml[ase]`)

### Installation (Development)
```bash
git clone https://github.com/stefanch/sGDML.git
cd sGDML
pip install -e .
```

## Code Style and Conventions

### Formatting
- **Black:** String normalization and numeric underscore normalization disabled
- **Flake8:** Max complexity 12, ignores E501 (line length), W503, E741
- **isort:** Multi-line output style 3, trailing comma enabled

### Code Patterns
- Use `np.load(path, allow_pickle=True)` when loading `.npz` files
- Models/tasks/datasets are dict-like objects with string type identifiers
- Shared memory arrays via `multiprocessing.RawArray` for parallel workers
- Callback functions for progress reporting (see `utils/ui.py`)

### Multiprocessing
- Uses `fork` context on Unix, `ThreadPool` on Windows
- Global `glob` dict for sharing data with worker processes
- Single-process fallback via `dummy_pool.Pool` when `max_processes=1`

## CLI Commands

The package provides two CLI entry points:

### `sgdml` - Main CLI
```bash
sgdml all <dataset> <n_train> <n_valid> [n_test]  # Full training pipeline
sgdml create <dataset> <n_train> <n_valid>        # Create training tasks
sgdml train <task_dir> <valid_dataset>            # Train models
sgdml validate <model_dir> <valid_dataset>        # Validate models
sgdml select <model_dir>                          # Select best model
sgdml test <model_file> <test_dataset> [n_test]   # Test model
sgdml show <file>                                 # Display file info
sgdml reset                                       # Clear caches
```

### `sgdml-get` - Download Utility
```bash
sgdml-get dataset [name]  # Download benchmark dataset
sgdml-get model [name]    # Download pre-trained model
```

### Key CLI Options
- `-o, --overwrite`: Overwrite existing files
- `-p, --max_processes`: Limit parallel processes
- `--torch`: Enable GPU acceleration
- `--gdml`: Disable symmetries (use GDML instead of sGDML)
- `--no_E`: Train forces only without energies
- `-s, --sig`: Kernel length scale hyperparameter(s)

## Python API Usage

### Basic Prediction
```python
import numpy as np
from sgdml.predict import GDMLPredict
from sgdml.utils import io

# Load model and geometry
model = np.load('model.npz')
r, z = io.read_xyz('geometry.xyz')

# Create predictor and predict
gdml = GDMLPredict(model)
gdml.prepare_parallel()  # Optimize for performance
e, f = gdml.predict(r)
```

### Training Pipeline
```python
from sgdml.train import GDMLTrain
import numpy as np

# Load dataset
dataset = np.load('dataset.npz', allow_pickle=True)

# Create trainer and task
gdml_train = GDMLTrain()
task = gdml_train.create_task(
    train_dataset=dataset,
    n_train=200,
    valid_dataset=dataset,
    n_valid=1000,
    sig=50,
    use_sym=True,
)

# Train model
model = gdml_train.train(task)
np.savez_compressed('model.npz', **model)
```

### ASE Integration
```python
from sgdml.intf.ase_calc import SGDMLCalculator
from ase.md import VelocityVerlet
from ase import Atoms

calc = SGDMLCalculator('model.npz')
atoms = Atoms(...)
atoms.calc = calc

# Run MD simulation
dyn = VelocityVerlet(atoms, timestep=0.5)
dyn.run(1000)
```

## Important Implementation Details

### Kernel Matrix Assembly
- Implemented in `train.py:_assemble_kernel_mat_wkr()`
- Uses Matern 5/2 kernel with gradient information
- Exploits symmetry for matrix compression
- Parallel assembly via multiprocessing

### Descriptor Computation
- Inverse pairwise distances as descriptors
- Jacobians computed for force predictions
- Efficient representation using lower triangular indices

### Symmetry Detection
- Automatic permutation detection in `utils/perm.py`
- Uses graph matching algorithms
- Symmetries reduce training data requirements

### Energy Integration
- Integration constant recovered via least squares
- Consistency checks for energy/force label alignment
- Optional energy constraints in kernel (`use_E_cstr`)

## Testing and Validation

### Model Validation Workflow
1. Train multiple models with different sigma values
2. Validate each on held-out validation set
3. Select model with lowest force RMSE
4. Final test on separate test set

### Error Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Force magnitude and angle errors

## File Format Details

### Dataset Structure
```python
{
    'type': 'd',
    'name': 'molecule_name',
    'theory': 'DFT/B3LYP',
    'z': np.array([...]),       # Atomic numbers (n_atoms,)
    'R': np.array([...]),       # Positions (n_samples, n_atoms, 3)
    'E': np.array([...]),       # Energies (n_samples,)
    'F': np.array([...]),       # Forces (n_samples, n_atoms, 3)
    'md5': 'fingerprint',
    # Optional: 'lattice', 'r_unit', 'e_unit'
}
```

### Model Structure
```python
{
    'type': 'm',
    'z': np.array([...]),           # Atomic numbers
    'R_desc': np.array([...]),      # Training descriptors
    'R_d_desc_alpha': np.array([...]), # Coefficients
    'alphas_F': np.array([...]),    # Force alphas
    'sig': 50,                      # Length scale
    'perms': np.array([...]),       # Permutations
    'c': 0.0,                       # Integration constant
    'std': 1.0,                     # Label standard deviation
    # ... metadata fields
}
```

## Common Tasks for AI Assistants

### Adding New Functionality
1. Check if feature fits CLI (`cli.py`) or API (`core.py`, `train.py`, `predict.py`)
2. Follow existing callback patterns for progress reporting
3. Update docstrings following NumPy style
4. Handle both dict-based and object-based data structures

### Debugging Training Issues
1. Check dataset integrity with `sgdml show <dataset.npz>`
2. Verify energy/force consistency (watch for sign issues)
3. Review symmetry detection output
4. Check kernel assembly for numerical issues

### Performance Optimization
1. Use `prepare_parallel()` before predictions
2. Consider GPU acceleration for large systems
3. Monitor memory usage during kernel assembly
4. Adjust `max_processes` for available CPU cores

## References

1. Chmiela et al., "Machine Learning of Accurate Energy-conserving Molecular Force Fields", Science Advances (2017)
2. Chmiela et al., "Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields", Nature Communications (2018)
3. Chmiela et al., "sGDML: Constructing Accurate and Data Efficient Molecular Force Fields Using Machine Learning", Computer Physics Communications (2019)

# dcm-lab

Dynamic Causal Modelling (DCM) research project for fMRI and EEG, including UDE (universal differential equation) and neural ODE approaches.

## Environment

Running on the **head node of a Slurm HPC cluster**. Long-running processes must never be executed directly — always dispatch via Slurm.

### Virtual environment

```bash
source /home/student/r/rofritzsche/projects/dcm-lab/venv/bin/activate
```

Python is loaded via spack (`spack load /f6bu3ie`); the venv was created with `create_venv.sh`. To install/check packages:

```bash
source venv/bin/activate && pip install -r requirements.txt
```

The package itself uses a `pyproject.toml` build (setuptools). It does **not** support editable installs with plain `pip install -e .` — install normally if needed:

```bash
pip install .
```

## Running experiments

**Never run experiment scripts directly on the head node.** Use `sbatch` for batch jobs and `srun` for interactive allocations.

### Submit a batch job

```bash
sbatch job_run_multistart.sbatch
```

Reference sbatch header (from `job_run_multistart.sbatch`):

```bash
#SBATCH --job-name="dcm_job"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
```

For GPU jobs, uncomment:

```bash
#SBATCH --partition gpu
#SBATCH --gres gpu
#SBATCH --constraint="A100|H100.80gb"
```

### Interactive allocation

```bash
srun --ntasks=1 --cpus-per-task=4 --time=00:30:00 --pty bash
source /home/student/r/rofritzsche/projects/dcm-lab/venv/bin/activate
```

### Check job status

```bash
squeue -u rofritzsche
```

Slurm output files are written to `slurm-<jobid>.out` in the project root.

## Project structure

```
src/
  dcm/          # Core DCM library (fMRI + EEG models, inference, simulation)
  ude/          # Universal differential equation extensions
  ml/           # MLP building blocks
experiments/
  eeg/          # EEG experiment scripts
  fmri/         # fMRI experiment scripts
  configs/      # YAML configs per experiment
  lib/          # Shared experiment utilities (I/O, plotting, diagnostics)
tests/          # pytest test suite
```

## Tests

```bash
source venv/bin/activate && pytest tests/
```

Tests are unit/integration tests — run them locally on the head node (they are fast). Only submit to Slurm if a test run is unexpectedly long.

## Linting / formatting

```bash
source venv/bin/activate && ruff check src/ && black --check src/
```

# Earthquake risk model

Refactored from the monolithic `risk.py` (fwdriskpy) into modules with the same data layout and Monte Carlo outputs.

## Layout

```
earthquake_risk/
├── main.py                  # Single-iteration pipeline
├── risk.py                  # Monte Carlo entry (hazard / vul / loss)
├── config.py                # Paths and constants
├── data/                    # Input CSV, XLSX, MAT files
├── Results/                 # Parquet outputs from risk.py
├── risk/                    # Core package
│   ├── data_loader.py       # Split exposure + faults
│   ├── earthquake_generator.py
│   ├── distance_calculator.py
│   ├── pga_calculator.py
│   ├── risk_calculator.py
│   ├── simulation.py        # Per-iteration hazard/vul/loss
│   └── runner.py            # Parallel Monte Carlo driver
├── Earthquake_Generator.py  # Legacy import aliases
├── Distance_Calculator.py
├── PGA_Calculator.py
└── Risk_Calculator.py
```

## Run

Single stochastic year:

```bash
python main.py
```

Monte Carlo analysis (parallel iterations, parquet under `Results/`):

```bash
python risk.py
```

Edit `analysis_type`, `investigation_time`, and `block_size` at the bottom of `risk.py`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Copy missing data files into `data/` (see `data/README.md`).

# Earthquake risk model

Refactored from the monolithic `risk.py` into modules that match the original structured project.

## Layout

```
earthquake_risk/
├── main.py                  # Pipeline entry point
├── config.py                # Paths and constants
├── data/                    # Input CSV, XLSX, MAT files
├── risk/                    # Core package
│   ├── data_loader.py       # Exposure + faults
│   ├── earthquake_generator.py
│   ├── distance_calculator.py
│   ├── pga_calculator.py
│   └── risk_calculator.py
├── Earthquake_Generator.py  # Legacy import aliases
├── Distance_Calculator.py
├── PGA_Calculator.py
└── Risk_Calculator.py
```

## Run

```bash
cd earthquake_risk
python main.py
```

Or from Desktop:

```bash
python earthquake_risk/main.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy missing data files into `data/` (see `data/README.md`).

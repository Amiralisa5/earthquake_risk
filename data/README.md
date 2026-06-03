# Input data

Place these files in this folder (paths are configured in `config.py`):

| File | Description |
|------|-------------|
| `site.csv` | Site locations and province metadata |
| `building_exposure.csv` | Building unit counts by type (SH, SM, SL, CH, CM, CL, MM, ML) |
| `cost_exposure.csv` | Replacement cost per unit by building type |
| `area_exposure.csv` | Floor area per unit by building type |
| `vul_threshold.csv` | Damage-ratio thresholds per building type |
| `main_faults_iran_mid.xlsx` | Fault sources with geometry |
| `GMPEs_Vs.mat` | Ground motion prediction equations lookup |
| `Vul1.xlsx` | Vulnerability curves by building type |

`Main_Exposure.csv` is a legacy combined file and is no longer required by the pipeline.

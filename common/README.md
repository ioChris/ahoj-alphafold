# `common/` — shared configuration

## Path configuration (one place for every machine-specific path)

All personal / machine-specific absolute paths used anywhere in this repo are
centralized here, so the project can be **detached from one machine and attached
to another by editing a single file**, and so no personal path is ever committed
to the public repo.

| File | Tracked? | Purpose |
|---|---|---|
| `config.py` | ✅ committed | Root **anchors** + a named **path registry** (each entry tagged `# CURRENT` / `# DEPRECATED`). Import this from scripts. |
| `paths.example.ini` | ✅ committed | Template. Copy to `paths.ini` and fill in your roots. |
| `paths.ini` | 🚫 git-ignored | Your real machine roots. Never committed. |

### Setup
```
cp common/paths.example.ini common/paths.ini   # then edit the 5 roots
```
…or set the environment variables instead (they take precedence over the ini):
`AHOJ_AF_WORK_DIR`, `AHOJ_DB_DIR`, `AFDB_STRUCTURES_DIR`, `P2RANK_PRED_DIR`,
`AHOJ_AF_PRESENTATION_DIR`.

### Using it from a script
Put this bootstrap at the top (it finds the repo root no matter where the script
lives or which directory you launch it from — including Spyder):

```python
import sys as _sys
from pathlib import Path as _Path
for _root in _Path(__file__).resolve().parents:
    if (_root / "common" / "config.py").is_file():
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        break
from common import config as cfg

INPUT_TABLE = str(cfg.ANALYSIS_UNIFIED_LEVELS_CSV)   # e.g.
OUTPUT_DIR  = str(cfg.PLOTS_DIR)
```

`cfg.require("WORK_DIR", "AHOJ_DB_DIR")` raises a friendly error up front if a
needed anchor is unconfigured. If nothing is configured, anchors resolve to an
obvious `<UNSET:NAME>` placeholder so the module still imports.

### Auditing / cleanup
Every input and output path lives in `config.py`, tagged `# CURRENT` or
`# DEPRECATED`. To retire an old run/location, prune its `# DEPRECATED` entry —
no need to hunt through individual scripts.

## Note on `confidence_labeling.py`
A small `confidence_labeling.py` helper (verbal confidence-band mapping, vendored
from the separate AHoJ-DB methodology code `parse-ahoj/kde_plot.py`) is kept
**co-located** inside `03_alphafill/ubs/`, `03_alphafill/entries/`, and
`04_p2rank/analysis/` rather than centralized here, so each analysis directory
runs standalone in Spyder without import-path setup. The first two copies are
identical; the `04_p2rank/analysis/` copy has diverged slightly — reconcile if you
later choose to centralize it.

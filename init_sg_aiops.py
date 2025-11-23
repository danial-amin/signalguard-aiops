import os

PROJECT_ROOT = "signalguard-aiops"

DIRS = [
    "signalguard_aiops",
    "signalguard_aiops/detectors",
    "signalguard_aiops/metrics",
    "signalguard_aiops/incidents",
    "signalguard_aiops/pipelines",
    "signalguard_aiops/examples",
    "tests",
]

FILES = {
    "signalguard_aiops/__init__.py": '''"""
signalguard_aiops
-----------------
A lightweight AIOps / observability machine learning toolkit.
"""
__version__ = "0.1.0"
''',

    "signalguard_aiops/detectors/__init__.py": "from .zscore import ZScoreDetector\nfrom .ema import EMADetector\n",
    "signalguard_aiops/detectors/base.py": "# TODO: paste base detector implementation here\n",
    "signalguard_aiops/detectors/zscore.py": "# TODO: paste ZScoreDetector implementation here\n",
    "signalguard_aiops/detectors/ema.py": "# TODO: paste EMADetector implementation here\n",
    "signalguard_aiops/detectors/isolation_forest.py": "# Optional: advanced detector (sklearn)\n",

    "signalguard_aiops/metrics/__init__.py": "from .timeseries import TimeSeries\n",
    "signalguard_aiops/metrics/timeseries.py": "# TODO: paste TimeSeries implementation here\n",

    "signalguard_aiops/incidents/__init__.py": "from .incident import Incident\nfrom .scorers import IncidentScorer\n",
    "signalguard_aiops/incidents/incident.py": "# TODO: paste Incident dataclass implementation here\n",
    "signalguard_aiops/incidents/scorers.py": "# TODO: paste IncidentScorer implementation here\n",

    "signalguard_aiops/pipelines/__init__.py": "from .prometheus_client import PrometheusSeriesFetcher\n",
    "signalguard_aiops/pipelines/prometheus_client.py": "# TODO: paste PrometheusSeriesFetcher implementation here\n",

    "signalguard_aiops/examples/simple_prometheus_demo.py": "# TODO: paste example script here\n",

    "tests/test_zscore.py": "# TODO: add simple tests for ZScoreDetector\n",

    ".gitignore": "__pycache__/\n*.pyc\n.env\n.venv/\n.idea/\n.vscode/\n",
    "README.md": "# signalguard-aiops\n\nA lightweight AIOps / observability ML toolkit.\n",
    "pyproject.toml": """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "signalguard-aiops"
version = "0.1.0"
description = "A lightweight AIOps / observability ML toolkit"
authors = [
  { name = "Danial Amin" }
]
requires-python = ">=3.9"
dependencies = [
  "numpy",
  "pandas",
  "requests"
]
"""
}


def main():
    print(f"Creating project at ./{PROJECT_ROOT}")
    os.makedirs(PROJECT_ROOT, exist_ok=True)

    for d in DIRS:
        path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(path, exist_ok=True)
        print("dir:", path)

    for rel, content in FILES.items():
        path = os.path.join(PROJECT_ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print("file:", path)
        else:
            print("skip (exists):", path)

    print("Done.")


if __name__ == "__main__":
    main()

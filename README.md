# 📜 signalguard-logs

**Lightweight Log Intelligence and AIOps Toolkit**

Structured parsing • Template extraction • Semantic anomaly detection • Error burst detection • New pattern detection

## 🔍 Overview

signalguard-logs is a modular AIOps toolkit designed for log analysis, log anomaly detection, and log-based incident signals.
It complements the metric-focused signalguard-aiops package, forming the second half of a complete AIOps suite:

- **signalguard-aiops**: time-series & metrics intelligence
- **signalguard-logs**: log intelligence & pattern discovery

This package includes:

✔ **Log ingestion & structuring**
- Regex-based log parsing
- JSON log parsing
- Unified LogRecord / LogStream abstraction

✔ **Template extraction (naive Drain-style)**
- Auto-replace numbers, hex, IDs
- Quick pattern grouping
- Useful for new pattern detection

✔ **Feature engineering**
- TF–IDF vectorization
- Template extraction
- Token masking & template frequency

✔ **Log anomaly detectors**
- Error burst detector (time-window based volume spikes)
- New template detector (pattern drift)
- Semantic Isolation Forest detector (TF-IDF + IForest)

✔ **High-level "AIOps Recipes"**

Ready-to-use logic blocks:
- `ErrorBurstRecipe`
- `NewErrorPatternRecipe`
- `CombinedLogHealthRecipe` (semantic + bursts)

✔ **Demo scripts**

Fully runnable examples with synthetic logs.

## 🧱 Project Structure

```
signalguard-logs/
  signalguard_logs/
    models/
      record.py          # LogRecord dataclass
      stream.py          # LogStream container
    parsing/
      regex_parser.py    # Parse plain text logs with regex
      json_parser.py     # Parse JSON logs
    features/
      templates.py       # Template extraction
      text_vectorizer.py # TF-IDF wrapper
    detectors/
      base.py            # Base class for log detectors
      burst.py           # Burst-based log anomaly detector
      new_template.py    # New pattern detector
      semantic_iforest.py# TF-IDF + IsolationForest detector
    recipes/
      error_burst.py     # High-level error burst recipe
      new_pattern.py     # New log template recipe
      combined_health.py # Combined semantic & volume recipe
    examples/
      synthetic_error_burst.py
      synthetic_new_pattern.py
      FULL_EXAMPLE.py    # <---- Added below
```


## 📦 Installation

You can install the local package for development:

```bash
pip install -e .
```

**Dependencies** (installed automatically):
- numpy
- pandas
- scikit-learn



## 🧰 Core Concepts

### LogRecord
A single structured log entry:
```python
LogRecord(timestamp, level, message, service, extra={})
```

### LogStream
A list of records with helpers:
```python
stream.filter_level("ERROR")
stream.filter_service("payments")
stream.messages()
stream.timestamps()
```

### Parsing
- `RegexLogParser` for plain text logs
- `JsonLogParser` for structured logs

### Template Extraction
Extract stable patterns:
```
"Timeout 3001 on connection abc123ffffffff"
-> "Timeout <NUM> on connection <HEX>"
```

### Feature Engineering
- TF-IDF vectors
- Template frequency

### Detectors
- `LogBurstDetector` – time-window volume spikes
- `NewTemplateDetector` – unseen error patterns
- `SemanticIForestDetector` – anomaly messages by content

### Recipes
High-level workflows that combine detectors into ready AIOps primitives:
- `ErrorBurstRecipe` → detect operational incidents
- `NewErrorPatternRecipe` → detect novel error patterns
- `CombinedLogHealthRecipe` → semantic + volume + pattern



## 🧪 Full Example

Copy into `examples/FULL_EXAMPLE.py`

This example generates synthetic logs, parses them, extracts templates, and evaluates multiple detectors + recipes.

```python
"""
FULL DEMO for signalguard-logs

Demonstrates:
- parsing LogStream
- burst detection
- new template detection
- semantic anomaly detection
- combined health recipe
"""

import time
import random

from signalguard_logs.models import LogRecord, LogStream
from signalguard_logs.recipes import (
    ErrorBurstRecipe,
    NewErrorPatternRecipe,
    CombinedLogHealthRecipe,
)


# ---------------------------------------------------------
# Generate demo synthetic logs
# ---------------------------------------------------------

def generate_synthetic_logs(n=400, service="api"):
    records = []
    now = time.time()

    stable_msgs = [
        "DB connection timeout <NUM>",
        "User <NUM> not found",
        "Cache lookup failed for ID <HEX>",
        "Input validation error at field <NUM>",
    ]

    # First 200: normal logs
    for i in range(200):
        ts = now + i
        template = random.choice(stable_msgs)
        msg = (
            template.replace("<NUM>", str(random.randint(1, 999)))
            .replace("<HEX>", hex(random.randint(10**5, 10**8))[2:])
        )

        level = "ERROR" if random.random() < 0.1 else "INFO"

        records.append(
            LogRecord(timestamp=ts, level=level, message=msg, service=service)
        )

    # Next 100: error burst
    for i in range(200, 300):
        ts = now + i
        msg = f"DB outage: failed to connect replica {random.randint(1,3)}"
        records.append(
            LogRecord(timestamp=ts, level="ERROR", message=msg, service=service)
        )

    # Final 100: brand new pattern
    for i in range(300, 400):
        ts = now + i
        msg = f"JWT signature verification failed for token {hex(random.randint(10**10,10**12))[2:]}"
        records.append(
            LogRecord(timestamp=ts, level="ERROR", message=msg, service=service)
        )

    return LogStream(records)


# ---------------------------------------------------------
# Main Demo
# ---------------------------------------------------------

def main():
    stream = generate_synthetic_logs(service="auth")

    print("\n=== SIGNALGUARD-LOGS FULL DEMO ===")
    print(f"Total log records: {len(stream.records)}")

    # -------------------------------
    # Error burst recipe
    # -------------------------------
    burst = ErrorBurstRecipe(service="auth", level="ERROR", window_size=30, baseline_factor=3)
    burst_result = burst.run(stream)

    burst_anomalies = (burst_result["labels"] == 1).sum()
    print(f"\n[1] Error Burst Detector:")
    print(f"  Anomalous ERROR entries: {burst_anomalies}")

    # -------------------------------
    # New Error Pattern recipe
    # -------------------------------
    newp = NewErrorPatternRecipe(service="auth", level="ERROR")
    newp_result = newp.run(stream)

    newp_anomalies = (newp_result["labels"] == 1).sum()
    print(f"\n[2] New Error Pattern Detector:")
    print(f"  New patterns detected: {newp_anomalies}")

    # -------------------------------
    # Combined Health recipe
    # -------------------------------
    comb = CombinedLogHealthRecipe(service="auth")
    comb_result = comb.run(stream)

    comb_anomalies = (comb_result["labels"] == 1).sum()
    print(f"\n[3] Combined Log Health Recipe:")
    print(f"  Overall anomalies: {comb_anomalies}")

    # Show a few example anomaly messages
    print("\nSample anomalous messages:")
    for rec, label in zip(stream.records, comb_result["labels"]):
        if label == 1:
            print("  ", rec.level, rec.message)
            comb_anomalies -= 1
            if comb_anomalies < 5:
                break


if __name__ == "__main__":
    main()
```


## 🎯 Positioning (for recruiters / interviews)

You can now pitch signalguard-logs like this:

> I built a lightweight AIOps toolkit for logs. It includes parsers, template extraction, semantic vectorization, several anomaly detectors, and opinionated recipes that combine volume bursts, new error patterns, and semantic isolation forest models.
> It acts as the "log intelligence" layer complementing my signalguard-aiops package, so together I can analyze both metrics and logs in a consistent incident pipeline.

## 🚀 Next Steps (optional upgrades)

- ✔ Add BERT / sentence-transformer embeddings
- ✔ Add Drain3 or IPLoM real template miners
- ✔ Add clustering-based pattern emergence detectors
- ✔ Integrate with Loki / Elastic / OpenSearch
- ✔ Attach log anomalies to metrics incidents (cross-signal correlation)

If you want, I can also create:

- A GitHub Pages mini-dashboard for this log package
- A combined AI/ML observability architecture diagram
- A portfolio write-up section for your website
- A LinkedIn announcement for the trilogy: SignalGuard + signalguard-aiops + signalguard-logs


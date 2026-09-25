"""
Data profiling and dataset diagnostic analysis.
Measures missingness rates, text lengths, unique entities, and country distribution.
"""

import json
from pathlib import Path
from typing import Dict, Any
from .loader import stream_tsv_records

def profile_source_tsv(file_path: Path, max_rows: int = 100000) -> Dict[str, Any]:
    """Stream through up to max_rows and compute comprehensive summary statistics."""
    total = 0
    missing_name = 0
    missing_addr = 0
    missing_country = 0
    name_lengths = []
    addr_lengths = []
    countries = {}

    for record in stream_tsv_records(str(file_path)):
        total += 1
        name = record.get("business_name", "").strip()
        addr = record.get("business_address", "").strip()
        country = record.get("country", "").strip()

        if not name:
            missing_name += 1
        else:
            name_lengths.append(len(name))

        if not addr:
            missing_addr += 1
        else:
            addr_lengths.append(len(addr))

        if not country:
            missing_country += 1
        else:
            countries[country] = countries.get(country, 0) + 1

        if total >= max_rows:
            break

    avg_name_len = sum(name_lengths) / len(name_lengths) if name_lengths else 0
    avg_addr_len = sum(addr_lengths) / len(addr_lengths) if addr_lengths else 0

    return {
        "file": file_path.name,
        "sample_size": total,
        "missing_name_count": missing_name,
        "missing_name_pct": round(missing_name / total, 4) if total else 0,
        "missing_addr_count": missing_addr,
        "missing_addr_pct": round(missing_addr / total, 4) if total else 0,
        "missing_country_pct": round(missing_country / total, 4) if total else 0,
        "avg_name_length": round(avg_name_len, 2),
        "avg_addr_length": round(avg_addr_len, 2),
        "countries": countries,
    }

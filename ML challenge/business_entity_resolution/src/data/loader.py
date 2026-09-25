"""
Efficient streaming and batch TSV data loader.
Adheres to strict memory constraints (~2GB free RAM) by processing line-by-line or in chunks.
"""

import csv
import os
from typing import Iterator, Dict, Any, List, Optional
import pandas as pd

def stream_tsv_records(file_path: str) -> Iterator[Dict[str, str]]:
    """Yield records one by one as a dictionary from a TSV file with zero memory bloat."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        if not header:
            return
        header = [col.strip() for col in header]
        for row in reader:
            if not row:
                continue
            # Pad row if shorter than header
            if len(row) < len(header):
                row.extend([''] * (len(header) - len(row)))
            yield dict(zip(header, row))

def load_source_df(file_path: str, nrows: Optional[int] = None, use_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a source TSV into pandas with explicit string dtypes to prevent memory bloat."""
    dtype_dict = {
        'entity_id': 'string',
        'business_name': 'string',
        'business_address': 'string',
        'country': 'string',
    }
    if use_cols:
        dtype_dict = {k: v for k, v in dtype_dict.items() if k in use_cols}
    return pd.read_csv(
        file_path,
        sep='\t',
        nrows=nrows,
        usecols=use_cols,
        dtype=dtype_dict,
        keep_default_na=False,
        encoding='utf-8',
        engine='c'
    )

def load_ground_truth_df(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load train_ground_truth.tsv into a DataFrame."""
    return pd.read_csv(
        file_path,
        sep='\t',
        nrows=nrows,
        dtype={'source1_entity_id': 'string', 'matched_entity_ids': 'string'},
        keep_default_na=False,
        encoding='utf-8',
        engine='c'
    )

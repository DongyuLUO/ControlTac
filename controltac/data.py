"""Read the distributed training manifests."""
import csv
from pathlib import Path

def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

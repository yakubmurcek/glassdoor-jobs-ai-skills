#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

extract_rows = 10
input_csv = "us_relevant.csv"
output_csv = f"us_relevant_{extract_rows}.csv"

df = pd.read_csv(
    input_csv,
    sep=";",
    nrows=extract_rows,
    encoding="utf-8"
)

df.to_csv(
    output_csv,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print(f"Done. Exported first {extract_rows} rows.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

input_csv = "us_relevant.csv"
output_csv = "us_relevant_50.csv"

df = pd.read_csv(
    input_csv,
    sep=";",
    nrows=50,
    encoding="utf-8"
)

df.to_csv(
    output_csv,
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("Done. Exported first 50 rows.")

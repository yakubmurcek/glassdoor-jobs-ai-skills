#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone shim so `python extract_csv.py` works like the CLI subcommand."""

from ai_skills.extract_csv import main


if __name__ == "__main__":
    raise SystemExit(main())

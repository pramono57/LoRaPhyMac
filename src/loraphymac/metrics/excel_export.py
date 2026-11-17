"""Excel serialization helpers for statistics output bundles."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

import pandas as pd


def export_bundle_to_excel(
    bundle: Mapping[str, Any],
    filepath: str | PathLike[str],
) -> None:
    """Write non-empty DataFrames to a workbook in mapping order.

    Excel limits worksheet names to 31 characters.  When every entry is empty
    or non-tabular, a small placeholder sheet keeps the workbook valid.
    """
    wrote_sheet = False

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, frame in bundle.items():
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue

            frame.to_excel(
                writer,
                sheet_name=str(sheet_name)[:31],
                index=False,
            )
            wrote_sheet = True

        if not wrote_sheet:
            pd.DataFrame([{"message": "No data available for this output mode."}]).to_excel(
                writer,
                sheet_name="empty",
                index=False,
            )

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.adapters.steo_vintages import (
    parse_steo_vintage_workbook,
    parse_vintage_month_from_filename,
)


def _month_header_rows(start_year: int = 2025) -> tuple[list[object], list[object]]:
    years: list[object] = ["Forecast date:", np.nan]
    months: list[object] = ["Thursday, February 5, 2026", np.nan]

    for year in (start_year, start_year + 1):
        for month_idx, month_name in enumerate(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        ):
            years.append(year if month_idx == 0 else np.nan)
            months.append(month_name)
    return years, months


def _table_5a_sheet() -> pd.DataFrame:
    years, months = _month_header_rows()
    rows: list[list[object]] = [
        ["Table of Contents", "Table 5a. U.S. Natural Gas Supply", *([np.nan] * 24)],
        [np.nan, "STEO - test", *([np.nan] * 24)],
        years,
        months,
        [np.nan, "Supply (billion cubic feet per day)", *([np.nan] * 24)],
        [
            "NGMPPUS",
            "U.S. total marketed natural gas production",
            *[100.0 + idx for idx in range(24)],
        ],
        [
            "NGPRPUS",
            "U.S. total dry natural gas production",
            *[95.0 + idx for idx in range(24)],
        ],
    ]
    return pd.DataFrame(rows)


def _table_2_sheet() -> pd.DataFrame:
    years, months = _month_header_rows()
    rows: list[list[object]] = [
        ["Table of Contents", "Table 2. Energy Prices", *([np.nan] * 24)],
        [np.nan, "STEO - test", *([np.nan] * 24)],
        years,
        months,
        [
            "WTIPUUS",
            "West Texas Intermediate Spot Average",
            *[70.0 + idx for idx in range(24)],
        ],
        ["BREPUUS", "Brent Spot Average", *[72.0 + idx for idx in range(24)]],
        ["NGHHMCF", "Henry Hub Spot", *[2.0 + 0.05 * idx for idx in range(24)]],
        ["NGHHUUS", "Henry Hub Spot MMBtu", *[2.1 + 0.05 * idx for idx in range(24)]],
        ["NGRCUUS", "Residential Sector", *[11.0 + 0.1 * idx for idx in range(24)]],
        ["NGCCUUS", "Commercial Sector", *[8.0 + 0.1 * idx for idx in range(24)]],
        ["NGICUUS", "Industrial Sector", *[5.0 + 0.1 * idx for idx in range(24)]],
    ]
    return pd.DataFrame(rows)


def _table_4a_sheet() -> pd.DataFrame:
    years, months = _month_header_rows()
    rows: list[list[object]] = [
        ["Table of Contents", "Table 4a. Petroleum Supply", *([np.nan] * 24)],
        [np.nan, "STEO - test", *([np.nan] * 24)],
        years,
        months,
        [
            "COPRPUS",
            "U.S. total crude oil production",
            *[12.0 + 0.1 * idx for idx in range(24)],
        ],
        ["PASUPPLY", "Total Supply", *[20.0 + 0.1 * idx for idx in range(24)]],
        [
            "PATCPUSX",
            "U.S. total petroleum products consumption",
            *[19.0 + 0.1 * idx for idx in range(24)],
        ],
        [
            "PASXPUS",
            "Total commercial inventory",
            *[1500.0 - 2 * idx for idx in range(24)],
        ],
    ]
    return pd.DataFrame(rows)


def _table_5b_sheet() -> pd.DataFrame:
    years, months = _month_header_rows()
    rows: list[list[object]] = [
        [
            "Table of Contents",
            "Table 5b. Regional Natural Gas Prices",
            *([np.nan] * 24),
        ],
        [np.nan, "STEO - test", *([np.nan] * 24)],
        years,
        months,
        ["NGHHMCF", "Henry Hub spot price", *[2.0 + 0.05 * idx for idx in range(24)]],
        ["NGRCUUS", "United States average", *[11.0 + 0.1 * idx for idx in range(24)]],
        ["NGRCU_NEC", "New England", *[13.0 + 0.1 * idx for idx in range(24)]],
        ["NGRCU_PAC", "Pacific", *[9.0 + 0.1 * idx for idx in range(24)]],
        ["NGCCUUS", "United States average", *[8.0 + 0.1 * idx for idx in range(24)]],
        ["NGCCU_NEC", "New England", *[10.0 + 0.1 * idx for idx in range(24)]],
        ["NGCCU_PAC", "Pacific", *[7.0 + 0.1 * idx for idx in range(24)]],
        ["NGICUUS", "United States average", *[5.0 + 0.1 * idx for idx in range(24)]],
        ["NGICU_NEC", "New England", *[6.0 + 0.1 * idx for idx in range(24)]],
        ["NGICU_PAC", "Pacific", *[4.0 + 0.1 * idx for idx in range(24)]],
    ]
    return pd.DataFrame(rows)


def _table_10a_sheet() -> pd.DataFrame:
    years, months = _month_header_rows()
    rows: list[list[object]] = [
        [
            "Table of Contents",
            "Table 10a. Drilling Productivity Metrics",
            *([np.nan] * 24),
        ],
        [np.nan, "STEO - test", *([np.nan] * 24)],
        years,
        months,
        [np.nan, "Active rigs", *([np.nan] * 24)],
        ["RIGSAP", "Appalachia region", *[50.0 + idx for idx in range(24)]],
        ["NWDAP", "Appalachia region", *[80.0 + idx for idx in range(24)]],
        ["NWCAP", "Appalachia region", *[70.0 + idx for idx in range(24)]],
        ["DUCSAP", "Appalachia region", *[200.0 - idx for idx in range(24)]],
        ["NGNWAP", "Appalachia region", *[6.0 + 0.1 * idx for idx in range(24)]],
        ["NGEOPAP", "Appalachia region", *[-1.0 + 0.01 * idx for idx in range(24)]],
    ]
    return pd.DataFrame(rows)


def _table_10b_sheet() -> pd.DataFrame:
    years, months = _month_header_rows()
    rows: list[list[object]] = [
        [
            "Table of Contents",
            "Table 10b. Crude Oil and Natural Gas Production from Shale and Tight Formations",
            *([np.nan] * 24),
        ],
        [np.nan, "STEO - test", *([np.nan] * 24)],
        years,
        months,
        [
            "SNGPRL48",
            "Total U.S. shale dry natural gas production",
            *[75.0 + idx for idx in range(24)],
        ],
        ["SNGPRBK", "Bakken formation", *[2.0 + 0.1 * idx for idx in range(24)]],
    ]
    return pd.DataFrame(rows)


def _write_workbook(path: Path, *, include_10b: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _table_2_sheet().to_excel(writer, index=False, header=False, sheet_name="2tab")
        _table_4a_sheet().to_excel(
            writer, index=False, header=False, sheet_name="4atab"
        )
        _table_5a_sheet().to_excel(
            writer, index=False, header=False, sheet_name="5atab"
        )
        _table_5b_sheet().to_excel(
            writer, index=False, header=False, sheet_name="5btab"
        )
        _table_10a_sheet().to_excel(
            writer, index=False, header=False, sheet_name="10atab"
        )
        if include_10b:
            _table_10b_sheet().to_excel(
                writer, index=False, header=False, sheet_name="10btab"
            )


def test_parse_steo_vintage_workbook_happy_path(tmp_path: Path) -> None:
    workbook = tmp_path / "feb26_base.xlsx"
    _write_workbook(workbook)

    parsed = parse_steo_vintage_workbook(workbook)

    assert parsed.vintage_month == "2026-02"
    assert set(parsed.tables.keys()) == {
        "table_2",
        "table_4a",
        "table_5a",
        "table_5b",
        "table_10a",
        "table_10b",
    }
    assert len(parsed.tables["table_2"]) > 0
    assert len(parsed.tables["table_4a"]) > 0
    assert len(parsed.tables["table_5a"]) > 0
    assert len(parsed.tables["table_5b"]) > 0
    assert len(parsed.tables["table_10a"]) > 0
    assert len(parsed.tables["table_10b"]) > 0
    assert parsed.table_metadata["table_5a"]["series_count"] >= 2


def test_parse_steo_vintage_workbook_fails_loud_when_required_sheet_missing(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "feb26_base.xlsx"
    _write_workbook(workbook, include_10b=False)

    with pytest.raises(ContractViolation, match="reason_code=source_schema_drift"):
        parse_steo_vintage_workbook(workbook)


def test_parse_vintage_month_from_filename_contract() -> None:
    assert parse_vintage_month_from_filename("feb26_base.xlsx") == "2026-02"
    with pytest.raises(ContractViolation, match="reason_code=source_schema_drift"):
        parse_vintage_month_from_filename("steo_unknown.xlsx")

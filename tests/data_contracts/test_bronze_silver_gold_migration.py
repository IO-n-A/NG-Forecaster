from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ng_forecaster.data.gold_publish import publish_steo_gold_marts
from ng_forecaster.data.migration import migrate_data_new_inventory
from ng_forecaster.data.silver_normalize import normalize_steo_vintages


def _month_header_rows(start_year: int = 2025) -> tuple[list[object], list[object]]:
    years: list[object] = ["Forecast date:", np.nan]
    months: list[object] = ["Thursday, February 5, 2026", np.nan]
    for year in (start_year, start_year + 1):
        for month_name in [
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
        ]:
            years.append(year if month_name == "Jan" else np.nan)
            months.append(month_name)
    return years, months


def _write_test_workbook(path: Path) -> None:
    years, months = _month_header_rows()
    table_2 = pd.DataFrame(
        [
            ["Table of Contents", "Table 2", *([np.nan] * 24)],
            [np.nan, "STEO - test", *([np.nan] * 24)],
            years,
            months,
            ["WTIPUUS", "WTI", *[70.0 + i for i in range(24)]],
            ["BREPUUS", "Brent", *[72.0 + i for i in range(24)]],
            ["NGHHMCF", "Henry Hub $/Mcf", *[2.0 + 0.05 * i for i in range(24)]],
            ["NGHHUUS", "Henry Hub $/MMBtu", *[2.1 + 0.05 * i for i in range(24)]],
            ["NGRCUUS", "Residential", *[11.0 + 0.1 * i for i in range(24)]],
            ["NGCCUUS", "Commercial", *[8.0 + 0.1 * i for i in range(24)]],
            ["NGICUUS", "Industrial", *[5.0 + 0.1 * i for i in range(24)]],
        ]
    )
    table_4a = pd.DataFrame(
        [
            ["Table of Contents", "Table 4a", *([np.nan] * 24)],
            [np.nan, "STEO - test", *([np.nan] * 24)],
            years,
            months,
            ["COPRPUS", "Crude oil production", *[12.0 + 0.1 * i for i in range(24)]],
            ["PASUPPLY", "Total supply", *[20.0 + 0.1 * i for i in range(24)]],
            ["PATCPUSX", "Total consumption", *[19.0 + 0.1 * i for i in range(24)]],
            ["PASXPUS", "Commercial inventory", *[1500.0 - 2 * i for i in range(24)]],
        ]
    )
    table_5a = pd.DataFrame(
        [
            ["Table of Contents", "Table 5a", *([np.nan] * 24)],
            [np.nan, "STEO - test", *([np.nan] * 24)],
            years,
            months,
            [np.nan, "Supply", *([np.nan] * 24)],
            ["NGMPPUS", "Marketed", *[100.0 + i for i in range(24)]],
            ["NGPRPUS", "Dry", *[95.0 + i for i in range(24)]],
        ]
    )
    table_5b = pd.DataFrame(
        [
            ["Table of Contents", "Table 5b", *([np.nan] * 24)],
            [np.nan, "STEO - test", *([np.nan] * 24)],
            years,
            months,
            ["NGHHMCF", "Henry Hub", *[2.0 + 0.05 * i for i in range(24)]],
            ["NGRCUUS", "Residential US avg", *[11.0 + 0.1 * i for i in range(24)]],
            [
                "NGRCU_NEC",
                "Residential New England",
                *[13.0 + 0.1 * i for i in range(24)],
            ],
            ["NGRCU_PAC", "Residential Pacific", *[9.0 + 0.1 * i for i in range(24)]],
            ["NGCCUUS", "Commercial US avg", *[8.0 + 0.1 * i for i in range(24)]],
            [
                "NGCCU_NEC",
                "Commercial New England",
                *[10.0 + 0.1 * i for i in range(24)],
            ],
            ["NGCCU_PAC", "Commercial Pacific", *[7.0 + 0.1 * i for i in range(24)]],
            ["NGICUUS", "Industrial US avg", *[5.0 + 0.1 * i for i in range(24)]],
            [
                "NGICU_NEC",
                "Industrial New England",
                *[6.0 + 0.1 * i for i in range(24)],
            ],
            ["NGICU_PAC", "Industrial Pacific", *[4.0 + 0.1 * i for i in range(24)]],
        ]
    )
    table_10a = pd.DataFrame(
        [
            ["Table of Contents", "Table 10a", *([np.nan] * 24)],
            [np.nan, "STEO - test", *([np.nan] * 24)],
            years,
            months,
            [np.nan, "Active rigs", *([np.nan] * 24)],
            ["RIGSAP", "Appalachia region", *[50.0 + i for i in range(24)]],
            ["NWDAP", "Appalachia region", *[80.0 + i for i in range(24)]],
            ["NWCAP", "Appalachia region", *[70.0 + i for i in range(24)]],
            ["DUCSAP", "Appalachia region", *[200.0 - i for i in range(24)]],
            ["NGNWAP", "Appalachia region", *[6.0 + 0.1 * i for i in range(24)]],
            ["NGEOPAP", "Appalachia region", *[-1.0 + 0.01 * i for i in range(24)]],
        ]
    )
    table_10b = pd.DataFrame(
        [
            ["Table of Contents", "Table 10b", *([np.nan] * 24)],
            [np.nan, "STEO - test", *([np.nan] * 24)],
            years,
            months,
            ["SNGPRL48", "Total shale dry", *[75.0 + i for i in range(24)]],
            ["SNGPRBK", "Bakken formation", *[2.0 + 0.1 * i for i in range(24)]],
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table_2.to_excel(writer, index=False, header=False, sheet_name="2tab")
        table_4a.to_excel(writer, index=False, header=False, sheet_name="4atab")
        table_5a.to_excel(writer, index=False, header=False, sheet_name="5atab")
        table_5b.to_excel(writer, index=False, header=False, sheet_name="5btab")
        table_10a.to_excel(writer, index=False, header=False, sheet_name="10atab")
        table_10b.to_excel(writer, index=False, header=False, sheet_name="10btab")


def test_migration_to_bronze_and_silver_gold_publish(tmp_path: Path) -> None:
    source_root = tmp_path / "data_new"
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    gold_root = tmp_path / "gold"
    report_root = tmp_path / "reports"

    workbook = source_root / "steo_archives/xlsx/feb26_base.xlsx"
    _write_test_workbook(workbook)
    pdf_path = source_root / "steo_archives/pdf/feb26.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-test")
    manifest_path = source_root / "steo_archives/manifest.csv"
    manifest_path.write_text("relative_path,sha256,bytes\n", encoding="utf-8")

    context_path = source_root / "EIA/Gas2025.pdf"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_bytes(b"context")

    inventory = pd.DataFrame(
        [
            {
                "relative_path": "steo_archives/xlsx/feb26_base.xlsx",
                "integration_layer_target": "bronze",
                "integration_action": "promote_to_bronze_snapshots_and_vintage_registry",
                "priority": "P0",
                "is_duplicate": "false",
                "duplicate_of": "",
            },
            {
                "relative_path": "steo_archives/pdf/feb26.pdf",
                "integration_layer_target": "bronze",
                "integration_action": "promote_to_bronze_snapshots_and_vintage_registry",
                "priority": "P0",
                "is_duplicate": "false",
                "duplicate_of": "",
            },
            {
                "relative_path": "steo_archives/manifest.csv",
                "integration_layer_target": "gold",
                "integration_action": "ingest_manifest_for_provenance_checks",
                "priority": "P0",
                "is_duplicate": "false",
                "duplicate_of": "",
            },
            {
                "relative_path": "EIA/Gas2025.pdf",
                "integration_layer_target": "gold",
                "integration_action": "extract_macro_context_and_check_for_duplicates",
                "priority": "P1",
                "is_duplicate": "false",
                "duplicate_of": "",
            },
        ]
    )
    inventory_path = tmp_path / "inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    migration = migrate_data_new_inventory(
        inventory_path=inventory_path,
        source_root=source_root,
        bronze_root=bronze_root,
        report_root=report_root,
    )
    assert migration.migrated_count == 4
    assert migration.unresolved_count == 0

    silver = normalize_steo_vintages(
        bronze_root=bronze_root / "eia_bulk/steo_vintages",
        silver_root=silver_root,
        report_root=report_root,
    )
    assert silver["vintage_count"] == 1

    gold = publish_steo_gold_marts(
        silver_root=silver_root,
        gold_root=gold_root,
        inventory_path=inventory_path,
        bronze_context_root=bronze_root / "context",
        report_root=report_root,
    )

    assert Path(gold["paths"]["steo_observation_panel"]).exists()
    assert Path(gold["paths"]["steo_energy_prices_panel"]).exists()
    assert Path(gold["paths"]["steo_petroleum_supply_panel"]).exists()
    assert Path(gold["paths"]["steo_regional_gas_prices_panel"]).exists()
    assert Path(gold["paths"]["steo_driver_panel"]).exists()
    assert Path(gold["paths"]["steo_shale_split_panel"]).exists()
    assert Path(gold["paths"]["context_priors"]).exists()
    assert gold["row_counts"]["steo_energy_prices_panel"] > 0
    assert gold["row_counts"]["steo_petroleum_supply_panel"] > 0
    assert gold["row_counts"]["steo_observation_panel"] > 0

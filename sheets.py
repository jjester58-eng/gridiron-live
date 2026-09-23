"""Google Sheets bridge for Gridiron Live."""
from __future__ import annotations

import json
import os

import gspread
import re

import pandas as pd
from google.oauth2.service_account import Credentials

from analysis.defense import DEFENSIVE_COLUMNS
from config import (
    DEFENSE_OUTPUT_SHEET_NAME,
    DEFENSE_SOURCE_SHEET_NAME,
    GOOGLE_CREDS_ENV,
    OUTPUT_SHEET_NAME,
    REQUIRED_COLUMNS,
    SOURCE_SHEET_NAME,
    SPREADSHEET_ID_ENV,
)


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def get_spreadsheet():
    raw_creds = os.environ.get(GOOGLE_CREDS_ENV)
    spreadsheet_id = os.environ.get(SPREADSHEET_ID_ENV)

    if not raw_creds:
        raise RuntimeError(
            f"Missing GitHub secret/environment variable: {GOOGLE_CREDS_ENV}"
        )
    if not spreadsheet_id:
        raise RuntimeError(
            f"Missing GitHub secret/environment variable: {SPREADSHEET_ID_ENV}"
        )

    credentials = Credentials.from_service_account_info(
        json.loads(raw_creds),
        scopes=SCOPES,
    )
    return gspread.authorize(credentials).open_by_key(spreadsheet_id.strip())


def load_source_df(spreadsheet):
    worksheet = spreadsheet.worksheet(SOURCE_SHEET_NAME)
    values = worksheet.get_all_values()

    if not values:
        raise ValueError(f"'{SOURCE_SHEET_NAME}' is empty.")

    headers = [str(h).strip() for h in values[0]]
    rows = [(row + [""] * len(headers))[: len(headers)] for row in values[1:]]

    df = pd.DataFrame(rows, columns=headers)
    df = df.loc[:, [c != "" for c in df.columns]]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in '{SOURCE_SHEET_NAME}': {missing}"
        )

    return df[REQUIRED_COLUMNS].copy()


def load_defense_source_df(spreadsheet):
    """Load WHS DATA and keep only our defensive snaps (ODK=D)."""
    worksheet = spreadsheet.worksheet(DEFENSE_SOURCE_SHEET_NAME)
    values = worksheet.get_all_values()

    if not values:
        raise ValueError(f"'{DEFENSE_SOURCE_SHEET_NAME}' is empty.")

    headers = [str(h).strip() for h in values[0]]
    rows = [(row + [""] * len(headers))[: len(headers)] for row in values[1:]]

    df = pd.DataFrame(rows, columns=headers)
    df = df.loc[:, [c != "" for c in df.columns]]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    missing = [c for c in DEFENSIVE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required defensive columns in '{DEFENSE_SOURCE_SHEET_NAME}': {missing}"
        )

    # WHS DATA contains offense, defense, and kicking snaps.
    # For this report, D means our defense. K is intentionally left for later.
    odk = df["ODK"].astype(str).str.strip().str.upper()
    defense = df.loc[odk == "D", DEFENSIVE_COLUMNS].copy()

    return defense.reset_index(drop=True)


def dataframe_values(df):
    if df is None:
        return []

    if df.empty:
        return [list(df.columns)]

    clean = df.copy().fillna("")
    return [list(clean.columns)] + clean.astype(str).values.tolist()


def _normalize_like_term(value):
    """Normalize a play/formation name for related-term matching."""
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _summarize_like_terms(df):
    """Summarize total pass targets by player number."""
    columns = ["TARGET", "COUNT"]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy().fillna("")
    if "TARGET" not in work.columns or "COUNT" not in work.columns:
        return pd.DataFrame(columns=columns)

    work["TARGET"] = work["TARGET"].astype(str).str.strip()
    work["COUNT"] = pd.to_numeric(work["COUNT"], errors="coerce").fillna(0)

    summary = (
        work[work["TARGET"] != ""]
        .groupby("TARGET", dropna=False)
        .agg(COUNT=("COUNT", "sum"))
        .reset_index()
    )

    return summary.sort_values(
        ["COUNT", "TARGET"],
        ascending=[False, True],
    ).reset_index(drop=True)


def write_report(spreadsheet, report):
    existing = {w.title for w in spreadsheet.worksheets()}

    worksheet = (
        spreadsheet.worksheet(OUTPUT_SHEET_NAME)
        if OUTPUT_SHEET_NAME in existing
        else spreadsheet.add_worksheet(
            title=OUTPUT_SHEET_NAME,
            rows=500,
            cols=20,
        )
    )

    worksheet.clear()

    summary = report.summary()
    top_line = [
        "Team Identity",
        f"RUN ATT: {summary['RUN ATT']}",
        f"RUN YDS: {summary['RUN YDS']}",
        f"PASS ATT / COMP: {summary['PASS ATT']} / {summary['PASS COMP']}",
        f"PASS YDS: {summary['PASS YDS']}",
        f"SACK: {summary['SACK']}",
        f"INT: {summary['INT']}",
        f"FUMBLES: {summary['FUMBLES']}",
        f"TD: {summary['TD']}",
        f"RUSH TD: {summary['RUSH TD']}",
        f"PASS TD: {summary['PASS TD']}",
    ]
    worksheet.update([top_line], "A1")

    sections = [
        ("DOWN EFFICIENCY", report.third_down_efficiency),
        ("SITUATION → PLAY CALLING PATTERNS", report.situation_play_calling),
        ("3-AND-OUTS", report.three_and_out_analysis),
    ]

    row = 3
    for title, section in sections:
        worksheet.update([[title]], f"A{row}")
        row += 1
        values = dataframe_values(section)
        if values:
            worksheet.update(values, f"A{row}")
            row += len(values) + 2
        else:
            worksheet.update([["No data"]], f"A{row}")
            row += 3

    worksheet.update([["PASSING TENDENCIES"]], f"A{row}")
    worksheet.update([["PASSING TENDENCIES"]], f"G{row}")

    completion_values = dataframe_values(report.completed_comment_patterns)
    summary_values = dataframe_values(
        _summarize_like_terms(report.completed_comment_patterns)
    )

    if completion_values:
        worksheet.update(completion_values, f"A{row + 1}")
    if summary_values:
        worksheet.update(summary_values, f"G{row + 1}")

    if completion_values or summary_values:
        detail_rows = len(completion_values) if completion_values else 0
        summary_rows = len(summary_values) if summary_values else 0
        row += max(detail_rows, summary_rows) + 4
    else:
        row += 3

    sections = [
        ("PLAY EFFICIENCY — FIELD ZONES", report.field_zone_efficiency),
        ("FIELD ZONE BY HASH", report.field_zone_by_hash),
        ("SITUATION → NEXT PLAY SEQUENCES", report.situation_sequences),
        ("HIGH-FREQUENCY — SCHEME + YARDS", report.frequency_by_scheme),
        ("RUN / PASS PATTERNS", report.run_pass_sequences),
        ("LEFT / RIGHT PATTERNS", report.left_right_sequences),
        ("HASH + DOWN/DISTANCE PLAY PROBABILITIES", report.hash_down_distance_play_probabilities),
        ("REPEATED PLAY SEQUENCES", report.repeated_sequences),
        ("REPEATED PLAY FOLLOW-UPS", report.repeated_followups),
        ("HIGH-FREQUENCY — SITUATIONS + YARDS", report.frequency_by_situation),
    ]
    for title, section in sections:
        worksheet.update([[title]], f"A{row}")
        row += 1

        values = dataframe_values(section)
        if values:
            worksheet.update(values, f"A{row}")
            row += len(values) + 2
        else:
            worksheet.update([["No data"]], f"A{row}")
            row += 3

    return worksheet


def write_defense_sheet(spreadsheet, report):
    """Write the defensive self-scout report to DEF SELF SCOUT.

    Build the entire report in memory and send it in one Sheets values update.
    This avoids the per-section write requests that can hit the Sheets API
    per-user write quota during GitHub Actions runs.
    """
    existing = {w.title for w in spreadsheet.worksheets()}

    worksheet = (
        spreadsheet.worksheet(DEFENSE_OUTPUT_SHEET_NAME)
        if DEFENSE_OUTPUT_SHEET_NAME in existing
        else spreadsheet.add_worksheet(
            title=DEFENSE_OUTPUT_SHEET_NAME,
            rows=1000,
            cols=30,
        )
    )

    overall = report.overall.iloc[0].to_dict() if not report.overall.empty else {}
    summary = [
        "DEF SELF SCOUT",
        f"PLAYS: {overall.get('PLAYS', 0)}",
        f"AVG YDS: {overall.get('AVG YDS', '')}",
        f"RUN: {overall.get('RUN', 0)}",
        f"PASS: {overall.get('PASS', 0)}",
        f"TFL: {overall.get('TFL', 0)} ({overall.get('TFL %', 0)}%)",
        f"HAVOC RATE: {overall.get('HAVOC %', 0)}%",
        f"SACK: {overall.get('SACK', 0)} ({overall.get('SACK %', 0)}%)",
        f"TURNOVERS: {overall.get('TURNOVERS', 0)} ({overall.get('TURNOVER %', 0)}%)",
        f"EXPLOSIVES: {overall.get('EXPLOSIVES', 0)} ({overall.get('EXPLOSIVE %', 0)}%)",
        f"TD: {overall.get('TD', 0)} ({overall.get('TD %', 0)}%)",
        f"INCOMPLETIONS: {overall.get('INCOMPLETIONS', 0)}",
    ]

    sections = [
        ("DEFENSIVE CALL → RESULT", report.by_def_call),
        ("SITUATION → DEFENSIVE CALL", report.situation_calls),
        ("SITUATION → CALL → RESULT", report.call_by_situation),
        ("OFFENSIVE FORMATION → RESULT", report.by_off_form),
        ("RUN / PASS → RESULT", report.by_play_type),
        ("EXPLOSIVE PLAYS", report.explosive_context),
        ("RESULTS — 5 YARDS OR LESS / INCOMPLETE / INTERCEPTION / FUMBLE", report.result_stops),
        ("MEASURABLE STRENGTHS / IMPROVEMENT INDICATORS", report.indicators),
    ]

    # Assemble the whole sheet as a single rectangular grid.
    grid = [summary, []]
    for title, section in sections:
        grid.append([title])
        values = dataframe_values(section)
        if values:
            grid.extend(values)
            grid.append([])
            grid.append([])
        else:
            grid.append(["No data"])
            grid.append([])
            grid.append([])

    # Google Sheets requires a rectangular value matrix.
    width = max(len(row) for row in grid) if grid else 1
    grid = [row + [""] * (width - len(row)) for row in grid]

    worksheet.clear()
    worksheet.update(grid, "A1", raw=True)

    return worksheet

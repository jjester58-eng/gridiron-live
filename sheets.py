"""Google Sheets bridge for Gridiron Live."""
from __future__ import annotations

import json
import os
import time

import gspread
import re

import pandas as pd
from google.oauth2.service_account import Credentials

from analysis.defense import DEFENSIVE_COLUMNS, analyze_matchup
from config import (
    DEFENSE_OUTPUT_SHEET_NAME,
    DEFENSE_SOURCE_SHEET_NAME,
    OFFENSE_OUTPUT_SHEET_NAME,
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

    # Only these fields are fundamental to the defensive self-scout.
    # Some WHS DATA versions do not include optional scouting fields such as
    # RPO, PERSONNEL, DEF FRONT, DEF STUNT, COVERAGE, BLITZ, or COMMENTS.
    # Add missing optional fields as blanks so the analyzer can still run.
    required_for_defense = [
        "PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN",
        "PLAY TYPE", "RESULT", "GN/LS", "OFF FORM", "MOTION",
        "OFF PLAY", "DEF CALL",
    ]
    missing = [c for c in required_for_defense if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required defensive columns in '{DEFENSE_SOURCE_SHEET_NAME}': {missing}"
        )

    for column in DEFENSIVE_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # WHS DATA contains offense, defense, and kicking snaps.
    # For this report, D means our defense. K is intentionally left for later.
    odk = df["ODK"].astype(str).str.strip().str.upper()
    defense = df.loc[odk == "D", DEFENSIVE_COLUMNS].copy()

    return defense.reset_index(drop=True)


def load_offense_source_df(spreadsheet):
    """Load WHS DATA and keep only our offensive snaps (ODK=O)."""
    worksheet = spreadsheet.worksheet(DEFENSE_SOURCE_SHEET_NAME)
    values = worksheet.get_all_values()

    if not values:
        raise ValueError(f"'{DEFENSE_SOURCE_SHEET_NAME}' is empty.")

    headers = [str(h).strip() for h in values[0]]
    rows = [(row + [""] * len(headers))[: len(headers)] for row in values[1:]]

    df = pd.DataFrame(rows, columns=headers)
    df = df.loc[:, [c != "" for c in df.columns]]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    # Offense uses BALL CARRIER for the actual player who ran the ball
    # or caught the pass. Some older WHS DATA columns are not used by the
    # offense self-scout, so add them as blanks rather than requiring them.
    # The offense sheet has used both COMMENT and COMMENTS over time.
    # Treat them as the same field so the self-scout is tolerant of either.
    if "COMMENTS" not in df.columns and "COMMENT" in df.columns:
        df["COMMENTS"] = df["COMMENT"]
    required_for_offense = [
        "PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN", "PLAY TYPE",
        "RESULT", "GN/LS", "OFF FORM", "OFF PLAY", "MOTION", "PLAY DIR",
        "BALL CARRIER", "DEF CALL", "COMMENTS",
    ]
    missing = [c for c in required_for_offense if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required offensive columns in '{DEFENSE_SOURCE_SHEET_NAME}': {missing}"
        )

    # analyze() still expects the shared opponent schema. Keep the offensive
    # fields plus BALL CARRIER, and supply unused legacy columns as blanks.
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    odk = df["ODK"].astype(str).str.strip().str.upper()
    offense = df.loc[odk == "O", REQUIRED_COLUMNS + ["BALL CARRIER"]].copy()
    return offense.reset_index(drop=True)

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


def write_report(spreadsheet, report, matchup_report=None, formation_call_report=None):
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

    completion_values = dataframe_values(report.completed_comment_patterns)
    summary_values = dataframe_values(report.passing_target_summary)

    if completion_values:
        worksheet.update(completion_values, f"A{row + 1}")
    if summary_values:
        worksheet.update(summary_values, f"G{row + 1}")

    # The right-side TARGET / COUNT / YARDS table is the same target
    # summary used by OFF SELF SCOUT, now including receiving yards.
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
        ("RUN / PASS PATTERNS", report.run_pass_sequences),
        ("LEFT / RIGHT PATTERNS", report.left_right_sequences),
        ("HASH + DOWN/DISTANCE PLAY PROBABILITIES", report.hash_down_distance_play_probabilities),
        ("REPEATED PLAY SEQUENCES", report.repeated_sequences),
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

    # Bottom formation/call chart. Formations and run/pass mix come from
    # ALL INFO SHEET; DEF CALL comes from WHS DATA rows where ODK == D.
    if formation_call_report is not None:
        worksheet.update([["FORMATION × WHS DEFENSIVE CALL — SITUATIONS"]], f"A{row}")
        row += 1
        values = dataframe_values(formation_call_report)
        if values:
            worksheet.update(values, f"A{row}")
            row += len(values) + 2
        else:
            worksheet.update([["No matching WHS defensive data"]], f"A{row}")
            row += 3


    return worksheet


def write_offense_self_scout(spreadsheet, report):
    """Write the factual offensive identity report to OFF SELF SCOUT."""
    existing = {w.title for w in spreadsheet.worksheets()}
    worksheet = (
        spreadsheet.worksheet(OFFENSE_OUTPUT_SHEET_NAME)
        if OFFENSE_OUTPUT_SHEET_NAME in existing
        else spreadsheet.add_worksheet(
            title=OFFENSE_OUTPUT_SHEET_NAME, rows=1000, cols=30
        )
    )

    summary = report.summary()
    top_line = [
        "OFF SELF SCOUT",
        f"TOTAL PLAYS: {summary.get('TOTAL PLAYS', 0)}",
        f"RUN ATT: {summary.get('RUN ATT', 0)}",
        f"RUN YDS: {summary.get('RUN YDS', 0)}",
        f"PASS ATT / COMP: {summary.get('PASS ATT', 0)} / {summary.get('PASS COMP', 0)}",
        f"PASS YDS: {summary.get('PASS YDS', 0)}",
        f"TD: {summary.get('TD', 0)}",
        f"RUSH TD: {summary.get('RUSH TD', 0)}",
        f"PASS TD: {summary.get('PASS TD', 0)}",
        f"FUMBLES: {summary.get('FUMBLES', 0)}",
        f"INT: {summary.get('INT', 0)}",
        f"EXPLOSIVE PLAYS: {summary.get('EXPLOSIVE PLAYS', 0)} ({summary.get('EXPLOSIVE RATE', 0)}%)",
    ]

    sections = [
        ("DOWN EFFICIENCY", report.third_down_efficiency),
        ("SITUATION → PLAY CALLING PATTERNS", report.situation_play_calling),
        ("3-AND-OUTS", report.three_and_out_analysis),
        ("PLAY EFFICIENCY — FIELD ZONES", report.field_zone_efficiency),
        ("FIELD ZONE BY HASH", report.field_zone_by_hash),
        ("HIGH-FREQUENCY — SCHEME + YARDS", report.frequency_by_scheme),
        ("RUN / PASS PATTERNS", report.run_pass_sequences),
        ("REPEATED PLAY SEQUENCES", report.repeated_sequences),
        ("HIGH-FREQUENCY — SITUATIONS + YARDS", report.frequency_by_situation),
        ("HIGH-FREQUENCY — FORMATIONS + YARDS", report.frequency_by_formation),
        ("HIGH-FREQUENCY — MOTION + YARDS", report.frequency_by_motion),
        ("EXPLOSIVES — FORMATION + SITUATION", report.explosive_by_formation_situation),
    ]

    grid = [top_line, []]

    # Keep all three offensive identity tables on the same row block A:I:
    # left = PLAY / FORMATION / blank / COUNT, middle = TARGET / COUNT / YARDS,
    # right = BALL CARRIER / COUNT / YARDS.
    grid.append([
        "PASSING TENDENCIES", "", "", "", "PASSING TENDENCIES", "", "",
        "RUSHING TENDENCIES", "", ""
    ])

    play_formation_values = dataframe_values(report.passing_play_formation_counts)
    target_summary_values = dataframe_values(report.passing_target_summary)
    rushing_values = dataframe_values(report.rushing_tendencies)

    # Normalize the left table to PLAY / FORMATION / blank / COUNT
    # so the three blocks line up cleanly across A:I.
    def normalize_play_formation(rows):
        if not rows:
            return []
        output = [["PLAY", "FORMATION", "", "COUNT", ""]]
        for row in rows[1:]:
            output.append([
                row[0] if len(row) > 0 else "",
                row[1] if len(row) > 1 else "",
                "",
                row[3] if len(row) > 3 else (row[2] if len(row) > 2 else ""),
            ])
        return output

    left_values = normalize_play_formation(play_formation_values)
    middle_values = target_summary_values
    right_values = rushing_values

    block_height = max(len(left_values), len(middle_values), len(right_values), 1)
    for i in range(block_height):
        left = left_values[i] if i < len(left_values) else ["", "", "", "", ""]
        middle = middle_values[i] if i < len(middle_values) else ["", "", ""]
        right = right_values[i] if i < len(right_values) else ["", "", ""]
        grid.append(left + middle + right)

    grid.extend([[], []])

    for title, section in sections:
        grid.append([title])
        values = dataframe_values(section)
        if values:
            grid.extend(values)
            grid.extend([[], []])
        else:
            grid.extend([["No data"], [], []])

    width = max(len(row) for row in grid) if grid else 1
    width = max(width, worksheet.col_count)
    row_count = max(len(grid), worksheet.row_count)
    grid = [row + [""] * (width - len(row)) for row in grid]
    grid.extend([[""] * width for _ in range(row_count - len(grid))])

    last_error = None
    for attempt in range(5):
        try:
            worksheet.update(grid, "A1", raw=True)
            return worksheet
        except gspread.exceptions.APIError as exc:
            last_error = exc
            if "429" not in str(exc):
                raise
            if attempt == 4:
                raise
            time.sleep(5 * (2 ** attempt))

    if last_error:
        raise last_error
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

    # Google Sheets requires a rectangular value matrix. Pad the report to
    # the existing worksheet size so old content is overwritten with blanks
    # without a separate clear() write request.
    width = max(len(row) for row in grid) if grid else 1
    width = max(width, worksheet.col_count)
    row_count = max(len(grid), worksheet.row_count)
    grid = [row + [""] * (width - len(row)) for row in grid]
    grid.extend([[""] * width for _ in range(row_count - len(grid))])

    # GitHub Actions runs can overlap after several commits. Sheets enforces a
    # per-user write quota, so retry 429s with exponential backoff.
    last_error = None
    for attempt in range(5):
        try:
            worksheet.update(grid, "A1", raw=True)
            return worksheet
        except gspread.exceptions.APIError as exc:
            last_error = exc
            if "429" not in str(exc):
                raise
            if attempt == 4:
                raise
            time.sleep(5 * (2 ** attempt))

    if last_error:
        raise last_error
    return worksheet

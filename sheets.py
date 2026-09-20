"""Google Sheets bridge for Gridiron Live."""
from __future__ import annotations

import json
import os

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

from config import (
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


def dataframe_values(df):
    if df is None:
        return []

    if df.empty:
        return [list(df.columns)]

    clean = df.copy().fillna("")
    return [list(clean.columns)] + clean.astype(str).values.tolist()


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

    # Keep the Google Sheet writer in lockstep with TendencyReport.
    # Explosive formation + situation and hash + direction are intentionally
    # combined sections. EXPLOSIVES BY PERSONNEL and DOWN & DISTANCE — ALL
    # PLAYS were removed from the report.
    sections = [
        ("EXPLOSIVE PLAYS", report.explosive_detail),
        ("FORMATIONS", report.formation_frequency),
        (
            "EXPLOSIVES BY FORMATION + SITUATION",
            report.explosive_by_formation_situation,
        ),
        (
            "EXPLOSIVES BY HASH + DIRECTION",
            report.explosive_by_hash_direction,
        ),
        ("EXPLOSIVES BY MOTION", report.by_motion),
        ("REPEATED PLAY SEQUENCES", report.repeated_sequences),
        ("REPEATED PLAY FOLLOW-UPS", report.repeated_followups),
        ("COMPLETIONS + COMMENTS", report.completed_comment_patterns),
        (
            "HIGH-FREQUENCY — SITUATIONS + YARDS",
            report.frequency_by_situation,
        ),
        (
            "HIGH-FREQUENCY — PERSONNEL + YARDS",
            report.frequency_by_personnel,
        ),
        (
            "HIGH-FREQUENCY — BACKFIELD + YARDS",
            report.frequency_by_backfield,
        ),
        (
            "HIGH-FREQUENCY — MOTION + YARDS",
            report.frequency_by_motion,
        ),
        (
            "HIGH-FREQUENCY — SCHEME + YARDS",
            report.frequency_by_scheme,
        ),
        (
            "HIGH-FREQUENCY — DIRECTION + YARDS",
            report.frequency_by_direction,
        ),
        ("HIGH-FREQUENCY — PLAY TYPES", report.play_type_frequency),
        ("RUN / PASS / QB RUN / SACK YARDS", report.run_pass_yards),
        (
            "SITUATION → NEXT PLAY SEQUENCES",
            report.situation_sequences,
        ),
    ]

    row = 1
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

    worksheet.freeze(rows=1)
    return worksheet

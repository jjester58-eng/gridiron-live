"""Flexible pairwise analysis of components found in full defensive calls.

The raw DEF CALL remains untouched. Terms are extracted for analysis only;
no terms are treated as synonyms or merged into aliases.
"""
from __future__ import annotations

import re
from itertools import combinations

import pandas as pd

from analysis.defense import (
    DEFENSIVE_COLUMNS,
    clean,
    is_explosive,
    is_sack,
    is_tfl,
    is_touchdown,
    is_turnover,
    play_type,
    result_yards,
)

# Preserve recognizable multi-word concepts as one term. Slash-separated
# entries are intentionally split so variants such as ROBBER and THIEF stay
# distinct instead of being treated as synonyms.
MULTIWORD_TERMS = (
    "JUMP TO ROBBER",
    "JUMP TO BEAR",
    "DOUBLE ROBBER",
    "DBL ROBBER",
    "STUFF IT",
    "BLUFF IT",
    "FIELD BDRY",
    "FIELD BOUNDARY",
    "FIELD",
    "BOUNDARY",
    "PIRATE",
    "EARS",
    "ROBBER",
    "THIEF",
    "STORM",
    "THUNDER",
    "APACHE",
    "BRAVE",
    "LOCK",
)


def _call_terms(value):
    text = clean(value).upper()
    if not text:
        return []
    # A slash in the notes separates alternatives/terms; never normalize them
    # into one shared label.
    text = text.replace("/", " ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    phrases = sorted(MULTIWORD_TERMS, key=len, reverse=True)
    found = []
    remaining = f" {text} "
    for phrase in phrases:
        phrase_norm = re.sub(r"[^A-Z0-9]+", " ", phrase).strip()
        pattern = rf"(?<![A-Z0-9]){re.escape(phrase_norm)}(?![A-Z0-9])"
        if re.search(pattern, remaining):
            found.append(phrase_norm)
            remaining = re.sub(pattern, " ", remaining)
    found.extend(re.findall(r"[A-Z0-9]+", remaining))
    # "22" is a meaningful call component and is retained.
    return list(dict.fromkeys(term for term in found if term))


def _row_terms(row):
    """Collect unique terms from the full call and any populated component fields."""
    terms = _call_terms(row.get("DEF CALL", ""))
    for column in ("DEF FRONT", "DEF STUNT", "COVERAGE", "BLITZ"):
        value = clean(row.get(column, ""))
        if value:
            # Structured fields are kept as their complete value; call terms
            # are added too, allowing pairings across entry styles.
            terms.append(re.sub(r"\s+", " ", value.upper()).strip())
    return sorted(set(term for term in terms if term))


def defensive_call_pairings(df, min_occurrences=2):
    """Return term-pair usage and defensive outcomes across full-call entries."""
    columns = [
        "TERM 1", "TERM 2", "PAIR PLAYS", "AVG YDS ALLOWED",
        "TFL", "SACK", "TURNOVERS", "EXPLOSIVES", "TD",
        "TFL %", "SACK %", "TURNOVER %", "EXPLOSIVE %", "TD %",
        "RUN PLAYS", "PASS PLAYS",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    for col in DEFENSIVE_COLUMNS:
        if col not in work.columns:
            work[col] = ""
    work = work[work["PLAY #"].map(clean) != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    work["_YARDS"] = work.apply(result_yards, axis=1)
    work["_TFL"] = work.apply(is_tfl, axis=1)
    work["_SACK"] = work.apply(is_sack, axis=1)
    work["_TURNOVER"] = work.apply(is_turnover, axis=1)
    work["_EXPLOSIVE"] = work.apply(is_explosive, axis=1)
    work["_TD"] = work.apply(is_touchdown, axis=1)
    work["_PLAY_TYPE"] = work.apply(play_type, axis=1)
    work["_TERMS"] = work.apply(_row_terms, axis=1)

    pair_rows = []
    for _, row in work.iterrows():
        terms = row["_TERMS"]
        for first, second in combinations(terms, 2):
            pair_rows.append({
                "_PAIR": (first, second),
                "_YARDS": row["_YARDS"],
                "_TFL": bool(row["_TFL"]),
                "_SACK": bool(row["_SACK"]),
                "_TURNOVER": bool(row["_TURNOVER"]),
                "_EXPLOSIVE": bool(row["_EXPLOSIVE"]),
                "_TD": bool(row["_TD"]),
                "_PLAY_TYPE": row["_PLAY_TYPE"],
            })

    if not pair_rows:
        return pd.DataFrame(columns=columns)

    pairs = pd.DataFrame(pair_rows)
    rows = []
    for (term1, term2), group in pairs.groupby("_PAIR", sort=True):
        plays = len(group)
        if plays < min_occurrences:
            continue
        yards = pd.to_numeric(group["_YARDS"], errors="coerce")
        rows.append({
            "TERM 1": term1,
            "TERM 2": term2,
            "PAIR PLAYS": plays,
            "AVG YDS ALLOWED": round(yards.mean(), 1) if yards.notna().any() else "",
            "TFL": int(group["_TFL"].sum()),
            "SACK": int(group["_SACK"].sum()),
            "TURNOVERS": int(group["_TURNOVER"].sum()),
            "EXPLOSIVES": int(group["_EXPLOSIVE"].sum()),
            "TD": int(group["_TD"].sum()),
            "TFL %": round(group["_TFL"].mean() * 100, 1),
            "SACK %": round(group["_SACK"].mean() * 100, 1),
            "TURNOVER %": round(group["_TURNOVER"].mean() * 100, 1),
            "EXPLOSIVE %": round(group["_EXPLOSIVE"].mean() * 100, 1),
            "TD %": round(group["_TD"].mean() * 100, 1),
            "RUN PLAYS": int((group["_PLAY_TYPE"] == "RUN").sum()),
            "PASS PLAYS": int((group["_PLAY_TYPE"] == "PASS").sum()),
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["PAIR PLAYS", "TERM 1", "TERM 2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def write_defensive_call_pairings(spreadsheet, defense_df, min_occurrences=2):
    """Write the pairings report to a dedicated tab without altering DEF CALL."""
    report = defensive_call_pairings(defense_df, min_occurrences=min_occurrences)
    title = "DEF CALL PAIRINGS"
    existing = {w.title for w in spreadsheet.worksheets()}
    worksheet = (
        spreadsheet.worksheet(title)
        if title in existing
        else spreadsheet.add_worksheet(title=title, rows=1000, cols=20)
    )

    grid = [
        ["DEFENSIVE CALL PAIRINGS"],
        ["Pairing is based on terms found within full DEF CALL entries and populated component fields."],
        ["Terms are not treated as synonyms. Raw DEF CALL values are not changed."],
        ["Minimum sample", min_occurrences],
    ]
    values = [list(report.columns)] + report.fillna("").astype(str).values.tolist()
    grid.extend(values if len(report) else [["No pairings met the minimum sample threshold"]])
    width = max(max(len(row) for row in grid), worksheet.col_count)
    row_count = max(len(grid), worksheet.row_count)
    grid = [row + [""] * (width - len(row)) for row in grid]
    grid.extend([[""] * width for _ in range(row_count - len(grid))])
    worksheet.update(grid, "A1", raw=True)
    return report

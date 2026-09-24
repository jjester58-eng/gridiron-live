"""
Gridiron Live defensive self-scout analyzer.

Descriptive only: measures what the defense did with supplied snaps and
surfaces measurable strength/improvement indicators without predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

DEFENSIVE_COLUMNS = [
    "PLAY #", "ODK", "DN", "DIST", "HASH", "YARD LN", "RPO",
    "PLAY TYPE", "RESULT", "GN/LS", "PERSONNEL", "OFF FORM", "MOTION",
    "OFF PLAY", "DEF CALL", "DEF FRONT", "DEF STUNT", "COVERAGE",
    "BLITZ", "COMMENTS",
]

SITUATION_ORDER = {
    "1st & Long (10+)": 1, "1st & Short (1-9)": 2,
    "2nd & Long (7+)": 3, "2nd & Medium (4-6)": 4,
    "2nd & Short (1-3)": 5, "3rd & Long (7+)": 6,
    "3rd & Medium (4-6)": 7, "3rd & Short (1-3)": 8, "4th Down": 9,
}


def clean(value):
    return "" if pd.isna(value) else str(value).strip()


def numeric(value):
    try:
        if pd.isna(value) or clean(value) == "":
            return None
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def result_blob(row):
    return " ".join(
        clean(row.get(c, "")) for c in
        ("PLAY TYPE", "RESULT", "OFF PLAY", "COMMENTS")
    ).upper()


def result_yards(row):
    yards = numeric(row.get("GN/LS"))
    if yards is not None:
        return yards
    match = re.search(
        r"(?<!\d)(-?\d+(?:\.\d+)?)\s*(?:YDS?|YARDS?)?(?!\d)",
        clean(row.get("RESULT", "")).upper(),
    )
    return float(match.group(1)) if match else None


def play_type(row):
    text = result_blob(row)
    if any(x in text for x in (
        "INTERCEPTION", "INTERCEPTED", "SACK", "INCOMPLETE", "COMPLETE",
        "PASS", "SCREEN", "RPO", "QB THROW",
    )):
        return "PASS"
    if any(x in text for x in (
        "RUN", "RUSH", "DRAW", "QB SNEAK", "KEEP", "SCRAMBLE",
    )):
        return "RUN"
    return "OTHER"


def is_explosive(row, run_yards=15, pass_yards=20):
    yards = result_yards(row)
    if yards is None:
        return False
    category = play_type(row)
    return (
        category == "PASS" and yards >= pass_yards
    ) or (
        category == "RUN" and yards >= run_yards
    )


def is_sack(row):
    return "SACK" in result_blob(row)


def is_interception(row):
    return bool(re.search(r"\b(INTERCEPTION|INTERCEPTED|INT)\b", result_blob(row)))


def is_fumble(row):
    return bool(re.search(r"\b(FUMBLE|FUMBLED)\b", result_blob(row)))


def is_touchdown(row):
    return bool(re.search(r"\b(TD|TOUCHDOWN)\b", result_blob(row)))


def is_incomplete(row):
    return bool(re.search(r"\bINCOMPLETE\b", result_blob(row)))


def is_tfl(row):
    yards = result_yards(row)
    return (
        (yards is not None and yards < 0)
        or bool(re.search(r"\b(TFL|TACKLE FOR LOSS|LOSS)\b", result_blob(row)))
    )


def is_turnover(row):
    return is_interception(row) or is_fumble(row)


def normalize_blitz(value):
    text = clean(value).upper()
    if not text:
        return ""
    if text in {"Y", "YES", "TRUE", "1", "BLITZ"}:
        return "BLITZ"
    if text in {"N", "NO", "FALSE", "0", "NONE", "NO BLITZ"}:
        return "NO BLITZ"
    return clean(value)


def _field_zone(value):
    """Map YARD LN to the same field zones used by Tendencies."""
    yard = numeric(value)
    if yard is None:
        return ""
    yard = abs(yard) if yard < 0 else yard
    raw = numeric(value)
    if raw < 0:
        if yard <= 20:
            return "OWN 1-20"
        if yard <= 40:
            return "OWN 21-40"
        return "OWN 41-49"
    if raw == 50 or raw >= 41:
        return "MIDFIELD"
    if raw >= 21:
        return "OPP 21-40"
    if raw >= 1:
        return "RED ZONE"
    return "MIDFIELD"


def situation_bucket(down, distance):
    down_value = numeric(down)
    distance_value = numeric(distance)
    if down_value is None or distance_value is None:
        return ""
    down_value = int(down_value)
    distance_value = max(0, distance_value)
    if down_value == 1:
        return "1st & Long (10+)" if distance_value >= 10 else "1st & Short (1-9)"
    if down_value == 2:
        if distance_value >= 7: return "2nd & Long (7+)"
        if distance_value >= 4: return "2nd & Medium (4-6)"
        return "2nd & Short (1-3)"
    if down_value == 3:
        if distance_value >= 7: return "3rd & Long (7+)"
        if distance_value >= 4: return "3rd & Medium (4-6)"
        return "3rd & Short (1-3)"
    if down_value == 4:
        return "4th Down"
    return ""


def _group_metrics(df, group_column, min_occurrences=5):
    columns = [
        group_column, "PLAYS", "AVG YDS", "TFL", "TFL %",
        "SACK", "SACK %", "TURNOVERS", "TURNOVER %",
        "EXPLOSIVES", "EXPLOSIVE %", "TD", "TD %",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work[group_column] = work[group_column].map(lambda x: clean(x) or "UNKNOWN")
    rows = []
    for value, group in work.groupby(group_column):
        if len(group) < min_occurrences:
            continue
        yards = pd.to_numeric(group["_YARDS"], errors="coerce")
        plays = len(group)
        rows.append({
            group_column: value,
            "PLAYS": plays,
            "AVG YDS": round(yards.mean(), 1) if yards.notna().any() else "",
            "TFL": int(group["_TFL"].sum()),
            "TFL %": round(group["_TFL"].mean() * 100, 1),
            "SACK": int(group["_SACK"].sum()),
            "SACK %": round(group["_SACK"].mean() * 100, 1),
            "TURNOVERS": int(group["_TURNOVER"].sum()),
            "TURNOVER %": round(group["_TURNOVER"].mean() * 100, 1),
            "EXPLOSIVES": int(group["_EXPLOSIVE"].sum()),
            "EXPLOSIVE %": round(group["_EXPLOSIVE"].mean() * 100, 1),
            "TD": int(group["_TD"].sum()),
            "TD %": round(group["_TD"].mean() * 100, 1),
        })
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["PLAYS", group_column], ascending=[False, True])
        .reset_index(drop=True)
        if rows else pd.DataFrame(columns=columns)
    )


def _call_result(df, group_column, min_occurrences=5):
    table = _group_metrics(df, group_column, min_occurrences)
    cols = [group_column, "PLAYS", "AVG YDS", "TFL", "SACK",
            "TURNOVERS", "EXPLOSIVES", "TD"]
    return table[cols] if not table.empty else pd.DataFrame(columns=cols)


def _situation_calls(df, min_occurrences=5, top_n=3):
    columns = ["DOWN & DISTANCE", "PLAYS", "CALL 1", "CALL 2", "CALL 3"]
    rows = []
    for situation, group in df[df["SITUATION"] != ""].groupby("SITUATION"):
        if len(group) < min_occurrences:
            continue
        calls = (
            group[group["DEF CALL"].map(clean) != ""]
            .groupby("DEF CALL").size()
            .sort_values(ascending=False).head(top_n)
        )
        values = [
            f"{call} ({count}/{len(group)}, {round(count / len(group) * 100, 1)}%)"
            for call, count in calls.items()
        ] + [""] * top_n
        rows.append({
            "DOWN & DISTANCE": situation, "PLAYS": len(group),
            "CALL 1": values[0], "CALL 2": values[1], "CALL 3": values[2],
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["_ORDER"] = out["DOWN & DISTANCE"].map(SITUATION_ORDER).fillna(99)
    return out.sort_values("_ORDER").drop(columns="_ORDER").reset_index(drop=True)


def _call_by_situation(df, min_occurrences=5):
    columns = ["DOWN & DISTANCE", "DEF CALL", "PLAYS", "AVG YDS",
               "TFL", "SACK", "TURNOVERS", "EXPLOSIVES", "TD"]
    work = df[(df["SITUATION"] != "") & (df["DEF CALL"].map(clean) != "")].copy()
    rows = []
    for (situation, call), group in work.groupby(["SITUATION", "DEF CALL"]):
        if len(group) < min_occurrences:
            continue
        yards = pd.to_numeric(group["_YARDS"], errors="coerce")
        rows.append({
            "DOWN & DISTANCE": situation, "DEF CALL": call, "PLAYS": len(group),
            "AVG YDS": round(yards.mean(), 1) if yards.notna().any() else "",
            "TFL": int(group["_TFL"].sum()), "SACK": int(group["_SACK"].sum()),
            "TURNOVERS": int(group["_TURNOVER"].sum()),
            "EXPLOSIVES": int(group["_EXPLOSIVE"].sum()),
            "TD": int(group["_TD"].sum()),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["_ORDER"] = out["DOWN & DISTANCE"].map(SITUATION_ORDER).fillna(99)
    return out.sort_values(
        ["_ORDER", "PLAYS", "DEF CALL"], ascending=[True, False, True]
    ).drop(columns="_ORDER").reset_index(drop=True)


def _explosive_context(df):
    columns = [
        "PLAY #", "DOWN & DISTANCE", "HASH", "OFF FORM",
        "OFF PLAY", "DEF CALL", "GN/LS", "RESULT", "COMMENTS",
    ]
    explosive = df[df["_EXPLOSIVE"]].copy()
    if explosive.empty:
        return pd.DataFrame(columns=columns)
    explosive["DOWN & DISTANCE"] = explosive["SITUATION"]
    explosive["_SITUATION_ORDER"] = explosive["SITUATION"].map(SITUATION_ORDER).fillna(99)
    explosive["_PLAY_NUM_ORDER"] = pd.to_numeric(explosive["PLAY #"], errors="coerce")
    explosive = explosive.sort_values(
        ["_SITUATION_ORDER", "_PLAY_NUM_ORDER"],
        ascending=[True, True],
        na_position="last",
    )
    return explosive[columns].reset_index(drop=True)


def _pattern_table(df, column, length=2, min_occurrences=2):
    columns = ["PATTERN", "OCCURRENCES"]
    values = df[column].map(clean).tolist()
    if len(values) < length:
        return pd.DataFrame(columns=columns)
    rows = []
    for i in range(len(values) - length + 1):
        chunk = values[i:i + length]
        if all(chunk):
            rows.append({"PATTERN": " → ".join(chunk)})
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows).groupby("PATTERN").size()
        .reset_index(name="OCCURRENCES")
        .query("OCCURRENCES >= @min_occurrences")
        .sort_values(["OCCURRENCES", "PATTERN"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _result_stops(df):
    """Return individual low-gain and disruptive results for quick review.

    Includes plays with 5 yards or fewer, plus incompletions, interceptions,
    and fumbles. This is a descriptive play/situation list, not a ranking.
    """
    columns = [
        "PLAY #", "DOWN & DISTANCE", "HASH", "OFF FORM", "OFF PLAY",
        "DEF CALL", "GN/LS", "RESULT", "COMMENTS",
    ]
    work = df.copy()
    work["DOWN & DISTANCE"] = work["SITUATION"]
    yards = pd.to_numeric(work["_YARDS"], errors="coerce")
    qualifying = (
        (yards.notna() & (yards <= 5))
        | work["_INCOMPLETE"]
        | work.apply(is_interception, axis=1)
        | work.apply(is_fumble, axis=1)
    )
    result = work[qualifying].copy()
    if result.empty:
        return pd.DataFrame(columns=columns)

    # Keep the section organized by down/distance, then by play number.
    result["_SITUATION_ORDER"] = result["SITUATION"].map(SITUATION_ORDER).fillna(99)
    result["_PLAY_NUM_ORDER"] = pd.to_numeric(result["PLAY #"], errors="coerce")
    result = result.sort_values(
        ["_SITUATION_ORDER", "_PLAY_NUM_ORDER"],
        ascending=[True, True],
        na_position="last",
    )

    return result[columns].reset_index(drop=True)


def _strength_improvement_indicators(df, min_occurrences=5):
    columns = ["CATEGORY", "GROUP", "PLAYS", "METRIC", "VALUE", "OVERALL", "DIRECTION"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    yards = pd.to_numeric(df["_YARDS"], errors="coerce")
    overall = {
        "AVG YDS": yards.mean() if yards.notna().any() else None,
        "EXPLOSIVE %": df["_EXPLOSIVE"].mean() * 100,
        "TFL %": df["_TFL"].mean() * 100,
        "SACK %": df["_SACK"].mean() * 100,
        "TURNOVER %": df["_TURNOVER"].mean() * 100,
        "TD %": df["_TD"].mean() * 100,
    }
    dimensions = {
        "DEF CALL": "DEF CALL", "DEF FRONT": "DEF FRONT",
        "COVERAGE": "COVERAGE", "BLITZ": "BLITZ",
        "OFF FORM": "OFF FORM", "PERSONNEL": "PERSONNEL",
        "SITUATION": "SITUATION",
    }
    rows = []
    for column, category in dimensions.items():
        for value, group in df.groupby(column):
            if len(group) < min_occurrences:
                continue
            y = pd.to_numeric(group["_YARDS"], errors="coerce")
            metrics = {
                "AVG YDS": y.mean() if y.notna().any() else None,
                "EXPLOSIVE %": group["_EXPLOSIVE"].mean() * 100,
                "TFL %": group["_TFL"].mean() * 100,
                "SACK %": group["_SACK"].mean() * 100,
                "TURNOVER %": group["_TURNOVER"].mean() * 100,
                "TD %": group["_TD"].mean() * 100,
            }
            for metric, value_metric in metrics.items():
                baseline = overall[metric]
                if baseline is None or value_metric is None:
                    continue
                delta = value_metric - baseline
                if abs(delta) < 0.01:
                    continue
                lower_is_better = metric in {"AVG YDS", "EXPLOSIVE %", "TD %"}
                direction = (
                    "LOWER THAN OVERALL" if delta < 0 else "HIGHER THAN OVERALL"
                ) if lower_is_better else (
                    "HIGHER THAN OVERALL" if delta > 0 else "LOWER THAN OVERALL"
                )
                rows.append({
                    "CATEGORY": category,
                    "GROUP": clean(value) or "UNKNOWN",
                    "PLAYS": len(group),
                    "METRIC": metric,
                    "VALUE": round(value_metric, 1),
                    "OVERALL": round(baseline, 1),
                    "DIRECTION": direction,
                })
    return pd.DataFrame(rows, columns=columns)



def _opponent_play_type(row):
    """Normalize opponent play type for matchup context."""
    return play_type(row)


def _matchup_context(opponent_df, defense_df, min_opponent=3, min_defense=5):
    """Compare opponent frequency with WHS defensive results.

    ALL INFO SHEET answers "what does the opponent do?" while WHS DATA
    answers "what has our defense done against the same situation + formation?"
    This is descriptive only and does not recommend a defensive call.
    """
    opponent = opponent_df.copy()
    defense = defense_df.copy()

    opponent = opponent[opponent["PLAY #"].map(clean) != ""].reset_index(drop=True)
    defense = defense[defense["PLAY #"].map(clean) != ""].reset_index(drop=True)

    opponent["SITUATION"] = [
        situation_bucket(d, dist) for d, dist in zip(opponent["DN"], opponent["DIST"])
    ]
    defense["SITUATION"] = [
        situation_bucket(d, dist) for d, dist in zip(defense["DN"], defense["DIST"])
    ]

    for frame in (opponent, defense):
        frame["OFF FORM"] = frame["OFF FORM"].map(clean)
        frame["PERSONNEL"] = frame["PERSONNEL"].map(clean)
        frame["MOTION"] = frame["MOTION"].map(clean)

    # Primary comparison: down/distance + offensive formation.
    opponent = opponent[
        (opponent["SITUATION"] != "") & (opponent["OFF FORM"] != "")
    ].copy()
    defense = defense[
        (defense["SITUATION"] != "") & (defense["OFF FORM"] != "")
    ].copy()

    context_columns = [
        "DOWN & DISTANCE", "OFF FORM", "OPP PLAYS", "OPP %",
        "OPP RUN %", "OPP PASS %", "WHS MATCH PLAYS",
        "WHS AVG YDS", "WHS EXP %", "WHS TD %",
    ]
    call_columns = [
        "DOWN & DISTANCE", "OFF FORM", "DEF CALL", "PLAYS",
        "AVG YDS", "EXP %", "TD %", "RELATIVE RESULT",
    ]

    if opponent.empty or defense.empty:
        return pd.DataFrame(columns=context_columns), pd.DataFrame(columns=call_columns)

    opponent["_PLAY_TYPE"] = opponent.apply(_opponent_play_type, axis=1)
    opp_groups = (
        opponent.groupby(["SITUATION", "OFF FORM"])
        .agg(
            **{
                "OPP PLAYS": ("PLAY #", "size"),
                "RUN": ("_PLAY_TYPE", lambda s: int((s == "RUN").sum())),
                "PASS": ("_PLAY_TYPE", lambda s: int((s == "PASS").sum())),
            }
        )
        .reset_index()
    )

    defense["_YARDS"] = defense.apply(result_yards, axis=1)
    defense["_EXPLOSIVE"] = defense.apply(is_explosive, axis=1)
    defense["_TD"] = defense.apply(is_touchdown, axis=1)

    defense_groups = (
        defense.groupby(["SITUATION", "OFF FORM"])
        .agg(
            **{
                "WHS MATCH PLAYS": ("PLAY #", "size"),
                "WHS AVG YDS": ("_YARDS", "mean"),
                "WHS EXP %": ("_EXPLOSIVE", "mean"),
                "WHS TD %": ("_TD", "mean"),
            }
        )
        .reset_index()
    )

    context = opp_groups.merge(
        defense_groups,
        on=["SITUATION", "OFF FORM"],
        how="inner",
    )
    context = context[
        (context["OPP PLAYS"] >= min_opponent)
        & (context["WHS MATCH PLAYS"] >= min_defense)
    ].copy()

    if context.empty:
        return pd.DataFrame(columns=context_columns), pd.DataFrame(columns=call_columns)

    opp_total = len(opponent)
    context["OPP %"] = (context["OPP PLAYS"] / opp_total * 100).round(1)
    context["OPP RUN %"] = (context["RUN"] / context["OPP PLAYS"] * 100).round(1)
    context["OPP PASS %"] = (context["PASS"] / context["OPP PLAYS"] * 100).round(1)
    context["WHS AVG YDS"] = context["WHS AVG YDS"].round(1)
    context["WHS EXP %"] = (context["WHS EXP %"] * 100).round(1)
    context["WHS TD %"] = (context["WHS TD %"] * 100).round(1)
    context = context.rename(columns={"SITUATION": "DOWN & DISTANCE"})[context_columns]

    # Compare the defensive calls used in the matching WHS sample.
    matched_keys = context.rename(columns={"DOWN & DISTANCE": "SITUATION"})[
        ["SITUATION", "OFF FORM"]
    ].drop_duplicates()
    defense_matched = defense.merge(
        matched_keys,
        on=["SITUATION", "OFF FORM"],
        how="inner",
    )
    defense_matched["DEF CALL"] = defense_matched["DEF CALL"].map(clean)
    defense_matched = defense_matched[defense_matched["DEF CALL"] != ""]

    rows = []
    for (situation, form, call), group in defense_matched.groupby(
        ["SITUATION", "OFF FORM", "DEF CALL"]
    ):
        # Require a small call sample before labeling relative results.
        if len(group) < 3:
            continue
        yards = pd.to_numeric(group["_YARDS"], errors="coerce")
        rows.append({
            "DOWN & DISTANCE": situation,
            "OFF FORM": form,
            "DEF CALL": call,
            "PLAYS": len(group),
            "AVG YDS": round(yards.mean(), 1) if yards.notna().any() else "",
            "EXP %": round(group["_EXPLOSIVE"].mean() * 100, 1),
            "TD %": round(group["_TD"].mean() * 100, 1),
        })

    calls = pd.DataFrame(rows, columns=call_columns[:-1])
    if calls.empty:
        calls["RELATIVE RESULT"] = pd.Series(dtype=str)
        return context.reset_index(drop=True), calls[call_columns]

    calls["RELATIVE RESULT"] = ""
    for (situation, form), group in calls.groupby(["DOWN & DISTANCE", "OFF FORM"]):
        valid = group[pd.to_numeric(group["AVG YDS"], errors="coerce").notna()]
        if len(valid) < 2:
            continue
        min_yards = valid["AVG YDS"].min()
        max_yards = valid["AVG YDS"].max()
        calls.loc[valid.index[valid["AVG YDS"] == min_yards], "RELATIVE RESULT"] = "LOWER YDS/PLAY"
        calls.loc[valid.index[valid["AVG YDS"] == max_yards], "RELATIVE RESULT"] = "HIGHER YDS/PLAY"

    situation_order = {
        "1st & Long (10+)": 1, "1st & Short (1-9)": 2,
        "2nd & Long (7+)": 3, "2nd & Medium (4-6)": 4,
        "2nd & Short (1-3)": 5, "3rd & Long (7+)": 6,
        "3rd & Medium (4-6)": 7, "3rd & Short (1-3)": 8,
        "4th Down": 9,
    }
    context["_ORDER"] = context["DOWN & DISTANCE"].map(situation_order).fillna(99)
    context = context.sort_values(
        ["_ORDER", "OPP PLAYS", "OFF FORM"],
        ascending=[True, False, True],
    ).drop(columns="_ORDER").reset_index(drop=True)

    calls["_ORDER"] = calls["DOWN & DISTANCE"].map(situation_order).fillna(99)
    calls = calls.sort_values(
        ["_ORDER", "OFF FORM", "AVG YDS", "PLAYS", "DEF CALL"],
        ascending=[True, True, True, False, True],
    ).drop(columns="_ORDER").reset_index(drop=True)

    return context, calls[call_columns]



def formation_call_matrix(opponent_df, defense_df, min_call_plays=2, top_n=3):
    """Build the bottom Tendencies formation/call matrix.

    Opponent formations and run/pass mix come from ALL INFO SHEET.
    Historical defensive calls come only from WHS DATA rows where ODK == D.
    Calls are matched on field zone + down/distance + offensive formation and
    the highest historical defensive success rate is shown.
    """
    zones = ["OWN 1-20", "OWN 21-40", "OWN 41-49", "MIDFIELD", "OPP 21-40", "RED ZONE"]
    situations = [
        "1st & Long (10+)", "2nd & Long (7+)", "2nd & Medium (4-6)",
        "2nd & Short (1-3)", "3rd & Long (7+)", "3rd & Medium (4-6)",
        "3rd & Short (1-3)",
    ]
    columns = [
        "FIELD ZONE", "DOWN & DISTANCE", "RUN %", "PASS %",
        "FORMATION 1", "DEF CALL", "FORMATION 2", "DEF CALL",
        "FORMATION 3", "DEF CALL",
    ]

    if opponent_df is None or opponent_df.empty:
        return pd.DataFrame(columns=columns)

    opp = opponent_df.copy()
    opp["FIELD ZONE"] = opp["YARD LN"].map(_field_zone)
    opp["SITUATION"] = [
        situation_bucket(d, dist) for d, dist in zip(opp["DN"], opp["DIST"])
    ]
    opp["FORMATION"] = opp["OFF FORM"].map(clean)
    opp["_RUN"] = opp.apply(lambda r: int(play_type(r) == "RUN"), axis=1)
    opp["_PASS"] = opp.apply(lambda r: int(play_type(r) == "PASS"), axis=1)
    opp = opp[opp["FIELD ZONE"].isin(zones) & opp["SITUATION"].isin(situations)].copy()

    if opp.empty:
        return pd.DataFrame(columns=columns)

    mix = opp.groupby(["FIELD ZONE", "SITUATION"]).agg(
        PLAYS=("_RUN", "size"), RUN=("_RUN", "sum"), PASS=("_PASS", "sum")
    ).reset_index()
    mix["RUN %"] = (mix["RUN"] / mix["PLAYS"] * 100).round(1)
    mix["PASS %"] = (mix["PASS"] / mix["PLAYS"] * 100).round(1)

    formations = (
        opp[opp["FORMATION"] != ""]
        .groupby(["FIELD ZONE", "SITUATION", "FORMATION"])
        .size().reset_index(name="PLAYS")
        .sort_values(["FIELD ZONE", "SITUATION", "PLAYS", "FORMATION"],
                     ascending=[True, True, False, True])
    )

    call_lookup = {}
    if defense_df is not None and not defense_df.empty:
        d = defense_df.copy()
        # load_defense_source_df already filters WHS DATA to ODK=D, but
        # retain this guard so the function is safe when called directly.
        if "ODK" in d.columns:
            d = d[d["ODK"].map(clean).str.upper() == "D"].copy()
        d["FIELD ZONE"] = d["YARD LN"].map(_field_zone)
        d["SITUATION"] = [
            situation_bucket(x, y) for x, y in zip(d["DN"], d["DIST"])
        ]
        d["FORMATION"] = d["OFF FORM"].map(clean)
        d["DEF CALL"] = d["DEF CALL"].map(clean)

        def formation_key(value):
            # Make formation matching tolerant of capitalization, punctuation,
            # and labels such as TRIPS RIGHT vs TRIPS.
            return re.sub(r"[^A-Z0-9]", "", clean(value).upper())

        d["_FORMATION_KEY"] = d["FORMATION"].map(formation_key)

        def success(row):
            """Measure defensive success relative to the down/distance."""
            yards = result_yards(row)
            result = result_blob(row)
            if any(x in result for x in (
                "INCOMPLETE", "INTERCEPTION", "INTERCEPTED",
                "FUMBLE", "FUMBLED", "SACK",
            )):
                return True
            if yards is None:
                return False

            down = numeric(row.get("DN"))
            distance = numeric(row.get("DIST"))
            if down is None or distance is None:
                return yards <= 5

            down = int(down)
            distance = max(0, distance)

            # Standard down-success thresholds:
            # 1st: prevent 40% of the line to gain
            # 2nd: prevent 60% of the line to gain
            # 3rd/4th: prevent the first down
            if down == 1:
                return yards <= distance * 0.40
            if down == 2:
                return yards <= distance * 0.60
            return yards < distance

        d["_SUCCESS"] = d.apply(success, axis=1)
        d = d[
            d["FIELD ZONE"].isin(zones)
            & d["SITUATION"].isin(situations)
            & (d["FORMATION"] != "")
            & (d["DEF CALL"] != "")
        ].copy()

        if not d.empty:
            stats = d.groupby(
                ["FIELD ZONE", "SITUATION", "_FORMATION_KEY", "DEF CALL"]
            ).agg(
                PLAYS=("_SUCCESS", "size"),
                SUCCESS=("_SUCCESS", "sum"),
            ).reset_index()
            stats["SUCCESS %"] = (stats["SUCCESS"] / stats["PLAYS"] * 100).round(1)
            stats = stats[stats["PLAYS"] >= min_call_plays]

            for key, group in stats.groupby(["FIELD ZONE", "SITUATION", "_FORMATION_KEY"]):
                ranked = group.sort_values(
                    ["SUCCESS %", "PLAYS", "DEF CALL"],
                    ascending=[False, False, True],
                )
                best = ranked.iloc[0]
                call_lookup[key] = best["DEF CALL"]

    rows = []
    for zone in zones:
        for situation in situations:
            m = mix[(mix["FIELD ZONE"] == zone) & (mix["SITUATION"] == situation)]
            run_pct = m.iloc[0]["RUN %"] if not m.empty else ""
            pass_pct = m.iloc[0]["PASS %"] if not m.empty else ""
            forms = formations[
                (formations["FIELD ZONE"] == zone) &
                (formations["SITUATION"] == situation)
            ].head(top_n)

            cells = []
            for form in forms["FORMATION"].tolist():
                form_key = re.sub(r"[^A-Z0-9]", "", clean(form).upper())
                call = call_lookup.get((zone, situation, form_key), "")

                # If the exact normalized formation is not present in WHS DATA,
                # allow a clear label variant such as TRIPS RIGHT/TRIPS or
                # BUNCH LEFT/BUNCH to use the same historical formation family.
                if not call and form_key:
                    candidates = [
                        (k, v) for k, v in call_lookup.items()
                        if k[0] == zone and k[1] == situation
                        and (k[2] in form_key or form_key in k[2])
                    ]
                    if candidates:
                        candidates.sort(key=lambda item: (len(item[0][2]), item[1]))
                        call = candidates[0][1]

                cells.extend([form, call])
            cells += [""] * (top_n * 2 - len(cells))

            rows.append([
                zone, situation, run_pct, pass_pct,
                cells[0], cells[1], cells[2], cells[3], cells[4], cells[5],
            ])

    return pd.DataFrame(rows, columns=columns)

@dataclass
class MatchupReport:
    context: pd.DataFrame
    calls: pd.DataFrame


def analyze_matchup(opponent_df, defense_df, min_opponent=3, min_defense=5):
    return MatchupReport(
        *_matchup_context(
            opponent_df,
            defense_df,
            min_opponent=min_opponent,
            min_defense=min_defense,
        )
    )

@dataclass
class DefenseReport:
    overall: pd.DataFrame
    by_def_call: pd.DataFrame
    by_front: pd.DataFrame
    by_coverage: pd.DataFrame
    by_blitz: pd.DataFrame
    by_off_form: pd.DataFrame
    by_personnel: pd.DataFrame
    by_situation: pd.DataFrame
    by_play_type: pd.DataFrame
    situation_calls: pd.DataFrame
    call_by_situation: pd.DataFrame
    explosive_context: pd.DataFrame
    call_patterns: pd.DataFrame
    coverage_patterns: pd.DataFrame
    blitz_patterns: pd.DataFrame
    result_stops: pd.DataFrame
    indicators: pd.DataFrame


def analyze_defense(df, min_occurrences=5):
    missing = [c for c in DEFENSIVE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing defensive columns: {missing}")

    work = df[DEFENSIVE_COLUMNS].copy()
    work = work[work["PLAY #"].map(clean) != ""].reset_index(drop=True)
    work["SITUATION"] = [
        situation_bucket(d, dist) for d, dist in zip(work["DN"], work["DIST"])
    ]
    work["BLITZ"] = work["BLITZ"].map(normalize_blitz)
    work["_YARDS"] = work.apply(result_yards, axis=1)
    work["_PLAY_TYPE"] = work.apply(play_type, axis=1)
    work["_EXPLOSIVE"] = work.apply(is_explosive, axis=1)
    work["_TFL"] = work.apply(is_tfl, axis=1)
    work["_SACK"] = work.apply(is_sack, axis=1)
    work["_TURNOVER"] = work.apply(is_turnover, axis=1)
    work["_TD"] = work.apply(is_touchdown, axis=1)
    work["_INCOMPLETE"] = work.apply(is_incomplete, axis=1)

    yards = pd.to_numeric(work["_YARDS"], errors="coerce")
    overall = pd.DataFrame([{
        "PLAYS": len(work),
        "AVG YDS": round(yards.mean(), 1) if yards.notna().any() else "",
        "RUN": int((work["_PLAY_TYPE"] == "RUN").sum()),
        "PASS": int((work["_PLAY_TYPE"] == "PASS").sum()),
        "TFL": int(work["_TFL"].sum()),
        "TFL %": round(work["_TFL"].mean() * 100, 1) if len(work) else 0,
        # Havoc is the share of snaps with at least one disruptive event:
        # TFL/sack or turnover. This counts each play once.
        "HAVOC": int((work["_TFL"] | work["_SACK"] | work["_TURNOVER"]).sum()),
        "HAVOC %": round(
            (work["_TFL"] | work["_SACK"] | work["_TURNOVER"]).mean() * 100, 1
        ) if len(work) else 0,
        "SACK": int(work["_SACK"].sum()),
        "SACK %": round(work["_SACK"].mean() * 100, 1) if len(work) else 0,
        "TURNOVERS": int(work["_TURNOVER"].sum()),
        "TURNOVER %": round(work["_TURNOVER"].mean() * 100, 1) if len(work) else 0,
        "EXPLOSIVES": int(work["_EXPLOSIVE"].sum()),
        "EXPLOSIVE %": round(work["_EXPLOSIVE"].mean() * 100, 1) if len(work) else 0,
        "TD": int(work["_TD"].sum()),
        "TD %": round(work["_TD"].mean() * 100, 1) if len(work) else 0,
        "INCOMPLETIONS": int(work["_INCOMPLETE"].sum()),
    }])

    situation = _group_metrics(work, "SITUATION", min_occurrences)
    if not situation.empty:
        situation["_ORDER"] = situation["SITUATION"].map(SITUATION_ORDER).fillna(99)
        situation = situation.sort_values(["_ORDER", "PLAYS"]).drop(columns="_ORDER")

    return DefenseReport(
        overall=overall,
        by_def_call=_call_result(work, "DEF CALL", min_occurrences),
        by_front=_call_result(work, "DEF FRONT", min_occurrences),
        by_coverage=_call_result(work, "COVERAGE", min_occurrences),
        by_blitz=_call_result(work, "BLITZ", min_occurrences),
        by_off_form=_call_result(work, "OFF FORM", min_occurrences),
        by_personnel=_call_result(work, "PERSONNEL", min_occurrences),
        by_situation=situation,
        by_play_type=_call_result(work, "_PLAY_TYPE", min_occurrences),
        situation_calls=_situation_calls(work, min_occurrences),
        call_by_situation=_call_by_situation(work, min_occurrences),
        explosive_context=_explosive_context(work),
        call_patterns=_pattern_table(work, "DEF CALL"),
        coverage_patterns=_pattern_table(work, "COVERAGE"),
        blitz_patterns=_pattern_table(work, "BLITZ"),
        result_stops=_result_stops(work),
        indicators=_strength_improvement_indicators(work, min_occurrences),
    )


def write_defense_report(report, output_dir="output"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.overall.to_csv(out / "defense_summary.csv", index=False)
    report.by_def_call.to_csv(out / "defense_by_call.csv", index=False)
    report.by_front.to_csv(out / "defense_by_front.csv", index=False)
    report.by_coverage.to_csv(out / "defense_by_coverage.csv", index=False)
    report.by_blitz.to_csv(out / "defense_by_blitz.csv", index=False)
    report.by_off_form.to_csv(out / "defense_by_off_form.csv", index=False)
    report.by_personnel.to_csv(out / "defense_by_personnel.csv", index=False)
    report.by_situation.to_csv(out / "defense_by_situation.csv", index=False)
    report.by_play_type.to_csv(out / "defense_by_play_type.csv", index=False)
    report.situation_calls.to_csv(out / "defense_situation_calls.csv", index=False)
    report.call_by_situation.to_csv(out / "defense_call_by_situation.csv", index=False)
    report.explosive_context.to_csv(out / "defense_explosives.csv", index=False)
    report.call_patterns.to_csv(out / "defense_call_patterns.csv", index=False)
    report.coverage_patterns.to_csv(out / "defense_coverage_patterns.csv", index=False)
    report.blitz_patterns.to_csv(out / "defense_blitz_patterns.csv", index=False)
    report.result_stops.to_csv(out / "defense_result_stops.csv", index=False)
    report.indicators.to_csv(out / "defense_strength_improvement_indicators.csv", index=False)

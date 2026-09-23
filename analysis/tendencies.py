"""
Gridiron Live tendency analyzer.
Reads coach-entered ALL INFO SHEET data and produces factual tendency summaries.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

from config import REQUIRED_COLUMNS, EXPLOSIVE_RUN_YARDS, EXPLOSIVE_PASS_YARDS


def clean(value):
    return "" if pd.isna(value) else str(value).strip()


def numeric(value):
    try:
        if pd.isna(value) or clean(value) == "":
            return None
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def result_yards(value):
    text = clean(value).upper()
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*(?:YDS?|YARDS?)?(?!\d)", text)
    return float(match.group(1)) if match else None


def play_blob(row):
    return " ".join(clean(row.get(c, "")) for c in ["PLAY TYPE", "OFF PLAY", "RESULT"]).upper()


def classify_play(row):
    """Classify a snap for factual statistical rollups."""
    b = play_blob(row)
    if any(x in b for x in ["SCRAMBLE", "QB RUN", "KEEP", "QB SNEAK"]):
        return "QB RUN"
    if "SACK" in b:
        return "SACK"
    if any(x in b for x in ["RUN", "RUSH", "DRAW", "SNEAK"]):
        return "RUN"
    if any(x in b for x in ["PASS", "SCREEN", "RPO", "QB THROW", "COMPLETE", "INCOMPLETE", "INTERCEPTION"]):
        return "PASS"
    return "OTHER"


def is_pass(row):
    return classify_play(row) == "PASS"


def is_run(row):
    return classify_play(row) in {"RUN", "QB RUN"}


def is_explosive(row):
    yards = numeric(row.get("GN/LS"))
    if yards is None:
        yards = result_yards(row.get("RESULT"))
    if yards is None:
        return False
    if is_pass(row):
        return yards >= EXPLOSIVE_PASS_YARDS
    if is_run(row):
        return yards >= EXPLOSIVE_RUN_YARDS
    return False


def is_penalty(row):
    b = play_blob(row)
    return any(x in b for x in [
        "PENALTY", "HOLDING", "FALSE START", "OFFSIDES", "ENCROACHMENT",
        "NEUTRAL ZONE", "PASS INTERFERENCE", "DEFENSIVE PASS INTERFERENCE",
        "OFFENSIVE PASS INTERFERENCE", "PERSONAL FOUL", "FACEMASK",
        "UNSPORTSMANLIKE", "BLOCK IN THE BACK", "ILLEGAL MOTION",
        "ILLEGAL FORMATION", "DELAY OF GAME"
    ])


def play_name(row):
    value = clean(row.get("OFF PLAY", ""))
    if value:
        return value
    value = clean(row.get("PLAY TYPE", ""))
    if value:
        return value
    result = clean(row.get("RESULT", ""))
    return result if result else "UNKNOWN"


def play_label(row):
    name = play_name(row)
    direction = clean(row.get("PLAY DIR", ""))
    return f"{name} {direction}".strip()


def down_distance_tendencies(df):
    """Consolidate down and distance into coach-friendly situations."""
    rows = []
    for _, row in df.iterrows():
        down = numeric(row.get("DN"))
        dist = numeric(row.get("DIST"))
        if down is None or dist is None:
            continue
        down = int(down)
        dist = max(0, dist)

        if down == 1:
            bucket = "1st & Long (10+)" if dist >= 10 else "1st & Short (1-9)"
        elif down == 2:
            if dist >= 7:
                bucket = "2nd & Long (7+)"
            elif dist >= 4:
                bucket = "2nd & Medium (4-6)"
            else:
                bucket = "2nd & Short (1-3)"
        elif down == 3:
            if dist >= 7:
                bucket = "3rd & Long (7+)"
            elif dist >= 4:
                bucket = "3rd & Medium (4-6)"
            else:
                bucket = "3rd & Short (1-3)"
        elif down == 4:
            bucket = "4th Down"
        else:
            continue

        rows.append({
            "DOWN & DISTANCE": bucket,
            "RUN": int(is_run(row)),
            "PASS": int(is_pass(row)),
            "EXPLOSIVE": int(bool(row.get("EXPLOSIVE", False))),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "DOWN & DISTANCE", "PLAYS", "RUN", "PASS", "EXPLOSIVES", "EXPLOSIVE_RATE"
        ])

    work = pd.DataFrame(rows)
    out = work.groupby("DOWN & DISTANCE").agg(
        PLAYS=("RUN", "size"),
        RUN=("RUN", "sum"),
        PASS=("PASS", "sum"),
        EXPLOSIVES=("EXPLOSIVE", "sum"),
    ).reset_index()

    out["EXPLOSIVE_RATE"] = (out["EXPLOSIVES"] / out["PLAYS"] * 100).round(1)

    order = {
        "1st & Long (10+)": 1,
        "1st & Short (1-9)": 2,
        "2nd & Long (7+)": 3,
        "2nd & Medium (4-6)": 4,
        "2nd & Short (1-3)": 5,
        "3rd & Long (7+)": 6,
        "3rd & Medium (4-6)": 7,
        "3rd & Short (1-3)": 8,
        "4th Down": 9,
    }
    out["_ORDER"] = out["DOWN & DISTANCE"].map(order)
    return out.sort_values("_ORDER").drop(columns="_ORDER").reset_index(drop=True)





def canonical_like_term_map(values):
    """Map longer terms to a shorter closely related term when it contains it."""
    unique = sorted(
        {clean(value) for value in values if clean(value)},
        key=lambda value: (
            len(re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()),
            value.upper(),
        ),
    )
    mapping = {}

    normalized_values = {
        value: re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
        for value in unique
    }

    for value in unique:
        normalized = normalized_values[value]
        match = None

        for candidate in unique:
            if candidate == value:
                continue

            candidate_normalized = normalized_values[candidate]
            if (
                len(candidate_normalized) < len(normalized)
                and re.search(
                    rf"(?<![A-Z0-9]){re.escape(candidate_normalized)}(?![A-Z0-9])",
                    normalized,
                )
            ):
                match = candidate
                break

        mapping[value] = match if match is not None else value

    return mapping

def hash_down_distance_play_probabilities(df, min_occurrences=1, top_n=3):
    """Show the top formation/play combinations for each down/distance situation."""
    rows = []

    for _, row in df.iterrows():
        situation = situation_bucket(row.get("DN", ""), row.get("DIST", ""))
        formation = clean(row.get("OFF FORM", ""))
        play = normalize_favorite_play(play_label(row))

        if not situation:
            continue

        rows.append({
            "DOWN & DISTANCE": situation,
            "FORMATION": formation or "UNKNOWN",
            "PLAY": play if play and play != "UNKNOWN" else "UNKNOWN",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "DOWN & DISTANCE", "PLAYS",
            "FORMATION 1", "FORMATION 2", "FORMATION 3"
        ])

    work = pd.DataFrame(rows)

    formation_map = canonical_like_term_map(
        work.loc[work["FORMATION"] != "UNKNOWN", "FORMATION"].tolist()
    )
    work["FORMATION"] = work["FORMATION"].map(
        lambda value: formation_map.get(value, value)
    )

    summary_rows = []
    for situation, group in work.groupby("DOWN & DISTANCE"):
        total = len(group)

        combo_counts = (
            group[
                (group["FORMATION"] != "UNKNOWN") &
                (group["PLAY"] != "UNKNOWN")
            ]
            .groupby(["FORMATION", "PLAY"])
            .size()
            .sort_values(ascending=False)
            .head(top_n)
        )

        top_combinations = [
            f"{formation}/{play} ({count}/{total}, {round(count / total * 100, 1)}%)"
            for (formation, play), count in combo_counts.items()
        ]
        top_combinations += ["—"] * (top_n - len(top_combinations))

        summary_rows.append({
            "DOWN & DISTANCE": situation,
            "PLAYS": total,
            "FORMATION 1": top_combinations[0],
            "FORMATION 2": top_combinations[1],
            "FORMATION 3": top_combinations[2],
        })

    out = pd.DataFrame(summary_rows)

    order = {
        "1st & Long (10+)": 1,
        "1st & Short (1-9)": 2,
        "2nd & Long (7+)": 3,
        "2nd & Medium (4-6)": 4,
        "2nd & Short (1-3)": 5,
        "3rd & Long (7+)": 6,
        "3rd & Medium (4-6)": 7,
        "3rd & Short (1-3)": 8,
        "4th Down": 9,
    }
    out["_ORDER"] = out["DOWN & DISTANCE"].map(order)

    return (
        out.sort_values("_ORDER")
        .drop(columns="_ORDER")
        .reset_index(drop=True)
    )

def group_rates(df, column):
    work = df[df[column].map(clean) != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=[column, "PLAYS", "EXPLOSIVES", "EXPLOSIVE_RATE"])
    out = work.groupby(column).agg(PLAYS=("EXPLOSIVE", "size"), EXPLOSIVES=("EXPLOSIVE", "sum")).reset_index()
    out["EXPLOSIVE_RATE"] = (out["EXPLOSIVES"] / out["PLAYS"] * 100).round(1)
    out = out[out["EXPLOSIVES"] > 0]
    return out.sort_values(
        ["EXPLOSIVE_RATE", "EXPLOSIVES", "PLAYS"],
        ascending=False,
    ).head(10)


def group_rates_combo(df, columns, top_n=20):
    """Group explosive rates by a combined contextual lens."""
    work = df.copy()
    for column in columns:
        work = work[work[column].map(clean) != ""]
    if work.empty:
        return pd.DataFrame(columns=list(columns) + ["PLAYS", "EXPLOSIVES", "EXPLOSIVE_RATE"])

    out = (
        work.groupby(list(columns))
        .agg(
            PLAYS=("EXPLOSIVE", "size"),
            EXPLOSIVES=("EXPLOSIVE", "sum"),
        )
        .reset_index()
    )
    out["EXPLOSIVE_RATE"] = (out["EXPLOSIVES"] / out["PLAYS"] * 100).round(1)
    out = out[out["EXPLOSIVES"] > 0]
    return out.sort_values(
        ["EXPLOSIVE_RATE", "EXPLOSIVES", "PLAYS"],
        ascending=False,
    ).head(top_n).reset_index(drop=True)


def previous_play_features(df):
    if len(df) < 2:
        return pd.DataFrame()
    cols = ["OFF FORM", "PERSONNEL", "MOTION", "PLAY TYPE", "PLAY DIR", "PLAY (STR/WK)", "HASH", "DN", "DIST"]
    rows = []
    for i in range(1, len(df)):
        if not bool(df.iloc[i]["EXPLOSIVE"]):
            continue
        prev = df.iloc[i - 1]
        for col in cols:
            value = clean(prev[col])
            if not value:
                continue
            previous_values = df.iloc[:-1][col].map(clean)
            mask = (previous_values == value).to_numpy()
            total = int(mask.sum())
            explosives = int(df["EXPLOSIVE"].iloc[1:].to_numpy()[mask].sum())
            overall_mask = df[col].map(clean) == value
            overall_total = int(overall_mask.sum())
            overall_explosives = int(df.loc[overall_mask, "EXPLOSIVE"].sum())
            rows.append({
                "PREVIOUS_FEATURE": col,
                "VALUE": value,
                "EXPLOSIVE_FOLLOWING_PLAYS": explosives,
                "TOTAL_FOLLOWING_PLAYS": total,
                "FOLLOWING_EXPLOSIVE_RATE": round(explosives / total * 100, 1) if total else 0,
                "OVERALL_FEATURE_PLAYS": overall_total,
                "OVERALL_EXPLOSIVES": overall_explosives,
                "OVERALL_EXPLOSIVE_RATE": round(overall_explosives / overall_total * 100, 1) if overall_total else 0,
            })
    out = pd.DataFrame(rows).drop_duplicates()
    if out.empty:
        return out
    out["RATE_DIFFERENCE"] = (out["FOLLOWING_EXPLOSIVE_RATE"] - out["OVERALL_EXPLOSIVE_RATE"]).round(1)
    return out.sort_values(["RATE_DIFFERENCE", "FOLLOWING_EXPLOSIVE_RATE", "TOTAL_FOLLOWING_PLAYS"], ascending=False)


def trigger_sequence_analysis(df):
    """Find what commonly follows meaningful trigger plays."""
    if len(df) < 2:
        return pd.DataFrame()

    trigger_rows = []
    for i in range(len(df) - 1):
        current = df.iloc[i]
        following = df.iloc[i + 1]
        trigger_types = []

        if is_penalty(current):
            trigger_types.append("PENALTY")
        if "INCOMPLETE" in play_blob(current):
            trigger_types.append("INCOMPLETE PASS")
        if is_pass(current) and "INCOMPLETE" not in play_blob(current):
            trigger_types.append("PASS")
        if is_run(current):
            trigger_types.append("RUN")
        if bool(current["EXPLOSIVE"]):
            trigger_types.append("EXPLOSIVE PLAY")
        if numeric(current.get("GN/LS")) is not None and numeric(current.get("GN/LS")) < 0:
            trigger_types.append("NEGATIVE PLAY")

        for trigger_type in trigger_types:
            trigger_rows.append({
                "TRIGGER": trigger_type,
                "TRIGGER_PLAY": play_label(current),
                "NEXT_PLAY": play_label(following),
                "NEXT_PLAY_TYPE": clean(following["PLAY TYPE"]),
                "NEXT_FORMATION": clean(following["OFF FORM"]),
            })

    out = pd.DataFrame(trigger_rows)
    if out.empty:
        return out

    grouped = (
        out.groupby(["TRIGGER", "NEXT_PLAY", "NEXT_PLAY_TYPE", "NEXT_FORMATION"])
        .size()
        .reset_index(name="FOLLOWING_COUNT")
    )
    totals = grouped.groupby("TRIGGER")["FOLLOWING_COUNT"].transform("sum")
    grouped["FOLLOWING_RATE"] = (grouped["FOLLOWING_COUNT"] / totals * 100).round(1)
    grouped = grouped[grouped["FOLLOWING_COUNT"] >= 2]
    return grouped.sort_values(
        ["TRIGGER", "FOLLOWING_RATE", "FOLLOWING_COUNT"],
        ascending=[True, False, False],
    )


def explosive_sequence_analysis(df):
    """Summarize the immediate pre-play context for explosive plays."""
    if len(df) < 2:
        return pd.DataFrame()
    rows = []
    for i in range(1, len(df)):
        if not bool(df.iloc[i]["EXPLOSIVE"]):
            continue
        p, cur = df.iloc[i - 1], df.iloc[i]
        rows.append({
            "PREV_FORM": clean(p["OFF FORM"]),
            "PREV_MOTION": clean(p["MOTION"]),
            "PREV_PLAY_TYPE": clean(p["PLAY TYPE"]),
            "EXPLOSIVE_FORM": clean(cur["OFF FORM"]),
            "EXPLOSIVE_PLAY_TYPE": clean(cur["PLAY TYPE"]),
            "EXPLOSIVE_PLAY": clean(cur["OFF PLAY"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    keys = ["PREV_FORM", "PREV_MOTION", "PREV_PLAY_TYPE", "EXPLOSIVE_FORM", "EXPLOSIVE_PLAY_TYPE"]
    return out.groupby(keys).size().reset_index(name="EXPLOSIVE_COUNT").sort_values("EXPLOSIVE_COUNT", ascending=False)



def sequence_play_type(row):
    """Normalize a snap to RUN or PASS for factual pattern analysis."""
    category = classify_play(row)
    if category in {"RUN", "QB RUN"}:
        return "RUN"
    if category in {"PASS", "SACK"}:
        return "PASS"
    return ""


def normalize_direction(value):
    """Normalize play direction to LEFT/RIGHT when possible."""
    text = clean(value).upper()
    if not text:
        return ""
    if re.search(r"\b(LEFT|L|LT|LW|WEAK|W)\b", text):
        return "LEFT"
    if re.search(r"\b(RIGHT|R|RT|RW|STRONG|S)\b", text):
        return "RIGHT"
    return ""


def sequence_continuity(df, i, j, max_play_gap=2):
    """Require adjacent snaps in the same offensive series when possible."""
    if j != i + 1:
        return False

    a_odk = clean(df.iloc[i].get("ODK", ""))
    b_odk = clean(df.iloc[j].get("ODK", ""))
    if a_odk and b_odk and a_odk.upper() != b_odk.upper():
        return False

    a = numeric(df.iloc[i].get("PLAY #"))
    b = numeric(df.iloc[j].get("PLAY #"))
    if a is None or b is None:
        return False
    return 1 <= (b - a) <= max_play_gap


def _pattern_counts(df, value_getter, length):
    """Count all valid N-play patterns in actual consecutive offensive snaps."""
    if len(df) < length:
        return pd.DataFrame(columns=["PATTERN", "OCCURRENCES"])

    rows = []
    for i in range(len(df) - length + 1):
        if not all(sequence_continuity(df, i + offset, i + offset + 1) for offset in range(length - 1)):
            continue

        values = [value_getter(df.iloc[i + offset]) for offset in range(length)]
        if not all(values):
            continue

        rows.append({"PATTERN": " → ".join(values)})

    if not rows:
        return pd.DataFrame(columns=["PATTERN", "OCCURRENCES"])

    return (
        pd.DataFrame(rows)
        .groupby("PATTERN")
        .size()
        .reset_index(name="OCCURRENCES")
        .sort_values(["OCCURRENCES", "PATTERN"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _play_patterns(df, value_getter, length, min_occurrences=2):
    """Find repeated N-play patterns in actual consecutive offensive snaps."""
    grouped = _pattern_counts(df, value_getter, length)
    return grouped[grouped["OCCURRENCES"] >= min_occurrences].reset_index(drop=True)


def _side_by_side_patterns(df, value_getter, min_occurrences=2):
    """Return repeated 2-play and 3-play patterns with descriptive frequencies."""
    two_all = _pattern_counts(df, value_getter, 2)
    three_all = _pattern_counts(df, value_getter, 3)

    two = two_all[two_all["OCCURRENCES"] >= min_occurrences].copy()
    three = three_all[three_all["OCCURRENCES"] >= min_occurrences].copy()

    two_total = int(two_all["OCCURRENCES"].sum()) if not two_all.empty else 0
    three_total = int(three_all["OCCURRENCES"].sum()) if not three_all.empty else 0

    two["%"] = (two["OCCURRENCES"] / two_total * 100).round(1) if two_total else 0.0
    three["%"] = (three["OCCURRENCES"] / three_total * 100).round(1) if three_total else 0.0

    max_len = max(len(two), len(three), 1)
    two = two.reindex(range(max_len)).fillna("")
    three = three.reindex(range(max_len)).fillna("")

    return pd.DataFrame({
        "2-PLAY PATTERN": two["PATTERN"].tolist(),
        "2-PLAY OCCURRENCES": two["OCCURRENCES"].tolist(),
        "2-PLAY %": two["%"].tolist(),
        "3-PLAY PATTERN": three["PATTERN"].tolist(),
        "3-PLAY OCCURRENCES": three["OCCURRENCES"].tolist(),
        "3-PLAY %": three["%"].tolist(),
    }).fillna("")


def run_pass_sequences(df, min_occurrences=2):
    """Find repeated 2-play and 3-play RUN/PASS patterns with descriptive frequencies."""
    return _side_by_side_patterns(df, sequence_play_type, min_occurrences)


def left_right_sequences(df, min_occurrences=2):
    """Find repeated 2-play and 3-play LEFT/RIGHT patterns with descriptive frequencies."""
    return _side_by_side_patterns(
        df,
        lambda row: normalize_direction(row.get("PLAY DIR", "")),
        min_occurrences,
    )


def repeated_play_sequences(df, min_occurrences=2, max_play_gap=2):
    """Find repeated exact play sequences using real play order.

    A sequence is only considered a sequence when the plays belong to the same
    offensive series (when ODK is available) and PLAY # values are close.
    PLAY # gaps of 1 are directly consecutive; a gap of 2 allows one
    bookkeeping/event row between snaps. Larger gaps are excluded.

    The report intentionally contains only information that is useful on the
    scouting sheet: the sequence, how often it occurred, where it occurred,
    the starting context, and what followed the sequence.
    """
    if len(df) < 2:
        return pd.DataFrame()

    work = df.copy()
    work["_PLAY_NUM"] = work["PLAY #"].map(numeric)
    labels = [normalize_favorite_play(play_label(work.iloc[i])) for i in range(len(work))]
    rows = []

    def same_offensive_series(i, j):
        a, b = clean(work.iloc[i].get("ODK", "")), clean(work.iloc[j].get("ODK", ""))
        return not a or not b or a.upper() == b.upper()

    def is_contiguous(i, j):
        if not same_offensive_series(i, j):
            return False
        a, b = work.iloc[i]["_PLAY_NUM"], work.iloc[j]["_PLAY_NUM"]
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            return False
        return 1 <= (b - a) <= max_play_gap

    def context(row):
        parts = []
        situation = situation_bucket(row.get("DN", ""), row.get("DIST", ""))
        if situation:
            parts.append(situation)
        for col in ("PERSONNEL", "OFF FORM", "MOTION"):
            value = clean(row.get(col, ""))
            if value:
                parts.append(value)
        return " | ".join(parts) if parts else "—"

    for length in (2, 3):
        if len(labels) < length:
            continue

        counts = {}
        examples = {}

        for i in range(len(labels) - length + 1):
            end = i + length - 1
            if not all(is_contiguous(k, k + 1) for k in range(i, end)):
                continue

            sequence = tuple(labels[i:i + length])
            if any(not x or x == "UNKNOWN" for x in sequence):
                continue

            key = sequence
            counts[key] = counts.get(key, 0) + 1
            examples.setdefault(key, [])

            play_numbers = "→".join(
                str(int(work.iloc[k]["_PLAY_NUM"])) for k in range(i, end + 1)
            )
            gaps = [
                int(work.iloc[k + 1]["_PLAY_NUM"] - work.iloc[k]["_PLAY_NUM"])
                for k in range(i, end)
            ]

            next_play = None
            if end + 1 < len(work) and is_contiguous(end, end + 1):
                candidate = normalize_favorite_play(play_label(work.iloc[end + 1]))
                if candidate and candidate != "UNKNOWN":
                    next_play = candidate

            examples[key].append({
                "PLAY_NUMBERS": play_numbers,
                "CONTEXT": context(work.iloc[i]),
                "NEXT_PLAY": next_play,
                "MAX_PLAY_GAP": max(gaps),
            })

        for sequence, count in counts.items():
            if count < min_occurrences:
                continue

            details = examples[sequence]
            followups = [d["NEXT_PLAY"] for d in details if d["NEXT_PLAY"]]
            followup_text = "—"
            if followups:
                followup_counts = pd.Series(followups).value_counts()
                top = followup_counts.index[0]
                top_count = int(followup_counts.iloc[0])
                rate = round(top_count / len(followups) * 100, 1)
                followup_text = f"{top} ({top_count}/{len(followups)}, {rate}%)"

            contexts = []
            for detail in details:
                if detail["CONTEXT"] not in contexts:
                    contexts.append(detail["CONTEXT"])

            rows.append({
                "SEQUENCE": " → ".join(sequence),
                "OCCURRENCES": count,
                "PLAY_NUMBERS": "; ".join(d["PLAY_NUMBERS"] for d in details[:10]),
                "CONTEXT": "; ".join(contexts[:5]),
                "FOLLOW-UP / RESULT": followup_text,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "SEQUENCE", "OCCURRENCES", "PLAY_NUMBERS", "CONTEXT", "FOLLOW-UP / RESULT"
        ])

    return pd.DataFrame(rows).sort_values(
        ["OCCURRENCES", "SEQUENCE"],
        ascending=[False, True],
    ).reset_index(drop=True)


def repeated_play_followups(df, min_occurrences=2):
    """Find what follows a play repeated two or three times consecutively."""
    if len(df) < 3:
        return pd.DataFrame()

    labels = [normalize_favorite_play(play_label(df.iloc[i])) for i in range(len(df))]
    rows = []

    for run_length in (2, 3):
        for i in range(len(labels) - run_length):
            repeated = labels[i:i + run_length]
            if not repeated or any(not x or x == "UNKNOWN" for x in repeated):
                continue
            if len(set(repeated)) != 1:
                continue

            next_play = labels[i + run_length]
            if not next_play or next_play == "UNKNOWN":
                continue

            rows.append({
                "REPEATED_PLAY": repeated[0],
                "TIMES_IN_ROW": run_length,
                "FOLLOWING_PLAY": next_play,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "REPEATED_PLAY", "TIMES_IN_ROW", "FOLLOWING_PLAY",
            "FOLLOWING_COUNT", "TOTAL_AFTER_REPEAT", "FOLLOWING_RATE"
        ])

    grouped = (
        out.groupby(["REPEATED_PLAY", "TIMES_IN_ROW", "FOLLOWING_PLAY"])
        .size()
        .reset_index(name="FOLLOWING_COUNT")
    )
    totals = grouped.groupby(["REPEATED_PLAY", "TIMES_IN_ROW"])["FOLLOWING_COUNT"].transform("sum")
    grouped["TOTAL_AFTER_REPEAT"] = totals
    grouped["FOLLOWING_RATE"] = (grouped["FOLLOWING_COUNT"] / totals * 100).round(1)
    return grouped[grouped["FOLLOWING_COUNT"] >= min_occurrences].sort_values(
        ["TIMES_IN_ROW", "FOLLOWING_RATE", "FOLLOWING_COUNT"],
        ascending=[True, False, False],
    )


def passing_target_summary(df, min_occurrences=1):
    """Summarize passing targets from BALL CARRIER with receiving yards."""
    rows = []
    for _, row in df.iterrows():
        result = clean(row.get("RESULT", "")).upper()
        if "INCOMPLETE" not in result and "COMPLETE" not in result:
            continue
        target = clean(row.get("BALL CARRIER", ""))
        if not target:
            continue
        yards = numeric(row.get("GN/LS"))
        if yards is None:
            yards = result_yards(row.get("RESULT")) or 0
        rows.append({"TARGET": target, "YARDS": yards})

    columns = ["TARGET", "COUNT", "YARDS"]
    if not rows:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(rows)
    summary = (
        work.groupby("TARGET")
        .agg(COUNT=("TARGET", "size"), YARDS=("YARDS", "sum"))
        .reset_index()
    )
    return (
        summary[summary["COUNT"] >= min_occurrences]
        .sort_values(["COUNT", "YARDS", "TARGET"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def rushing_tendencies(df, min_occurrences=1):
    """Summarize rushing ball carriers with attempts and rushing yards."""
    carrier = ball_carrier_identity(df, min_occurrences=1)
    if carrier.empty:
        return pd.DataFrame(columns=["BALL CARRIER", "COUNT", "YARDS"])

    rush = carrier[carrier["ROLE"] == "RUSH"].copy()
    if rush.empty:
        return pd.DataFrame(columns=["BALL CARRIER", "COUNT", "YARDS"])

    rush = (
        rush.groupby("BALL CARRIER", as_index=False)
        .agg(COUNT=("PLAYS", "sum"), YARDS=("YARDS", "sum"))
    )
    return (
        rush[rush["COUNT"] >= min_occurrences]
        .sort_values(["COUNT", "YARDS", "BALL CARRIER"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def ball_carrier_identity(df, min_occurrences=1):
    """Summarize the actual offensive player recorded in BALL CARRIER."""
    rows = []
    for _, row in df.iterrows():
        carrier = clean(row.get("BALL CARRIER", ""))
        if not carrier:
            continue

        category = classify_play(row)
        yards = numeric(row.get("GN/LS"))
        if yards is None:
            yards = result_yards(row.get("RESULT")) or 0

        result = clean(row.get("RESULT", "")).upper()
        is_completion = "INCOMPLETE" not in result and "COMPLETE" in result
        is_rushing = category in {"RUN", "QB RUN"}
        is_receiving = category == "PASS" and is_completion

        if not (is_rushing or is_receiving):
            continue

        rows.append({
            "BALL CARRIER": carrier,
            "ROLE": "RUSH" if is_rushing else "RECEIVE",
            "PLAYS": 1,
            "YARDS": yards,
            "TD": int(bool(re.search(r"TOUCHDOWN|\\bTD\\b", result))),
            "EXPLOSIVE": int(is_explosive(row)),
            "PLAY_NUMBER": clean(row.get("PLAY #", "")),
        })

    columns = [
        "BALL CARRIER", "ROLE", "PLAYS", "YARDS", "AVG YDS",
        "TD", "EXPLOSIVES", "PLAY_NUMBERS",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(rows)
    summary = (
        work.groupby(["BALL CARRIER", "ROLE"])
        .agg(
            PLAYS=("PLAYS", "sum"),
            YARDS=("YARDS", "sum"),
            TD=("TD", "sum"),
            EXPLOSIVES=("EXPLOSIVE", "sum"),
            PLAY_NUMBERS=("PLAY_NUMBER", lambda s: ", ".join(s.astype(str))),
        )
        .reset_index()
    )
    summary["AVG YDS"] = (summary["YARDS"] / summary["PLAYS"]).round(1)
    summary = summary[summary["PLAYS"] >= min_occurrences]
    return summary.sort_values(
        ["PLAYS", "YARDS", "BALL CARRIER"],
        ascending=[False, False, True],
    ).loc[:, columns].reset_index(drop=True)


def completed_comment_patterns(df, min_occurrences=1, target_source="comments"):
    """Build passing target tendencies from the selected target source.

    Tendencies/ALL INFO SHEET uses COMMENTS (for example, #8) as the target.
    OFF SELF SCOUT uses BALL CARRIER because that sheet records the actual
    offensive player.
    """
    rows = []

    for _, row in df.iterrows():
        result = clean(row.get("RESULT", "")).upper()
        if "INCOMPLETE" not in result and "COMPLETE" not in result:
            continue

        play = play_name(row)
        formation = clean(row.get("OFF FORM", ""))

        if target_source == "ball_carrier":
            target = clean(row.get("BALL CARRIER", ""))
        else:
            comment = clean(row.get("COMMENTS", ""))
            targets = re.findall(r"#\s*(\d{1,2})", comment)
            target = targets[0] if targets else ""

        if not play or not formation or not target:
            continue

        rows.append({
            "PLAY": play,
            "FORMATION": formation,
            "TARGET": target,
            "PLAY_NUMBER": clean(row.get("PLAY #", "")),
        })

    columns = [
        "PLAY",
        "FORMATION",
        "TARGET",
        "COUNT",
        "PLAY_NUMBERS",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(rows)

    summary = (
        work.groupby(["PLAY", "FORMATION", "TARGET"])
        .agg(
            COUNT=("PLAY_NUMBER", "size"),
            PLAY_NUMBERS=("PLAY_NUMBER", lambda s: ", ".join(
                s.astype(str)
            )),
        )
        .reset_index()
    )

    summary = summary[summary["COUNT"] >= min_occurrences]

    return summary.sort_values(
        ["PLAY", "FORMATION", "COUNT", "TARGET"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

def overall_summary(df):
    """Compact top-of-sheet game summary."""
    work = df.copy()
    work["_CLASS"] = work.apply(classify_play, axis=1)
    work["_YARDS"] = work.apply(
        lambda r: numeric(r.get("GN/LS")) if numeric(r.get("GN/LS")) is not None
        else (result_yards(r.get("RESULT")) or 0), axis=1
    )
    run_mask = work["_CLASS"].isin({"RUN", "QB RUN"})
    pass_mask = work["_CLASS"] == "PASS"
    sack_mask = work["_CLASS"] == "SACK"
    blob = work.apply(play_blob, axis=1)

    td_mask = blob.str.contains(r"TOUCHDOWN|\bTD\b", regex=True)
    rush_td_mask = run_mask & td_mask
    pass_td_mask = pass_mask & td_mask
    fumbles = int(blob.str.contains(r"FUMBLE|FUMBLED", regex=True).sum())
    interceptions = int(blob.str.contains(r"INTERCEPTION|\bINT\b", regex=True).sum())
    pass_completions = int(
        (pass_mask & blob.str.contains(r"(?<!IN)\bCOMPLETE(?:D)?\b", regex=True)).sum()
    )

    return {
        "TOTAL PLAYS": int(len(work)),
        "RUN ATT": int(run_mask.sum()),
        "RUN YDS": round(float(work.loc[run_mask, "_YARDS"].sum()), 1),
        "PASS ATT": int(pass_mask.sum()),
        "PASS COMP": pass_completions,
        "PASS YDS": round(float(work.loc[pass_mask, "_YARDS"].sum()), 1),
        "SACK": int(sack_mask.sum()),
        "INT": interceptions,
        "FUMBLES": fumbles,
        "TD": int(td_mask.sum()),
        "RUSH TD": int(rush_td_mask.sum()),
        "PASS TD": int(pass_td_mask.sum()),
    }

def frequency_profile(df, column, min_plays=2, top_n=15):
    """Frequency plus run/pass usage and yard production for one analytical lens."""
    work = df[df[column].map(clean) != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=[
            column, "PLAYS", "FREQUENCY_RATE", "RUN", "PASS",
            "RUN_YARDS", "PASS_YARDS", "TOTAL_YARDS", "YARDS_PER_PLAY"
        ])

    work["_CLASS"] = work.apply(classify_play, axis=1)
    work["_YARDS"] = work.apply(
        lambda r: numeric(r.get("GN/LS")) if numeric(r.get("GN/LS")) is not None
        else (result_yards(r.get("RESULT")) or 0), axis=1
    )
    work["_RUN_YARDS"] = work.apply(
        lambda r: r["_YARDS"] if r["_CLASS"] in {"RUN", "QB RUN"} else 0, axis=1
    )
    work["_PASS_YARDS"] = work.apply(
        lambda r: r["_YARDS"] if r["_CLASS"] == "PASS" else 0, axis=1
    )
    out = work.groupby(column).agg(
        PLAYS=("_CLASS", "size"),
        RUN=("_CLASS", lambda s: int((s == "RUN").sum())),
        PASS=("_CLASS", lambda s: int((s == "PASS").sum())),
        RUN_YARDS=("_RUN_YARDS", "sum"),
        PASS_YARDS=("_PASS_YARDS", "sum"),
        TOTAL_YARDS=("_YARDS", "sum"),
    ).reset_index()
    out["FREQUENCY_RATE"] = (out["PLAYS"] / len(work) * 100).round(1)
    out["YARDS_PER_PLAY"] = (out["TOTAL_YARDS"] / out["PLAYS"]).round(1)
    return out[out["PLAYS"] >= min_plays].sort_values(
        ["PLAYS", "FREQUENCY_RATE"], ascending=False
    ).head(top_n).reset_index(drop=True)


def run_pass_yard_summary(df):
    """Overall run/pass/QB-run/sack production."""
    work = df.copy()
    work["_CLASS"] = work.apply(classify_play, axis=1)
    work["_YARDS"] = work.apply(
        lambda r: numeric(r.get("GN/LS")) if numeric(r.get("GN/LS")) is not None
        else (result_yards(r.get("RESULT")) or 0), axis=1
    )
    rows = []
    for label in ("RUN", "QB RUN", "PASS", "SACK"):
        mask = work["_CLASS"] == label
        plays = int(mask.sum())
        yards = float(work.loc[mask, "_YARDS"].sum())
        rows.append({
            "PLAY_CATEGORY": label,
            "PLAYS": plays,
            "YARDS": round(yards, 1),
            "YARDS_PER_PLAY": round(yards / plays, 1) if plays else 0,
        })
    return pd.DataFrame(rows)


def frequency_tendencies(df, column, min_plays=2, top_n=15):
    """Describe the most frequently used values without filtering for explosives."""
    work = df[df[column].map(clean) != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=[column, "PLAYS", "FREQUENCY_RATE"])
    out = work.groupby(column).size().reset_index(name="PLAYS")
    out["FREQUENCY_RATE"] = (out["PLAYS"] / len(work) * 100).round(1)
    return out[out["PLAYS"] >= min_plays].sort_values(
        ["PLAYS", "FREQUENCY_RATE"], ascending=False
    ).head(top_n).reset_index(drop=True)


def situation_bucket(down, dist):
    """Return the same coach-friendly down/distance buckets used elsewhere."""
    down = numeric(down)
    dist = numeric(dist)
    if down is None or dist is None:
        return ""
    down, dist = int(down), max(0, dist)
    if down == 1:
        return "1st & Long (10+)" if dist >= 10 else "1st & Short (1-9)"
    if down == 2:
        return "2nd & Long (7+)" if dist >= 7 else ("2nd & Medium (4-6)" if dist >= 4 else "2nd & Short (1-3)")
    if down == 3:
        return "3rd & Long (7+)" if dist >= 7 else ("3rd & Medium (4-6)" if dist >= 4 else "3rd & Short (1-3)")
    if down == 4:
        return "4th Down"
    return ""



def field_zone(value):
    """Map YARD LN to coach-friendly field zones.
    
    Supports the sheet's signed convention: negative = own territory,
    positive = opponent territory, with 50 representing midfield.
    """
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
    if raw == 50:
        return "MIDFIELD"
    if raw >= 41:
        return "MIDFIELD"
    if raw >= 21:
        return "OPP 21-40"
    if raw >= 1:
        return "RED ZONE"
    return "MIDFIELD"



def normalize_favorite_play(value):
    """Combine simple left/right variants into one base play name."""
    text = clean(value)
    return re.sub(r"\s+(?:L|R)$", "", text, flags=re.IGNORECASE).strip()

def field_zone_efficiency(df, top_n=3):
    """Measure play volume and production within each field zone."""
    rows = []

    for _, row in df.iterrows():
        zone = field_zone(row.get("YARD LN"))
        if not zone:
            continue

        yards = numeric(row.get("GN/LS"))
        if yards is None:
            yards = result_yards(row.get("RESULT")) or 0

        rows.append({
            "FIELD ZONE": zone,
            "PLAY": normalize_favorite_play(play_label(row)),
            "PLAYS": 1,
            "RUN": int(is_run(row)),
            "PASS": int(is_pass(row)),
            "YARDS": yards,
        })

    columns = [
        "FIELD ZONE", "PLAYS", "RUN %", "PASS %", "PLAY MIX",
        "YARDS", "YARDS/PLAY", "PLAY 1", "PLAY 2", "PLAY 3",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(rows)

    out = (
        work.groupby("FIELD ZONE")
        .agg(
            PLAYS=("PLAYS", "sum"),
            RUN=("RUN", "sum"),
            PASS=("PASS", "sum"),
            YARDS=("YARDS", "sum"),
        )
        .reset_index()
    )

    play_counts = (
        work[work["PLAY"].map(clean) != ""]
        .groupby(["FIELD ZONE", "PLAY"])
        .size()
        .reset_index(name="PLAY_COUNT")
        .sort_values(
            ["FIELD ZONE", "PLAY_COUNT", "PLAY"],
            ascending=[True, False, True],
        )
    )

    favorite_rows = []
    for zone, group in play_counts.groupby("FIELD ZONE", sort=False):
        favorites = [clean(play) for play in group["PLAY"].head(top_n)]
        favorite_rows.append({
            "FIELD ZONE": zone,
            "PLAY 1": favorites[0] if len(favorites) > 0 else "",
            "PLAY 2": favorites[1] if len(favorites) > 1 else "",
            "PLAY 3": favorites[2] if len(favorites) > 2 else "",
        })

    favorites_df = pd.DataFrame(favorite_rows)
    out = out.merge(favorites_df, on="FIELD ZONE", how="left")

    out["RUN %"] = (out["RUN"] / out["PLAYS"] * 100).round(1)
    out["PASS %"] = (out["PASS"] / out["PLAYS"] * 100).round(1)
    out["PLAY MIX"] = out.apply(
        lambda r: f"{int(r['PLAYS'])} Plays {r['RUN %']:.0f}% Run {r['PASS %']:.0f}% Pass",
        axis=1,
    )
    out["YARDS/PLAY"] = (out["YARDS"] / out["PLAYS"]).round(1)

    order = {
        "OWN 1-20": 1,
        "OWN 21-40": 2,
        "OWN 41-49": 3,
        "MIDFIELD": 4,
        "OPP 21-40": 5,
        "RED ZONE": 6,
    }
    out["_ORDER"] = out["FIELD ZONE"].map(order).fillna(99)

    return (
        out.sort_values("_ORDER")
        .drop(columns=["_ORDER", "RUN", "PASS"])
        .loc[:, columns]
        .reset_index(drop=True)
    )

def field_zone_by_hash(df, top_n=3):
    """Measure play volume/production within each field zone + hash."""
    rows = []

    for _, row in df.iterrows():
        zone = field_zone(row.get("YARD LN"))
        hash_value = clean(row.get("HASH", ""))

        if not zone or not hash_value:
            continue

        yards = numeric(row.get("GN/LS"))
        if yards is None:
            yards = result_yards(row.get("RESULT")) or 0

        rows.append({
            "FIELD ZONE": zone,
            "HASH": hash_value,
            "PLAY": normalize_favorite_play(play_label(row)),
            "PLAYS": 1,
            "RUN": int(is_run(row)),
            "PASS": int(is_pass(row)),
            "YARDS": yards,
        })

    columns = [
        "FIELD ZONE", "HASH", "PLAYS", "YARDS",
        "RUN %", "PASS %", "PLAY MIX", "YARDS/PLAY", "PLAY 1", "PLAY 2", "PLAY 3",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(rows)
    summary = (
        work.groupby(["FIELD ZONE", "HASH"])
        .agg(
            PLAYS=("PLAYS", "sum"),
            RUN=("RUN", "sum"),
            PASS=("PASS", "sum"),
            YARDS=("YARDS", "sum"),
        )
        .reset_index()
    )

    play_counts = (
        work[work["PLAY"].map(clean) != ""]
        .groupby(["FIELD ZONE", "HASH", "PLAY"])
        .size()
        .reset_index(name="PLAY_COUNT")
        .sort_values(
            ["FIELD ZONE", "HASH", "PLAY_COUNT", "PLAY"],
            ascending=[True, True, False, True],
        )
    )

    favorite_rows = []
    for (zone, hash_value), group in play_counts.groupby(
        ["FIELD ZONE", "HASH"],
        sort=False,
    ):
        favorites = [clean(play) for play in group["PLAY"].head(top_n)]
        favorite_rows.append({
            "FIELD ZONE": zone,
            "HASH": hash_value,
            "PLAY 1": favorites[0] if len(favorites) > 0 else "",
            "PLAY 2": favorites[1] if len(favorites) > 1 else "",
            "PLAY 3": favorites[2] if len(favorites) > 2 else "",
        })

    favorites_df = pd.DataFrame(favorite_rows)

    out = summary.merge(
        favorites_df,
        on=["FIELD ZONE", "HASH"],
        how="left",
    )

    out["RUN %"] = (out["RUN"] / out["PLAYS"] * 100).round(1)
    out["PASS %"] = (out["PASS"] / out["PLAYS"] * 100).round(1)
    out["PLAY MIX"] = out.apply(
        lambda r: f"{int(r['PLAYS'])} Plays {r['RUN %']:.0f}% Run {r['PASS %']:.0f}% Pass",
        axis=1,
    )
    out["YARDS/PLAY"] = (out["YARDS"] / out["PLAYS"]).round(1)

    order = {
        "OWN 1-20": 1,
        "OWN 21-40": 2,
        "OWN 41-49": 3,
        "MIDFIELD": 4,
        "OPP 21-40": 5,
        "RED ZONE": 6,
    }
    out["_ORDER"] = out["FIELD ZONE"].map(order).fillna(99)

    return (
        out.sort_values(
            ["_ORDER", "HASH"],
            ascending=[True, True],
        )
        .drop(columns=["_ORDER", "RUN", "PASS"])
        .loc[:, columns]
        .reset_index(drop=True)
    )

def down_efficiency(df):
    """Measure first-down conversion efficiency on every down."""
    rows = []
    work = df.reset_index(drop=True)
    for i, row in work.iterrows():
        down = numeric(row.get("DN"))
        if down is None or int(down) not in {1, 2, 3, 4}:
            continue
        down = int(down)
        dist = numeric(row.get("DIST"))
        gain = numeric(row.get("GN/LS"))
        if gain is None:
            gain = result_yards(row.get("RESULT"))
        converted = bool(dist is not None and gain is not None and gain >= dist)
        if not converted:
            for j in range(i + 1, len(work)):
                next_down = numeric(work.iloc[j].get("DN"))
                if next_down is None:
                    continue
                if int(next_down) in {1, 2, 3, 4}:
                    converted = int(next_down) == 1
                    break
        suffix = {1: "st", 2: "nd", 3: "rd", 4: "th"}[down]
        rows.append({"DOWN": f"{down}{suffix} Down", "ATTEMPTS": 1, "CONVERSIONS": int(converted)})
    if not rows:
        return pd.DataFrame(columns=["DOWN", "ATTEMPTS", "CONVERSIONS", "CONVERSION RATE"])
    out = pd.DataFrame(rows).groupby("DOWN").agg(
        ATTEMPTS=("ATTEMPTS", "sum"), CONVERSIONS=("CONVERSIONS", "sum")
    ).reset_index()
    out["CONVERSION RATE"] = (out["CONVERSIONS"] / out["ATTEMPTS"] * 100).round(1)
    order = {"1st Down": 1, "2nd Down": 2, "3rd Down": 3, "4th Down": 4}
    out["_ORDER"] = out["DOWN"].map(order)
    return out.sort_values("_ORDER").drop(columns="_ORDER").reset_index(drop=True)


def third_down_efficiency(df):
    """Backward-compatible 3rd-down view."""
    out = down_efficiency(df)
    if out.empty:
        return pd.DataFrame(columns=["3RD DOWN", "ATTEMPTS", "CONVERSIONS", "CONVERSION RATE"])
    return out[out["DOWN"] == "3rd Down"].rename(columns={"DOWN": "3RD DOWN"}).reset_index(drop=True)


def three_and_out_analysis(df):
    """Identify 3-and-outs using play-number series and actual snaps.

    PLAY # is treated as the series/drive marker with its final digit being
    the snap within that series. For example, 50-51-52 followed by 61 is a
    3-and-out; 50-51-52 followed by 53 is not.

    Administrative rows such as timeouts, scoreboard/clock entries, and
    penalty-only entries are ignored. A penalty attached to an actual play
    still counts as a snap, because the play occurred.
    """
    if df.empty:
        return pd.DataFrame([{
            "3-AND-OUTS": 0,
            "3RD-DOWN DRIVES": 0,
            "3-AND-OUT RATE": 0.0,
        }])

    work = df.copy().reset_index(drop=True)
    work["_PLAY_NUM"] = work["PLAY #"].map(numeric)

    def series_key(play_num):
        if play_num is None or pd.isna(play_num):
            return None
        value = int(play_num)
        return str(value)[:-1] if abs(value) >= 10 else ""

    def is_administrative(row):
        blob = play_blob(row)
        admin_terms = [
            "TIMEOUT", "SCOREBOARD", "CLOCK", "QUARTER", "HALFTIME",
            "END OF HALF", "END OF GAME", "KNEEL", "NO PLAY"
        ]
        if any(term in blob for term in admin_terms):
            return True

        # A penalty-only entry is bookkeeping rather than an offensive snap.
        # If it also contains a recognizable run/pass/sack, keep it as a snap.
        if is_penalty(row) and classify_play(row) == "OTHER":
            return True

        return False

    # Keep only actual offensive snaps with a valid down. This allows timeout,
    # scoreboard, and similar rows to sit between snaps without affecting
    # the drive calculation.
    snaps = []
    for _, row in work.iterrows():
        down = numeric(row.get("DN"))
        if down is None or int(down) not in {1, 2, 3, 4}:
            continue
        if is_administrative(row):
            continue

        play_num = numeric(row.get("_PLAY_NUM"))
        key = series_key(play_num)
        if key is None:
            continue

        snaps.append({
            "PLAY_NUM": int(play_num),
            "SERIES": key,
            "DOWN": int(down),
        })

    if not snaps:
        return pd.DataFrame([{
            "3-AND-OUTS": 0,
            "3RD-DOWN DRIVES": 0,
            "3-AND-OUT RATE": 0.0,
        }])

    snap_df = pd.DataFrame(snaps)
    drives = []

    # Each PLAY # series is a drive/series in the coach's numbering convention.
    for series, segment in snap_df.groupby("SERIES", sort=False):
        segment = segment.reset_index(drop=True)
        downs = segment["DOWN"].tolist()

        if downs[:3] != [1, 2, 3]:
            continue

        # Any fourth snap in the same series means the offense continued the
        # possession, so it was not a 3-and-out.
        is_three_and_out = len(segment) == 3

        drives.append({
            "PLAY NUMBERS": " → ".join(str(x) for x in segment["PLAY_NUM"].iloc[:3]),
            "3-AND-OUT": int(is_three_and_out),
        })

    if not drives:
        return pd.DataFrame([{
            "3-AND-OUTS": 0,
            "3RD-DOWN DRIVES": 0,
            "3-AND-OUT RATE": 0.0,
        }])

    drive_df = pd.DataFrame(drives)
    third_down_drives = len(drive_df)
    outs = int(drive_df["3-AND-OUT"].sum())

    return pd.DataFrame([{
        "3-AND-OUTS": outs,
        "3RD-DOWN DRIVES": third_down_drives,
        "3-AND-OUT RATE": round(outs / third_down_drives * 100, 1),
    }])

def play_result_type(row):
    """Normalize the result into a small, factual sequence category."""
    blob = play_blob(row)
    if is_penalty(row):
        return "PENALTY"
    if "INTERCEPTION" in blob or "INT" in blob:
        return "INTERCEPTION"
    if "INCOMPLETE" in blob:
        return "INCOMPLETE PASS"
    if is_pass(row):
        return "COMPLETION/PASS"
    if is_run(row):
        if "SCRAMBLE" in blob or "QB RUN" in blob or "KEEP" in blob:
            return "QB RUN"
        return "RUN"
    return clean(row.get("PLAY TYPE")) or "OTHER"



def situation_play_calling_patterns(df, min_occurrences=2, top_n=3):
    """Show observed play-calling mix within each down/distance situation.

    This is descriptive only: it reports what was called in the supplied
    sample and does not make predictions or recommendations.
    """
    rows = []

    for _, row in df.iterrows():
        situation = situation_bucket(row.get("DN", ""), row.get("DIST", ""))
        if not situation:
            continue

        play_type = sequence_play_type(row)
        play = normalize_favorite_play(play_label(row))

        if not play_type:
            continue

        rows.append({
            "DOWN & DISTANCE": situation,
            "PLAY_TYPE": play_type,
            "PLAY": play if play and play != "UNKNOWN" else "",
        })

    columns = [
        "DOWN & DISTANCE", "PLAYS",
        "RUN", "RUN %", "PASS", "PASS %",
        "PLAY 1", "PLAY 2", "PLAY 3",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    work = pd.DataFrame(rows)

    summary_rows = []
    for situation, group in work.groupby("DOWN & DISTANCE"):
        total = len(group)
        if total < min_occurrences:
            continue

        run_count = int((group["PLAY_TYPE"] == "RUN").sum())
        pass_count = int((group["PLAY_TYPE"] == "PASS").sum())
        play_counts = (
            group[group["PLAY"] != ""]
            .groupby("PLAY")
            .size()
            .sort_values(ascending=False)
            .head(top_n)
        )

        favorites = [
            f"{play} ({count}/{total}, {round(count / total * 100, 1)}%)"
            for play, count in play_counts.items()
        ]
        favorites += [""] * (top_n - len(favorites))

        summary_rows.append({
            "DOWN & DISTANCE": situation,
            "PLAYS": total,
            "RUN": run_count,
            "RUN %": round(run_count / total * 100, 1),
            "PASS": pass_count,
            "PASS %": round(pass_count / total * 100, 1),
            "PLAY 1": favorites[0],
            "PLAY 2": favorites[1],
            "PLAY 3": favorites[2],
        })

    if not summary_rows:
        return pd.DataFrame(columns=columns)

    order = {
        "1st & Long (10+)": 1,
        "1st & Short (1-9)": 2,
        "2nd & Long (7+)": 3,
        "2nd & Medium (4-6)": 4,
        "2nd & Short (1-3)": 5,
        "3rd & Long (7+)": 6,
        "3rd & Medium (4-6)": 7,
        "3rd & Short (1-3)": 8,
        "4th Down": 9,
    }

    out = pd.DataFrame(summary_rows)
    out["_ORDER"] = out["DOWN & DISTANCE"].map(order).fillna(99)

    return (
        out.sort_values(["_ORDER", "PLAYS"], ascending=[True, False])
        .drop(columns="_ORDER")
        .loc[:, columns]
        .reset_index(drop=True)
    )


def situation_sequence_analysis(df, min_occurrences=2, top_n=10):
    """Find what follows a specific situation/result combination.

    Results are ordered first by the previous situation, then by the next
    down-and-distance situation.
    """
    if len(df) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(df) - 1):
        current, following = df.iloc[i], df.iloc[i + 1]
        situation = situation_bucket(current["DN"], current["DIST"])
        if not situation:
            continue

        next_play_type = (
            "PASS" if is_pass(following)
            else "RUN" if is_run(following)
            else ""
        )
        if not next_play_type:
            continue

        rows.append({
            "PREVIOUS_SITUATION": situation,
            "PREVIOUS_RESULT": play_result_type(current),
            "NEXT_DOWN & DISTANCE": situation_bucket(
                following["DN"], following["DIST"]
            ),
            "NEXT_PLAY_TYPE": next_play_type,
            "NEXT_SCHEME": clean(following.get("OFF PLAY", "")),
        })

    if not rows:
        return pd.DataFrame()

    work = pd.DataFrame(rows)
    grouped = (
        work.groupby([
            "PREVIOUS_SITUATION",
            "PREVIOUS_RESULT",
            "NEXT_DOWN & DISTANCE",
            "NEXT_PLAY_TYPE",
            "NEXT_SCHEME",
        ])
        .size()
        .reset_index(name="FOLLOWING_COUNT")
    )

    grouped = grouped[grouped["FOLLOWING_COUNT"] >= min_occurrences]

    situation_order = {
        "1st & Long (10+)": 1,
        "1st & Short (1-9)": 2,
        "2nd & Long (7+)": 3,
        "2nd & Medium (4-6)": 4,
        "2nd & Short (1-3)": 5,
        "3rd & Long (7+)": 6,
        "3rd & Medium (4-6)": 7,
        "3rd & Short (1-3)": 8,
        "4th Down": 9,
    }

    grouped["_SITUATION_ORDER"] = grouped["PREVIOUS_SITUATION"].map(situation_order).fillna(99)
    grouped["_NEXT_DOWN_ORDER"] = grouped["NEXT_DOWN & DISTANCE"].map(situation_order).fillna(99)

    return (
        grouped.sort_values(
            [
                "_SITUATION_ORDER",
                "_NEXT_DOWN_ORDER",
                "PREVIOUS_RESULT",
                "FOLLOWING_COUNT",
                "NEXT_PLAY_TYPE",
                "NEXT_SCHEME",
            ],
            ascending=[True, True, True, False, True, True],
        )
        .groupby(
            ["PREVIOUS_SITUATION", "PREVIOUS_RESULT"],
            group_keys=False,
        )
        .head(top_n)
        .drop(columns=["_SITUATION_ORDER", "_NEXT_DOWN_ORDER"])
        .loc[:, [
            "PREVIOUS_SITUATION",
            "PREVIOUS_RESULT",
            "NEXT_DOWN & DISTANCE",
            "NEXT_PLAY_TYPE",
            "NEXT_SCHEME",
            "FOLLOWING_COUNT",
        ]]
        .reset_index(drop=True)
    )

def general_play_type_frequency(df):
    """Overall run/pass/QB-run frequency using the same classifier as explosives."""
    rows = []
    for _, row in df.iterrows():
        if is_pass(row):
            category = "PASS"
        elif is_run(row):
            category = "QB RUN" if any(
                x in play_blob(row) for x in ["SCRAMBLE", "QB RUN", "KEEP"]
            ) else "RUN"
        else:
            category = "OTHER"
        rows.append(category)
    out = pd.Series(rows, name="PLAY_TYPE").value_counts().reset_index()
    out.columns = ["PLAY_TYPE", "PLAYS"]
    out["FREQUENCY_RATE"] = (out["PLAYS"] / len(df) * 100).round(1) if len(df) else 0
    return out


@dataclass
class TendencyReport:
    total_plays: int
    explosive_plays: int
    explosive_detail: pd.DataFrame
    explosive_by_formation_situation: pd.DataFrame
    explosive_by_hash_direction: pd.DataFrame
    by_motion: pd.DataFrame
    prior_play_features: pd.DataFrame
    sequences: pd.DataFrame
    trigger_sequences: pd.DataFrame
    repeated_sequences: pd.DataFrame
    repeated_followups: pd.DataFrame
    run_pass_sequences: pd.DataFrame
    left_right_sequences: pd.DataFrame
    frequency_by_formation: pd.DataFrame
    frequency_by_situation: pd.DataFrame
    hash_down_distance_play_probabilities: pd.DataFrame
    play_type_frequency: pd.DataFrame
    situation_sequences: pd.DataFrame
    situation_play_calling: pd.DataFrame
    field_zone_efficiency: pd.DataFrame
    field_zone_by_hash: pd.DataFrame
    third_down_efficiency: pd.DataFrame
    three_and_out_analysis: pd.DataFrame
    frequency_by_personnel: pd.DataFrame
    frequency_by_backfield: pd.DataFrame
    frequency_by_motion: pd.DataFrame
    frequency_by_scheme: pd.DataFrame
    frequency_by_direction: pd.DataFrame
    run_pass_yards: pd.DataFrame
    completed_comment_patterns: pd.DataFrame
    ball_carrier_identity: pd.DataFrame
    passing_target_summary: pd.DataFrame
    rushing_tendencies: pd.DataFrame
    formation_frequency: pd.DataFrame
    _summary_details: dict

    def summary(self):
        result = {
            "TOTAL PLAYS": self.total_plays,
            "EXPLOSIVE PLAYS": self.explosive_plays,
            "EXPLOSIVE RATE": round(self.explosive_plays / self.total_plays * 100, 1) if self.total_plays else 0,
        }
        result.update(self._summary_details)
        return result


def analyze(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep BALL CARRIER when present. It is used by OFF SELF SCOUT only;
    # opponent scouting data does not have to provide it.
    analysis_columns = REQUIRED_COLUMNS.copy()
    if "BALL CARRIER" in df.columns:
        analysis_columns.append("BALL CARRIER")
    df = df[analysis_columns].copy()
    df = df[df["PLAY #"].map(clean) != ""].reset_index(drop=True)
    df["EXPLOSIVE"] = df.apply(is_explosive, axis=1)

    situation = df.copy()
    situation["SITUATION"] = (situation["DN"].map(clean) + " & " + situation["DIST"].map(clean)).str.strip(" &")

    return TendencyReport(
        len(df),
        int(df["EXPLOSIVE"].sum()),
        df[df["EXPLOSIVE"]].copy(),
        group_rates_combo(df.assign(SITUATION=situation["SITUATION"]), ["OFF FORM", "SITUATION"]),
        group_rates_combo(df, ["HASH", "PLAY DIR"]),
        group_rates(df, "MOTION"),
        previous_play_features(df),
        explosive_sequence_analysis(df),
        trigger_sequence_analysis(df),
        repeated_play_sequences(df),
        repeated_play_followups(df),
        run_pass_sequences(df),
        left_right_sequences(df),
        frequency_profile(df, "OFF FORM"),
        frequency_profile(situation, "SITUATION"),
        hash_down_distance_play_probabilities(df),
        general_play_type_frequency(df),
        situation_sequence_analysis(df),
        situation_play_calling_patterns(df),
        field_zone_efficiency(df),
        field_zone_by_hash(df),
        down_efficiency(df),
        three_and_out_analysis(df),
        frequency_profile(df, "PERSONNEL"),
        frequency_profile(df, "BACKFIELD"),
        frequency_profile(df, "MOTION"),
        frequency_profile(df, "SCHEME"),
        frequency_profile(df, "PLAY DIR"),
        run_pass_yard_summary(df),
        completed_comment_patterns(df, target_source="comments"),
        ball_carrier_identity(df),
        passing_target_summary(df),
        rushing_tendencies(df),
        frequency_tendencies(df, "OFF FORM"),
        overall_summary(df),
    )


def write_report(report, output_dir="output"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report.summary()]).to_csv(out / "summary.csv", index=False)
    report.explosive_detail.to_csv(out / "explosive_plays.csv", index=False)
    report.explosive_by_formation_situation.to_csv(out / "explosives_by_formation_situation.csv", index=False)
    report.explosive_by_hash_direction.to_csv(out / "explosives_by_hash_direction.csv", index=False)
    report.by_motion.to_csv(out / "explosives_by_motion.csv", index=False)
    report.prior_play_features.to_csv(out / "explosive_prior_play_features.csv", index=False)
    report.sequences.to_csv(out / "explosive_sequences.csv", index=False)
    report.trigger_sequences.to_csv(out / "trigger_sequences.csv", index=False)
    report.repeated_sequences.to_csv(out / "repeated_play_sequences.csv", index=False)
    report.repeated_followups.to_csv(out / "repeated_play_followups.csv", index=False)
    report.run_pass_sequences.to_csv(out / "run_pass_sequences.csv", index=False)
    report.left_right_sequences.to_csv(out / "left_right_sequences.csv", index=False)
    report.frequency_by_formation.to_csv(out / "frequency_by_formation.csv", index=False)
    report.frequency_by_situation.to_csv(out / "frequency_by_situation.csv", index=False)
    report.frequency_by_personnel.to_csv(out / "frequency_by_personnel.csv", index=False)
    report.frequency_by_backfield.to_csv(out / "frequency_by_backfield.csv", index=False)
    report.frequency_by_motion.to_csv(out / "frequency_by_motion.csv", index=False)
    report.frequency_by_scheme.to_csv(out / "frequency_by_scheme.csv", index=False)
    report.frequency_by_direction.to_csv(out / "frequency_by_direction.csv", index=False)
    report.run_pass_yards.to_csv(out / "run_pass_yards.csv", index=False)
    report.completed_comment_patterns.to_csv(out / "completed_comment_patterns.csv", index=False)


def main(input_csv, output_dir="output"):
    write_report(analyze(pd.read_csv(input_csv)), output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(args.input_csv, args.output)

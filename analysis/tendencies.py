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
    labels = [play_label(work.iloc[i]) for i in range(len(work))]
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
                candidate = play_label(work.iloc[end + 1])
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

    labels = [play_label(df.iloc[i]) for i in range(len(df))]
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


def frequency_profile(df, column, min_plays=2, top_n=15):
    """Frequency plus run/pass usage and yard production for one analytical lens."""
    work = df[df[column].map(clean) != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=[
            column, "PLAYS", "FREQUENCY_RATE", "RUN", "PASS", "QB_RUN",
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
        QB_RUN=("_CLASS", lambda s: int((s == "QB RUN").sum())),
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


def situation_sequence_analysis(df, min_occurrences=2, top_n=10):
    """Find what follows a specific situation/result combination.

    Example: 2nd & Short + INCOMPLETE PASS -> next-down play type.
    This analyzes every qualifying snap, not just explosives.
    """
    if len(df) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(df) - 1):
        current, following = df.iloc[i], df.iloc[i + 1]
        situation = situation_bucket(current["DN"], current["DIST"])
        if not situation:
            continue
        rows.append({
            "PREVIOUS_SITUATION": situation,
            "PREVIOUS_RESULT": play_result_type(current),
            "NEXT_DOWN": clean(following["DN"]),
            "NEXT_PLAY_TYPE": play_result_type(following),
            "NEXT_PLAY": play_label(following),
        })

    if not rows:
        return pd.DataFrame()

    work = pd.DataFrame(rows)
    grouped = (
        work.groupby(["PREVIOUS_SITUATION", "PREVIOUS_RESULT", "NEXT_DOWN",
                      "NEXT_PLAY_TYPE"])
        .size()
        .reset_index(name="FOLLOWING_COUNT")
    )
    totals = grouped.groupby(
        ["PREVIOUS_SITUATION", "PREVIOUS_RESULT"]
    )["FOLLOWING_COUNT"].transform("sum")
    grouped["TOTAL_AFTER_TRIGGER"] = totals
    grouped["FOLLOWING_RATE"] = (
        grouped["FOLLOWING_COUNT"] / totals * 100
    ).round(1)
    grouped = grouped[grouped["FOLLOWING_COUNT"] >= min_occurrences]
    return grouped.sort_values(
        ["PREVIOUS_SITUATION", "PREVIOUS_RESULT", "FOLLOWING_RATE", "FOLLOWING_COUNT"],
        ascending=[True, True, False, False],
    ).groupby(
        ["PREVIOUS_SITUATION", "PREVIOUS_RESULT"], group_keys=False
    ).head(top_n).reset_index(drop=True)


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
    frequency_by_formation: pd.DataFrame
    frequency_by_situation: pd.DataFrame
    play_type_frequency: pd.DataFrame
    situation_sequences: pd.DataFrame
    frequency_by_personnel: pd.DataFrame
    frequency_by_backfield: pd.DataFrame
    frequency_by_motion: pd.DataFrame
    frequency_by_scheme: pd.DataFrame
    frequency_by_direction: pd.DataFrame
    run_pass_yards: pd.DataFrame

    def summary(self):
        return {
            "TOTAL_PLAYS": self.total_plays,
            "EXPLOSIVE_PLAYS": self.explosive_plays,
            "EXPLOSIVE_RATE": round(self.explosive_plays / self.total_plays * 100, 1) if self.total_plays else 0,
        }


def analyze(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
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
        frequency_profile(df, "OFF FORM"),
        frequency_profile(situation, "SITUATION"),
        general_play_type_frequency(df),
        situation_sequence_analysis(df),
        frequency_profile(df, "PERSONNEL"),
        frequency_profile(df, "BACKFIELD"),
        frequency_profile(df, "MOTION"),
        frequency_profile(df, "SCHEME"),
        frequency_profile(df, "PLAY DIR"),
        run_pass_yard_summary(df),
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
    report.frequency_by_formation.to_csv(out / "frequency_by_formation.csv", index=False)
    report.frequency_by_situation.to_csv(out / "frequency_by_situation.csv", index=False)
    report.frequency_by_personnel.to_csv(out / "frequency_by_personnel.csv", index=False)
    report.frequency_by_backfield.to_csv(out / "frequency_by_backfield.csv", index=False)
    report.frequency_by_motion.to_csv(out / "frequency_by_motion.csv", index=False)
    report.frequency_by_scheme.to_csv(out / "frequency_by_scheme.csv", index=False)
    report.frequency_by_direction.to_csv(out / "frequency_by_direction.csv", index=False)
    report.run_pass_yards.to_csv(out / "run_pass_yards.csv", index=False)


def main(input_csv, output_dir="output"):
    write_report(analyze(pd.read_csv(input_csv)), output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(args.input_csv, args.output)

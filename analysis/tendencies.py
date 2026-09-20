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


def is_pass(row):
    b = play_blob(row)
    return any(x in b for x in ["PASS", "SCREEN", "RPO", "QB THROW", "COMPLETE", "INCOMPLETE", "INTERCEPTION"])


def is_run(row):
    b = play_blob(row)
    return any(x in b for x in ["RUN", "RUSH", "SCRAMBLE", "QB RUN", "KEEP", "DRAW", "SNEAK"])


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


def group_rates(df, column):
    work = df[df[column].map(clean) != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=[column, "PLAYS", "EXPLOSIVES", "EXPLOSIVE_RATE"])
    out = work.groupby(column).agg(PLAYS=("EXPLOSIVE", "size"), EXPLOSIVES=("EXPLOSIVE", "sum")).reset_index()
    out["EXPLOSIVE_RATE"] = (out["EXPLOSIVES"] / out["PLAYS"] * 100).round(1)
    return out.sort_values(["EXPLOSIVE_RATE", "EXPLOSIVES", "PLAYS"], ascending=False)


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
    """Find what commonly follows meaningful trigger plays.

    This is intentionally independent of explosive-play analysis. It looks at
    every snap and reports factual follow-up patterns for common triggers.
    """
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


def repeated_play_sequences(df, min_occurrences=2):
    """Find repeated exact play sequences of length 2 and 3.

    The sequence is based on OFF PLAY plus PLAY DIR when available. This keeps
    'Power RT' and 'Power LT' distinct while still working if direction is blank.
    """
    if len(df) < 2:
        return pd.DataFrame()

    labels = [play_label(df.iloc[i]) for i in range(len(df))]
    rows = []

    for length in (2, 3):
        if len(labels) < length:
            continue
        counts = {}
        for i in range(len(labels) - length + 1):
            sequence = tuple(labels[i:i + length])
            if any(not x or x == "UNKNOWN" for x in sequence):
                continue
            counts[sequence] = counts.get(sequence, 0) + 1

        for sequence, count in counts.items():
            if count >= min_occurrences:
                rows.append({
                    "SEQUENCE_LENGTH": length,
                    "SEQUENCE": " → ".join(sequence),
                    "OCCURRENCES": count,
                })

    if not rows:
        return pd.DataFrame(columns=["SEQUENCE_LENGTH", "SEQUENCE", "OCCURRENCES"])

    return pd.DataFrame(rows).sort_values(
        ["SEQUENCE_LENGTH", "OCCURRENCES"],
        ascending=[True, False],
    )


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


@dataclass
class TendencyReport:
    total_plays: int
    explosive_plays: int
    explosive_detail: pd.DataFrame
    by_formation: pd.DataFrame
    by_situation: pd.DataFrame
    by_hash: pd.DataFrame
    by_direction: pd.DataFrame
    by_personnel: pd.DataFrame
    by_motion: pd.DataFrame
    prior_play_features: pd.DataFrame
    sequences: pd.DataFrame
    trigger_sequences: pd.DataFrame
    repeated_sequences: pd.DataFrame
    repeated_followups: pd.DataFrame

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
        group_rates(df, "OFF FORM"),
        group_rates(situation, "SITUATION"),
        group_rates(df, "HASH"),
        group_rates(df, "PLAY DIR"),
        group_rates(df, "PERSONNEL"),
        group_rates(df, "MOTION"),
        previous_play_features(df),
        explosive_sequence_analysis(df),
        trigger_sequence_analysis(df),
        repeated_play_sequences(df),
        repeated_play_followups(df),
    )


def write_report(report, output_dir="output"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report.summary()]).to_csv(out / "summary.csv", index=False)
    report.explosive_detail.to_csv(out / "explosive_plays.csv", index=False)
    report.by_formation.to_csv(out / "explosives_by_formation.csv", index=False)
    report.by_situation.to_csv(out / "explosives_by_situation.csv", index=False)
    report.by_hash.to_csv(out / "explosives_by_hash.csv", index=False)
    report.by_direction.to_csv(out / "explosives_by_direction.csv", index=False)
    report.by_personnel.to_csv(out / "explosives_by_personnel.csv", index=False)
    report.by_motion.to_csv(out / "explosives_by_motion.csv", index=False)
    report.prior_play_features.to_csv(out / "explosive_prior_play_features.csv", index=False)
    report.sequences.to_csv(out / "explosive_sequences.csv", index=False)
    report.trigger_sequences.to_csv(out / "trigger_sequences.csv", index=False)
    report.repeated_sequences.to_csv(out / "repeated_play_sequences.csv", index=False)
    report.repeated_followups.to_csv(out / "repeated_play_followups.csv", index=False)


def main(input_csv, output_dir="output"):
    write_report(analyze(pd.read_csv(input_csv)), output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(args.input_csv, args.output)

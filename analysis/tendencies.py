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
        if pd.isna(value) or clean(value) == "": return None
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError): return None

def result_yards(value):
    text = clean(value).upper()
    match = re.search(r"(?<!\\d)(-?\\d+(?:\\.\\d+)?)\\s*(?:YDS?|YARDS?)?(?!\\d)", text)
    return float(match.group(1)) if match else None

def play_blob(row):
    return " ".join(clean(row.get(c, "")) for c in ["PLAY TYPE","OFF PLAY","RESULT"]).upper()

def is_pass(row):
    b = play_blob(row)
    return any(x in b for x in ["PASS","SCREEN","RPO","QB THROW","COMPLETE","INCOMPLETE","INTERCEPTION"])

def is_run(row):
    b = play_blob(row)
    return any(x in b for x in ["RUN","RUSH","SCRAMBLE","QB RUN","KEEP","DRAW","SNEAK"])

def is_explosive(row):
    yards = numeric(row.get("GN/LS"))
    if yards is None: yards = result_yards(row.get("RESULT"))
    if yards is None: return False
    if is_pass(row): return yards >= EXPLOSIVE_PASS_YARDS
    if is_run(row): return yards >= EXPLOSIVE_RUN_YARDS
    return False

def group_rates(df, column):
    work = df[df[column].map(clean) != ""].copy()
    if work.empty: return pd.DataFrame(columns=[column,"PLAYS","EXPLOSIVES","EXPLOSIVE_RATE"])
    out = work.groupby(column).agg(PLAYS=("EXPLOSIVE","size"), EXPLOSIVES=("EXPLOSIVE","sum")).reset_index()
    out["EXPLOSIVE_RATE"] = (out["EXPLOSIVES"] / out["PLAYS"] * 100).round(1)
    return out.sort_values(["EXPLOSIVE_RATE","EXPLOSIVES","PLAYS"], ascending=False)

def previous_play_features(df):
    if len(df) < 2: return pd.DataFrame()
    cols = ["OFF FORM","PERSONNEL","MOTION","PLAY TYPE","PLAY DIR","PLAY (STR/WK)","HASH","DN","DIST"]
    rows = []
    for i in range(1, len(df)):
        if not bool(df.iloc[i]["EXPLOSIVE"]): continue
        prev = df.iloc[i-1]
        for col in cols:
            value = clean(prev[col])
            if not value: continue
            previous_values = df.iloc[:-1][col].map(clean)
            mask = previous_values == value
            total = int(mask.sum())
            explosives = int(df["EXPLOSIVE"].iloc[1:][mask].sum())
            overall_mask = df[col].map(clean) == value
            overall_total = int(overall_mask.sum())
            overall_explosives = int(df.loc[overall_mask, "EXPLOSIVE"].sum())
            rows.append({"PREVIOUS_FEATURE":col,"VALUE":value,"EXPLOSIVE_FOLLOWING_PLAYS":explosives,"TOTAL_FOLLOWING_PLAYS":total,"FOLLOWING_EXPLOSIVE_RATE":round(explosives/total*100,1) if total else 0,"OVERALL_FEATURE_PLAYS":overall_total,"OVERALL_EXPLOSIVES":overall_explosives,"OVERALL_EXPLOSIVE_RATE":round(overall_explosives/overall_total*100,1) if overall_total else 0})
    out = pd.DataFrame(rows).drop_duplicates()
    if out.empty: return out
    out["RATE_DIFFERENCE"] = (out["FOLLOWING_EXPLOSIVE_RATE"] - out["OVERALL_EXPLOSIVE_RATE"]).round(1)
    return out.sort_values(["RATE_DIFFERENCE","FOLLOWING_EXPLOSIVE_RATE","TOTAL_FOLLOWING_PLAYS"], ascending=False)

def sequence_analysis(df):
    if len(df) < 2: return pd.DataFrame()
    rows = []
    for i in range(1, len(df)):
        if not bool(df.iloc[i]["EXPLOSIVE"]): continue
        p, c = df.iloc[i-1], df.iloc[i]
        rows.append({"PREV_FORM":clean(p["OFF FORM"]),"PREV_MOTION":clean(p["MOTION"]),"PREV_PLAY_TYPE":clean(p["PLAY TYPE"]),"EXPLOSIVE_FORM":clean(c["OFF FORM"]),"EXPLOSIVE_PLAY_TYPE":clean(c["PLAY TYPE"]),"EXPLOSIVE_PLAY":clean(c["OFF PLAY"])})
    out = pd.DataFrame(rows)
    if out.empty: return out
    keys = ["PREV_FORM","PREV_MOTION","PREV_PLAY_TYPE","EXPLOSIVE_FORM","EXPLOSIVE_PLAY_TYPE"]
    return out.groupby(keys).size().reset_index(name="EXPLOSIVE_COUNT").sort_values("EXPLOSIVE_COUNT", ascending=False)

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
    def summary(self):
        return {"TOTAL_PLAYS":self.total_plays,"EXPLOSIVE_PLAYS":self.explosive_plays,"EXPLOSIVE_RATE":round(self.explosive_plays/self.total_plays*100,1) if self.total_plays else 0}

def analyze(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df = df[df["PLAY #"].map(clean) != ""].reset_index(drop=True)
    df["EXPLOSIVE"] = df.apply(is_explosive, axis=1)
    situation = df.copy()
    situation["SITUATION"] = (situation["DN"].map(clean) + " & " + situation["DIST"].map(clean)).str.strip(" &")
    return TendencyReport(len(df), int(df["EXPLOSIVE"].sum()), df[df["EXPLOSIVE"]].copy(), group_rates(df,"OFF FORM"), group_rates(situation,"SITUATION"), group_rates(df,"HASH"), group_rates(df,"PLAY DIR"), group_rates(df,"PERSONNEL"), group_rates(df,"MOTION"), previous_play_features(df), sequence_analysis(df))

def write_report(report, output_dir="output"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report.summary()]).to_csv(out/"summary.csv", index=False)
    report.explosive_detail.to_csv(out/"explosive_plays.csv", index=False)
    report.by_formation.to_csv(out/"explosives_by_formation.csv", index=False)
    report.by_situation.to_csv(out/"explosives_by_situation.csv", index=False)
    report.by_hash.to_csv(out/"explosives_by_hash.csv", index=False)
    report.by_direction.to_csv(out/"explosives_by_direction.csv", index=False)
    report.by_personnel.to_csv(out/"explosives_by_personnel.csv", index=False)
    report.by_motion.to_csv(out/"explosives_by_motion.csv", index=False)
    report.prior_play_features.to_csv(out/"explosive_prior_play_features.csv", index=False)
    report.sequences.to_csv(out/"explosive_sequences.csv", index=False)

def main(input_csv, output_dir="output"):
    write_report(analyze(pd.read_csv(input_csv)), output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    main(args.input_csv, args.output)
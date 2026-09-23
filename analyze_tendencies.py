"""Gridiron Live GitHub Actions entry point.

Google Sheet -> GitHub Actions -> Python -> Google Sheet.
Opponent/game data never needs to be committed to this repository.
"""
from sheets import (
    get_spreadsheet,
    load_source_df,
    load_defense_source_df,
    write_report,
    write_defense_sheet,
)
from analysis.tendencies import analyze
from analysis.defense import analyze_defense


def main():
    spreadsheet = get_spreadsheet()

    # Opponent scouting remains on ALL INFO SHEET -> Tendencies.
    df = load_source_df(spreadsheet)
    report = analyze(df)
    write_report(spreadsheet, report)

    # Self-scout defense comes from WHS DATA.  WHS DATA contains O/D/K,
    # so only ODK=D snaps are sent to the defensive analyzer.
    defense_df = load_defense_source_df(spreadsheet)
    defense_report = analyze_defense(defense_df)
    write_defense_sheet(spreadsheet, defense_report)

    print(
        f"Gridiron Live complete: {report.total_plays} opponent plays, "
        f"{report.explosive_plays} explosive plays; "
        f"{len(defense_df)} defensive self-scout plays."
    )


if __name__ == "__main__":
    main()

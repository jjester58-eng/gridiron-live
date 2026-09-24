"""Gridiron Live GitHub Actions entry point.

Google Sheet -> GitHub Actions -> Python -> Google Sheet.
Opponent/game data never needs to be committed to this repository.
"""
from sheets import (
    get_spreadsheet,
    load_source_df,
    load_defense_source_df,
    load_offense_source_df,
    write_report,
    write_defense_sheet,
    write_offense_self_scout,
)
from analysis.tendencies import analyze
from analysis.defense import analyze_defense, formation_call_matrix


def main():
    spreadsheet = get_spreadsheet()

    # Opponent scouting remains on ALL INFO SHEET -> Tendencies.
    df = load_source_df(spreadsheet)

    # Self-scout defense comes from WHS DATA. WHS DATA contains O/D/K,
    # so only ODK=D snaps are sent to the defensive analyzer.
    defense_df = load_defense_source_df(spreadsheet)

    # Build the main Tendencies report from opponent data only.
    # The separate bottom formation/call matrix combines this opponent data
    # with WHS DATA defensive history.
    report = analyze(df)

    # Compare opponent frequency to WHS defensive results for matching
    # down/distance + offensive formation situations.
    formation_call_report = formation_call_matrix(df, defense_df)

    write_report(spreadsheet, report, formation_call_report=formation_call_report)
    defense_report = analyze_defense(defense_df)
    write_defense_sheet(spreadsheet, defense_report)

    # Self-scout offense comes from the same WHS DATA source. ODK=O only.
    offense_df = load_offense_source_df(spreadsheet)
    offense_report = analyze(offense_df, target_source="ball_carrier")
    write_offense_self_scout(spreadsheet, offense_report)

    print(
        f"Gridiron Live complete: {report.total_plays} opponent plays, "
        f"{report.explosive_plays} explosive plays; "
        f"{len(defense_df)} defensive self-scout plays; "
        f"{len(offense_df)} offensive self-scout plays."
    )


if __name__ == "__main__":
    main()

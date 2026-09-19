"""Gridiron Live GitHub Actions entry point.

Google Sheet -> GitHub Actions -> Python -> Google Sheet.
Opponent/game data never needs to be committed to this repository.
"""
from sheets import get_spreadsheet, load_source_df, write_report
from analysis.tendencies import analyze

def main():
    spreadsheet = get_spreadsheet()
    df = load_source_df(spreadsheet)
    report = analyze(df)
    write_report(spreadsheet, report)
    print(f"Gridiron Live complete: {report.total_plays} plays, {report.explosive_plays} explosive plays.")

if __name__ == "__main__":
    main()

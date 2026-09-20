# Gridiron Live

Factual football tendency analysis for the weekly war room.

## Architecture

**Google Sheet -> Apps Script -> GitHub Actions -> Python -> same Google Sheet**

This follows the proven pattern used by the Sheets-Hudl project. There is no Vercel API and no requirement to run a local Python server.

## Weekly workflow

1. Enter/paste opponent plays into the Google Sheets **ALL INFO SHEET**.
2. From the Sheet's **Football** menu, run the Gridiron Live tendency report.
3. Apps Script dispatches the **Gridiron Live Tendency Report** GitHub Actions workflow.
4. GitHub Actions reads the Sheet through the Google Sheets API.
5. Python analyzes the complete sample.
6. Python rebuilds the **Tendencies** tab in the same spreadsheet.
7. Copy/archive the completed Sheet for the opponent/game, then clear the working data for the next week.

Opponent/game play data is never committed to this repository.

## Current analysis

- Explosive runs: **15+ yards**
- Explosive passes: **20+ yards**
- Explosives by formation
- Explosives by down + distance
- Hash, direction, personnel, and motion
- Previous-play feature analysis
- Immediate explosive-play sequences
- High-frequency tendencies by formation, situation, personnel, backfield, motion, scheme, and direction
- Situation/result → next-play sequence analysis
- Run, pass, QB-run, and sack yard production
- Run/pass yards and yards per play within frequency profiles
- Rate comparisons against the overall sample

The analyzer is descriptive and does not make coaching decisions.

## Required Google Sheet columns

`PLAY #, ODK, DN, DIST, HASH, YARD LN, PLAY TYPE, RESULT, GN/LS, PERSONNEL, OFF FORM, BACKFIELD, MOTION, PS ALIGNMENT, WS ALIGNMENT, STUD ALIGNMENT, FIB, SCHEME, OFF PLAY, PLAY DIR, PLAY (STR/WK), PZ HOLLEY, PASS PRO, READ, AWAY, COMMENTS`

Extra blank columns are ignored.

## GitHub Actions secrets

The repository workflow expects these repository secrets:

- `GOOGLE_CREDS` — Google service-account JSON
- `SPREADSHEET_ID` — the target Google Sheet ID

The Apps Script uses its existing `GITHUB_TOKEN` Script Property to dispatch the workflow.

## No Vercel

Vercel/API files are intentionally not part of this architecture. The GitHub Action is the Python execution environment, just like the existing Sheets-Hudl workflow.

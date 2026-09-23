# Gridiron Live

Factual football tendency analysis for the weekly war room.

## Architecture

**Google Sheet -> Apps Script -> GitHub Actions -> Python -> same Google Sheet**

This follows the proven pattern used by the Sheets-Hudl project. There is no Vercel API and no requirement to run a local Python server.

The Python analysis engine now has a separate defensive self-scout module in `analysis/defense.py`. It is kept separate from the existing opponent report so the current workflow is not changed while the offensive/opponent report is being fine-tuned. A future Google Sheet tab can call the same engine with the defensive schema.

## Weekly workflow

1. Enter/paste opponent plays into the Google Sheets **ALL INFO SHEET**.
2. From the Sheet's **Football** menu, run the Gridiron Live tendency report.
3. Apps Script dispatches the **Gridiron Live Tendency Report** GitHub Actions workflow.
4. GitHub Actions reads the Sheet through the Google Sheets API.
5. Python analyzes the complete sample.
6. Python rebuilds the **Tendencies** tab in the same spreadsheet.
7. Copy/archive the completed Sheet for the opponent/game, then clear the working data for the next week.

Opponent/game play data is never committed to this repository.

## Current opponent analysis

- Explosive runs: **15+ yards**
- Explosive passes: **20+ yards**
- Explosives by formation and situation
- Hash, direction, personnel, and motion
- Run/pass and left/right 2-play and 3-play patterns
- Situation → play-calling patterns
- Repeated play sequences
- Completion/incompletion + formation + comment tallies
- Pass attempts/completions and run/pass yard summaries

The analyzer is descriptive and does not make coaching decisions.

## Defensive self-scout engine

The new `analysis/defense.py` module accepts the defensive sheet structure:

`PLAY #, ODK, DN, DIST, HASH, YARD LN, RPO, PLAY TYPE, RESULT, GN/LS, PERSONNEL, OFF FORM, MOTION, OFF PLAY, DEF CALL, DEF FRONT, DEF STUNT, COVERAGE, BLITZ, COMMENTS`

The defensive engine measures:

- Overall defensive production
- Run/pass results
- Defensive call → result
- Front → result
- Coverage → result
- Blitz → result
- Offensive formation/personnel → result
- Down/distance → result
- Situation → defensive call frequency
- Situation → defensive call → result
- Explosive-play context
- Repeated defensive-call, coverage, and blitz patterns
- Defensive call/coverage/comment tallies
- Objective **strength/improvement indicators** compared with the overall sample

Only groups meeting the minimum sample threshold are surfaced in the strength/improvement indicators. Those indicators compare observed results to the overall sample; they are not predictions.

The defensive module intentionally ignores the fields not used by the current defensive self-scout: **FIB, BACKFIELD, PS ALIGNMENT, WS ALIGNMENT, STUD ALIGNMENT, SCHEME, PLAY DIR, PLAY (STR/WK), PZ HOLLEY, PASS PRO, READ, AWAY**.

The defensive tab/report wiring will be added later, after the current opponent report is finished.

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

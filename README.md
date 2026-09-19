# Gridiron Live

Factual football tendency analysis for the weekly war room.

## Workflow
1. Paste opponent plays into the Google Sheets ALL INFO SHEET.
2. The Sheets bridge sends the data to Python.
3. Python scans the complete sample and returns factual tendency tables.
4. Review the Tendencies sheet and filter/sort as needed.
5. Archive the completed game by copying the Sheet, then clear the working data.

Do not commit opponent/game play data to this repository.

## Current analysis
- Explosive runs: 15+ yards
- Explosive passes: 20+ yards
- Explosives by formation
- Explosives by down + distance
- Hash, direction, personnel, and motion
- Previous-play feature analysis
- Immediate explosive-play sequences
- Rate comparisons against the overall sample

The analyzer is descriptive and does not make coaching decisions.

## Input columns
PLAY #, ODK, DN, DIST, HASH, YARD LN, PLAY TYPE, RESULT, GN/LS, PERSONNEL, OFF FORM, BACKFIELD, MOTION, PS ALIGNMENT, WS ALIGNMENT, STUD ALIGNMENT, FIB, SCHEME, OFF PLAY, PLAY DIR, PLAY (STR/WK), PZ HOLLEY, PASS PRO, READ, AWAY, COMMENTS

Extra blank columns are ignored.

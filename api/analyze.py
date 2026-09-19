"""
Vercel API wrapper for Gridiron Live.

POST JSON:
{
  "plays": [
    {
      "PLAY #": 1,
      ...
    }
  ]
}

Returns the factual tendency report as JSON for Google Apps Script.
"""

import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Allow the Vercel function to import analysis/tendencies.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.tendencies import analyze


def dataframe_records(df):
    if df is None or df.empty:
        return []

    # Convert NaN/NaT values so the response is valid JSON.
    clean_df = df.copy().fillna("")
    return clean_df.to_dict(orient="records")


def report_to_dict(report):
    return {
        "summary": report.summary(),
        "explosive_plays": dataframe_records(report.explosive_detail),
        "explosives_by_formation": dataframe_records(report.by_formation),
        "explosives_by_situation": dataframe_records(report.by_situation),
        "explosives_by_hash": dataframe_records(report.by_hash),
        "explosives_by_direction": dataframe_records(report.by_direction),
        "explosives_by_personnel": dataframe_records(report.by_personnel),
        "explosives_by_motion": dataframe_records(report.by_motion),
        "explosive_prior_play_features": dataframe_records(report.prior_play_features),
        "explosive_sequences": dataframe_records(report.sequences),
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))

            if length <= 0:
                self._send_json(400, {
                    "ok": False,
                    "error": "Request body is empty."
                })
                return

            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))

            plays = payload.get("plays")

            if not isinstance(plays, list) or not plays:
                self._send_json(400, {
                    "ok": False,
                    "error": "Request must contain a non-empty 'plays' array."
                })
                return

            # The Apps Script sends dictionaries keyed by the exact
            # ALL INFO SHEET column names.
            import pandas as pd

            df = pd.DataFrame(plays)
            report = analyze(df)

            self._send_json(200, {
                "ok": True,
                "summary": report.summary(),
                "sections": report_to_dict(report)
            })

        except ValueError as exc:
            self._send_json(400, {
                "ok": False,
                "error": str(exc)
            })

        except Exception as exc:
            self._send_json(500, {
                "ok": False,
                "error": f"Gridiron Live analysis failed: {exc}"
            })


    def do_GET(self):
        self._send_json(200, {
            "ok": True,
            "service": "Gridiron Live",
            "message": "Python analysis API is online. Send play data with POST."
        })

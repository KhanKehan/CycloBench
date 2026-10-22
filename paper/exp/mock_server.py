# -*- coding: utf-8 -*-
"""Minimal OpenAI-compatible mock server for end-to-end testing of run_api.py.

Returns a fixed (mostly wrong) JSON answer, so the pipeline can be verified
without a provider key:  python mock_server.py --port 8765
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER = {
    "Q1_total_duration_Myr": 6.0,
    "Q2_age_250m_Ma": 304.0,
    "Q3_hiatuses": [{"depth_m": 100.0, "duration_kyr": 200.0}],
    "Q4_mean_SR_cm_kyr": 7.0,
    "Q5_obliquity_kyr": 41.0,
    "Q6": {"p60_autogenic": True, "p112_autogenic": True,
           "carb_has_60kyr_peak": False, "carb_has_112kyr_peak": False},
    "Q7": {"exists": True, "stronger_in": "CARB", "GR_factor": 0.4, "CARB_factor": 0.6},
    "Q8": {"wellA_depth_m": 100.0, "wellB_depth_m": 110.0, "duration_kyr": 200.0},
}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(n)
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "```json\n" + json.dumps(ANSWER) + "\n```"}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

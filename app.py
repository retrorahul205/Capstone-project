#!/usr/bin/env python
"""Flask API: train, predict, and logfile endpoints for AAVAIL revenue forecasts."""

from __future__ import annotations

import traceback
from pathlib import Path

from flask import Flask, jsonify, request

from logger import read_logs
from model import DEFAULT_DATA_DIR, MODEL_VERSION, model_predict, model_train

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "app": "AAVAIL revenue forecast API",
            "version": MODEL_VERSION,
            "endpoints": ["/train", "/predict", "/logs"],
        }
    )


@app.route("/train", methods=["GET", "POST"])
def train():
    payload = request.get_json(silent=True) or {}
    test = bool(payload.get("test", request.args.get("test", "false").lower() == "true"))
    data_dir = payload.get("data_dir") or request.args.get("data_dir") or str(DEFAULT_DATA_DIR)
    try:
        results = model_train(data_dir=data_dir, test=test)
        return jsonify({"status": "ok", "test": test, "results": results}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc), "trace": traceback.format_exc()}), 400


@app.route("/predict", methods=["GET", "POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    args = {**payload, **request.args.to_dict()}
    try:
        country = args.get("country", "all")
        year = args["year"]
        month = args["month"]
        day = args["day"]
        test = str(args.get("test", "false")).lower() == "true"
        result = model_predict(country, year, month, day, test=test)
        return jsonify({"status": "ok", **result}), 200
    except KeyError as exc:
        return jsonify({"status": "error", "message": f"missing parameter: {exc}"}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/logs", methods=["GET"])
def logs():
    kind = request.args.get("kind", "predict")
    test = request.args.get("test", "false").lower() == "true"
    try:
        content = read_logs(kind=kind, test=test)
        return jsonify({"status": "ok", "kind": kind, "log": content}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

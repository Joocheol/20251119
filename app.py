from __future__ import annotations

from typing import Any, Dict

from flask import Flask, render_template, request

from monte_carlo_option import price_option_from_params


app = Flask(__name__)


def _get_default_form_data() -> Dict[str, Any]:
    return {
        "s0": 100.0,
        "r": 0.03,
        "sigma": 0.2,
        "t": 1.0,
        "steps": 50,
        "paths": 20000,
        "k": 110.0,
        "payoff": "max(ST - K, 0)",
        "seed": "",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result: Dict[str, float] | None = None
    error: str | None = None
    form_data = _get_default_form_data()

    if request.method == "POST":
        form_data.update({
            "s0": request.form.get("s0", form_data["s0"]),
            "r": request.form.get("r", form_data["r"]),
            "sigma": request.form.get("sigma", form_data["sigma"]),
            "t": request.form.get("t", form_data["t"]),
            "steps": request.form.get("steps", form_data["steps"]),
            "paths": request.form.get("paths", form_data["paths"]),
            "k": request.form.get("k", form_data["k"]),
            "payoff": request.form.get("payoff", form_data["payoff"]),
            "seed": request.form.get("seed", form_data["seed"]),
        })

        try:
            seed_value = request.form.get("seed", "").strip()
            seed = int(seed_value) if seed_value else None
            result = price_option_from_params(
                s0=float(form_data["s0"]),
                r=float(form_data["r"]),
                sigma=float(form_data["sigma"]),
                t=float(form_data["t"]),
                steps=int(form_data["steps"]),
                paths=int(form_data["paths"]),
                k=float(form_data["k"]) if form_data["k"] not in (None, "") else None,
                payoff=form_data["payoff"],
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 - surface error to user
            error = str(exc)

    return render_template("index.html", result=result, error=error, form_data=form_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

"""
Gradio frontend: the human-facing 'View'.

Organized as a `GradioUI` class so the HTTP-vs-in-process choice
(`STANDALONE`) and the Blocks layout are encapsulated together rather than
spread across module-level functions and globals.

By default this calls the FastAPI backend over HTTP (so the frontend
container/process can be scaled or deployed independently of the API).
Set STANDALONE=1 to instead call the controller in-process (handy for local
development without running two servers).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

from student_success.config import Settings
from student_success.config import settings as default_settings


class GradioUI:
    """Builds and serves the Gradio Blocks app for student success prediction."""

    def __init__(self, settings: Settings | None = None, standalone: bool | None = None) -> None:
        self.settings = settings or default_settings
        if standalone is not None:
            self.standalone = standalone
        else:
            self.standalone = os.getenv("STANDALONE", "0") == "1"
        self._controller = None
        if self.standalone:
            from student_success.controllers.prediction_controller import controller

            self._controller = controller

    # ------------------------------------------------------------------ #
    # Prediction callbacks
    # ------------------------------------------------------------------ #
    def predict_single(
        self, gender: str, residence: str, entry_exam: float, study_hours: float
    ) -> str:
        if self.standalone:
            pct, grade = self._predict_single_standalone(
                gender, residence, entry_exam, study_hours
            )
        else:
            pct, grade = self._predict_single_remote(gender, residence, entry_exam, study_hours)

        if pct is None:
            return f"❌ Error: {grade}"
        return f"📊 Predicted Total Percentage: **{pct}%**\n🎓 Grade: {grade}"

    def _predict_single_standalone(self, gender, residence, entry_exam, study_hours):
        from student_success.models.schemas import StudentInput

        result = self._controller.predict_single(
            StudentInput(
                gender=gender, residence=residence, entry_exam=entry_exam, study_hours=study_hours
            )
        )
        return result.predicted_total_percentage, result.grade

    def _predict_single_remote(self, gender, residence, entry_exam, study_hours):
        resp = requests.post(
            f"{self.settings.api_base_url}/predict",
            json={
                "gender": gender,
                "residence": residence,
                "entry_exam": entry_exam,
                "study_hours": study_hours,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None, resp.json().get("detail", resp.text)
        data = resp.json()
        return data["predicted_total_percentage"], data["grade"]

    def predict_file(self, file) -> pd.DataFrame | None:
        if file is None:
            return None

        if self.standalone:
            df = self._predict_file_standalone(file)
        else:
            df = self._predict_file_remote(file)
        return df

    def _predict_file_standalone(self, file) -> pd.DataFrame | None:
        with open(file.name, "rb") as f:
            csv_bytes = f.read()
        try:
            df = self._controller.predict_batch(csv_bytes)
        except Exception as exc:  # noqa: BLE001
            gr.Warning(str(exc))
            return None
        return df[["Predicted_Total_Percentage", "Grade"]]

    def _predict_file_remote(self, file) -> pd.DataFrame | None:
        with open(file.name, "rb") as f:
            resp = requests.post(
                f"{self.settings.api_base_url}/predict/batch",
                files={"file": (Path(file.name).name, f, "text/csv")},
                timeout=60,
            )
        if resp.status_code != 200:
            gr.Warning(resp.json().get("detail", resp.text))
            return None
        return pd.read_csv(io.StringIO(resp.text))[["Predicted_Total_Percentage", "Grade"]]

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Student Success Predictor") as demo:
            gr.Markdown("# 🎓 Student Success Predictor")
            gr.Markdown(
                "Predict a student's total percentage score from entry exam results, "
                "study hours, gender and residence."
            )

            with gr.Tab("Manual Input"):
                gender = gr.Radio(self.settings.gender_categories, label="Gender", value="Male")
                residence = gr.Radio(
                    self.settings.residence_categories, label="Residence", value="BI Residence"
                )
                entry_exam = gr.Slider(0, 100, value=70, label="Entry Exam Score")
                study_hours = gr.Slider(0, 60, value=15, label="Weekly Study Hours")
                predict_btn = gr.Button("Predict", variant="primary")
                output = gr.Markdown()

                predict_btn.click(
                    fn=self.predict_single,
                    inputs=[gender, residence, entry_exam, study_hours],
                    outputs=output,
                )

            with gr.Tab("Batch (CSV Upload)"):
                cols = "`, `".join(self.settings.feature_order)
                gr.Markdown(f"Upload a CSV with columns: `{cols}`")
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                batch_output = gr.Dataframe(label="Predictions")
                file_input.change(fn=self.predict_file, inputs=file_input, outputs=batch_output)

        return demo

    def launch(self, **launch_kwargs) -> None:
        demo = self.build()
        launch_kwargs.setdefault("server_name", "0.0.0.0")
        launch_kwargs.setdefault("server_port", self.settings.gradio_port)
        demo.launch(**launch_kwargs)


def main() -> None:
    """Console-script entrypoint: `serve-ui` (see pyproject.toml)."""
    GradioUI().launch()


if __name__ == "__main__":
    main()

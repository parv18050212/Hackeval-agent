import os
from typing import Any, Dict, List, Optional

class ProjectAnalysisContext:
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        base = os.path.basename(file_path)
        self.team_name: str = os.path.splitext(base)[0].strip()

        # New field for user-provided parameters
        self.evaluation_parameters: Dict[str, Any] = {}

        # Raw content
        self.raw_text: str = ""
        self.images_base64: List[str] = []
        self.images_meta: List[Dict[str, Any]] = []

        # Diagram/workflow
        self.workflow_report: Optional[Dict[str, Any]] = None
        self.workflow_report_text: str = ""
        self.workflow_analysis: Optional[Dict[str, Any]] = None

        # Scoring + feedback
        self.scores: Dict[str, Any] = {}
        self.scoring_summary: str = ""
        self.feedback: Optional[Dict[str, Any]] = None

        # Format notes
        self.format_notes: List[str] = []

        # Error
        self.evaluation_error: Optional[str] = None

    # -------- content --------
    def update_raw_content(self, text: str, images_b64: List[str], images_meta: Optional[List[Dict[str, Any]]] = None):
        self.raw_text = text or ""
        self.images_base64 = images_b64 or []
        self.images_meta = images_meta or []

    # -------- workflow/diagram --------
    def update_workflow_results(self, report: Dict[str, Any]):
        self.workflow_report = report or {}
        self.workflow_analysis = report or {}
        self.workflow_report_text = ""
        if isinstance(report, dict):
            self.workflow_report_text = report.get("overall_summary", "") or ""

    def update_workflow_report(self, report: Dict[str, Any]):
        self.update_workflow_results(report)

    # -------- scoring --------
    def update_scoring_results(self, team_name: str, scores: Dict[str, Any], summary: str,
                               workflow_analysis: Optional[Dict[str, Any]] = None):
        if team_name:
            self.team_name = team_name
        self.scores = scores or {}
        self.scoring_summary = summary or ""
        if workflow_analysis:
            self.workflow_analysis = workflow_analysis
            if isinstance(workflow_analysis, dict) and not self.workflow_report_text:
                self.workflow_report_text = workflow_analysis.get("overall_summary", "") or ""

    # -------- feedback --------
    def update_feedback_results(self, feedback: Dict[str, Any]):
        self.feedback = feedback or {}

    # -------- format --------
    def update_format_results(self, score: int, notes: List[str]):
        self.format_notes = notes or []

    # -------- errors --------
    def set_error(self, msg: str):
        self.evaluation_error = msg or "Unknown error"

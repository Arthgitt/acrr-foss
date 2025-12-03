# app/agents/financial_agents.py

from __future__ import annotations

from typing import Dict, Any

from app.rag.rag_pipeline import RAGPipeline
from app.agents.base import BaseAgent


class OverviewAgent(BaseAgent):
    name = "overview"
    description = "Summarizes the financial document in plain language."

    def run(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        question = (
            "You are a financial analyst. "
            "Give a concise high-level summary of this document for a non-expert. "
            "Use at most 5 bullet points. Be clear and avoid jargon."
        )

        result = pipeline.query(question=question, top_k=8)

        return {
            "agent": self.name,
            "title": "Document Overview",
            "summary": result.get("answer", ""),
            "contexts": result.get("contexts", []),
        }


class RiskAgent(BaseAgent):
    name = "risk"
    description = "Identifies potential risks and red flags in the document."

    def run(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        question = (
            "You are a risk and compliance officer. "
            "Review this document and list the main risks and red flags. "
            "Focus on things like:\n"
            "- Missing or unclear terms\n"
            "- Unusual fees or interest rates\n"
            "- Clauses that could harm the customer\n"
            "- Any signs of potential fraud or non-compliance\n\n"
            "Return your answer as a bullet list. For each item include:\n"
            "- A short title\n"
            "- Severity (low/medium/high)\n"
            "- A 2–3 sentence explanation."
        )

        result = pipeline.query(question=question, top_k=10)

        return {
            "agent": self.name,
            "title": "Risk & Red Flags",
            "risks": result.get("answer", ""),
            "contexts": result.get("contexts", []),
        }


class ChecklistAgent(BaseAgent):
    name = "checklist"
    description = "Produces a checklist of follow-up questions for compliance review."

    def run(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        question = (
            "You are helping a compliance officer review this document. "
            "Create a checklist of specific questions they should ask to fully "
            "understand the risks and terms.\n\n"
            "Return 8-12 checklist items. Each item should be a short question "
            "starting with words like 'Is', 'Does', 'Are', 'How', or 'What'."
        )

        result = pipeline.query(question=question, top_k=8)

        return {
            "agent": self.name,
            "title": "Compliance Checklist",
            "checklist": result.get("answer", ""),
            "contexts": result.get("contexts", []),
        }

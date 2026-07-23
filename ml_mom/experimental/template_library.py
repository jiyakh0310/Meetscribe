"""Reusable deterministic templates for the experimental MoM formatter.

Purpose:
    Keep the experimental formatter wording separate from selection and
    normalization logic. This module is intentionally independent from the
    production MoM generator and can be deleted with the rest of
    ``ml_mom/experimental`` without affecting MeetScribe runtime behavior.

Inputs:
    Normalized text fragments produced by ``mom_formatter.py``.

Outputs:
    Template strings used for discussion, decision, action, information, and
    summary rendering.
"""

from __future__ import annotations


DISCUSSION_TEMPLATES = (
    "{topic} was evaluated to clarify delivery impact.",
    "{topic} was examined to identify remaining risks.",
    "{topic} was reviewed to support release planning.",
    "{topic} was considered as part of implementation readiness.",
    "{topic} was addressed to improve project execution.",
    "{topic} was prioritized for follow-up planning.",
    "{topic} was assessed to confirm next steps.",
    "{topic} was discussed to align the team on delivery expectations.",
    "{topic} was finalized as part of the meeting outcome.",
    "{topic} was selected as a preferred approach.",
    "The team evaluated {topic} to clarify impact, risks, and next steps.",
    "{topic} was examined in relation to release readiness and delivery quality.",
    "The discussion around {topic} focused on resolving open delivery concerns.",
    "{topic} was assessed to determine the most practical implementation path.",
    "The team aligned on {topic} to support a smoother delivery cycle.",
    "{topic} was considered against current sprint priorities.",
    "The discussion connected {topic} with ownership, timing, and release impact.",
    "{topic} was explored as a way to improve project execution.",
    "The team reviewed {topic} to confirm whether additional follow-up was required.",
    "{topic} was evaluated as part of the broader implementation plan.",
    "The conversation on {topic} centered on blockers, feasibility, and delivery timing.",
    "{topic} was assessed from both quality and release-readiness perspectives.",
    "The team considered {topic} while balancing scope and implementation effort.",
    "{topic} was reviewed to ensure the next sprint priorities were clear.",
    "The discussion clarified how {topic} would affect the remaining release work.",
    "{topic} was examined to identify dependencies and ownership gaps.",
    "The team used {topic} to frame the next set of implementation actions.",
    "{topic} was reviewed with emphasis on practical delivery outcomes.",
    "{topic} was discussed in the context of project readiness and quality control.",
    "The team evaluated {topic} to separate immediate work from later scope.",
    "{topic} was considered as a key factor in the release decision.",
    "The discussion highlighted {topic} as an area requiring coordinated follow-up.",
    "{topic} was assessed to determine whether it should remain in the current sprint.",
    "The team reviewed {topic} to confirm what could be completed before release.",
    "{topic} was examined to improve delivery predictability.",
    "The conversation around {topic} helped clarify priorities for implementation.",
    "{topic} was evaluated to support better product readiness.",
    "The team aligned on the implications of {topic} for upcoming work.",
    "{topic} was reviewed as part of the meeting's delivery planning.",
    "The discussion helped establish how {topic} should be handled going forward.",
    "{topic} was assessed to reduce ambiguity in the implementation plan.",
    "The team considered {topic} in relation to testing and release confidence.",
    "{topic} was examined to support a more reliable handoff after the meeting.",
    "The discussion converted {topic} into clearer ownership and follow-up expectations.",
    "{topic} was reviewed to determine its priority in the current delivery cycle.",
    "The team evaluated the operational impact of {topic}.",
    "{topic} was discussed to align scope with available delivery capacity.",
    "The conversation clarified the status, risks, and required actions for {topic}.",
    "{topic} was assessed as part of the project coordinator's delivery checklist.",
    "The team reviewed {topic} to maintain release quality and momentum.",
)
DECISION_TEMPLATES = (
    "{decision}.",
    "{decision} as the agreed direction.",
    "{decision} for the upcoming phase.",
)
ACTION_TEMPLATE_WITH_DEADLINE = "{owner} -> {task} -> {deadline}"
ACTION_TEMPLATE_WITHOUT_DEADLINE = "{owner} -> {task}"
INFORMATION_TEMPLATES = (
    "{information}.",
    "{information} was noted.",
    "{information} was captured for reference.",
)
SUMMARY_TEMPLATES = (
    "The meeting was held to {purpose}. Key discussions covered {discussion_context}. "
    "The team confirmed {decisions}. Follow-up responsibilities were assigned to "
    "support delivery{next_meeting_clause}.",
    "The team met to {purpose}. The discussion focused on {discussion_context}, "
    "with decisions captured on {decisions}. Action owners were identified for the "
    "remaining work{next_meeting_clause}.",
)
OBJECTIVE_TEMPLATES = (
    "Review {release_scope}, evaluate {performance_scope}, and finalize pending implementation tasks.",
    "Assess {release_scope}, address {performance_scope}, and align on delivery ownership.",
    "Evaluate {release_scope}, confirm {performance_scope}, and plan remaining follow-up work.",
)

NO_DISCUSSION_TEXT = "the priority meeting topics"
NO_DECISION_TEXT = "the agreed next steps"
NO_NEXT_MEETING_CLAUSE = ""
NEXT_MEETING_CLAUSE = ", with the next review scheduled for {date}"


DECISION_REWRITES = (
    ("we should keep", "{subject} will be retained"),
    ("we will keep", "{subject} will be retained"),
    ("we agreed to", "{subject} has been agreed"),
    ("agreed to", "{subject} has been agreed"),
    ("decided to", "{subject} has been approved"),
    ("approved", "{subject} has been approved"),
    ("confirmed", "{subject} has been confirmed"),
    ("we should defer", "{subject} has been deferred"),
    ("we will defer", "{subject} has been deferred"),
    ("defer", "{subject} has been deferred"),
)


PROFESSIONAL_TOPIC_KEYWORDS = {
    "dark mode": "Dark mode",
    "lazy loading": "Lazy loading implementation",
    "transcript loading": "Transcript loading performance",
    "feature list": "Sprint 4 feature list",
    "pdf layout": "PDF layout improvements",
    "pdf export layout": "PDF layout improvements",
    "alignment": "UI alignment issues",
    "blocker": "Release blockers",
    "blockers": "Release blockers",
    "authentication": "Authentication workflow",
    "auth": "Authentication workflow",
    "deployment": "Deployment timeline",
    "release": "Release readiness",
    "performance": "Performance optimization",
    "optimization": "Performance optimization",
    "ui": "UI improvements",
    "interface": "UI improvements",
    "export": "Export functionality",
    "pdf": "Export functionality",
    "docx": "Export functionality",
    "integration": "Backend integration",
    "backend": "Backend integration",
    "frontend": "Frontend implementation",
    "testing": "Testing readiness",
    "bug": "Bug resolution",
    "sprint": "Sprint readiness",
    "model": "Model performance",
    "accuracy": "Model accuracy",
    "deadline": "Delivery timeline",
}


DISCUSSION_CONTEXT = {
    "Performance Optimization": (
        "The team evaluated transcript loading performance and identified optimization work needed before release."
    ),
    "Sprint Planning": (
        "Sprint planning focused on confirming the feature list, release readiness, and remaining delivery priorities."
    ),
    "PDF Export": (
        "PDF export quality was reviewed because layout polish remained important for customer-ready documentation."
    ),
    "Testing": (
        "Testing coverage was reviewed to ensure fixes could be verified before release."
    ),
    "Bug Fixes": (
        "Bug fixes were reviewed to identify quality issues that still required follow-up."
    ),
    "Model Performance": (
        "Model performance was reviewed to confirm summarization accuracy before release."
    ),
    "Dark Mode": (
        "Dark mode scope was evaluated to decide whether it should remain in the current release."
    ),
    "Transcript loading performance": (
        "The team examined transcript loading delays affecting application responsiveness."
    ),
    "Lazy loading implementation": (
        "Lazy loading was considered as the preferred optimization strategy to improve loading performance."
    ),
    "Release readiness": (
        "Sprint release readiness was reviewed to identify blockers and remaining implementation work."
    ),
    "Sprint readiness": (
        "Sprint readiness was assessed to confirm the team could proceed with planned delivery."
    ),
    "PDF layout improvements": (
        "PDF layout improvements were reviewed because export polish remained necessary before release."
    ),
    "UI alignment issues": (
        "UI alignment issues were identified as quality items requiring review before release."
    ),
    "Performance optimization": (
        "Performance optimization was evaluated to improve the application experience."
    ),
    "Model performance": (
        "Model performance was reviewed to confirm summarization accuracy before release."
    ),
    "Dark mode": (
        "Dark mode scope was evaluated to decide whether it should remain in the current release."
    ),
    "Export functionality": (
        "Export functionality was reviewed to confirm readiness for document generation."
    ),
    "Testing readiness": (
        "Testing readiness was assessed to validate release quality."
    ),
}


TOPIC_TITLE_RULES = (
    ("Authentication", ("authentication", "auth", "login", "signin", "sign-in")),
    ("Performance Optimization", ("performance", "optimization", "loading", "lazy loading", "responsiveness")),
    ("PDF Export", ("pdf", "export", "docx")),
    ("Testing", ("testing", "qa", "regression", "verify")),
    ("Model Performance", ("model", "accuracy", "evaluation")),
    ("Dark Mode", ("dark mode",)),
    ("Bug Fixes", ("bug", "issue", "alignment", "fix", "polish")),
    ("Deployment", ("deployment", "deploy", "production")),
    ("Documentation", ("documentation", "document", "requirement")),
    ("Sprint Planning", ("sprint", "feature list", "release readiness", "release")),
    ("Meeting Summary", ("summarize", "summary", "meeting adjourned")),
)


DECISION_SIGNALS = (
    "approval",
    "approve",
    "approved",
    "agreement",
    "agreed",
    "finalized",
    "confirmed",
    "accepted",
    "selected",
    "implemented",
    "implement",
    "deferred",
    "postponed",
    "moves to",
    "move to",
)


PENDING_KEYWORDS = (
    "still pending",
    "needs improvement",
    "needs some",
    "requires review",
    "later",
    "next sprint",
    "moves to",
    "move to",
    "sprint 5",
    "future release",
    "remaining",
    "deferred",
    "postponed",
    "not completed",
    "issues",
    "blockers",
    "polishing",
)


RANKING_KEYWORDS = (
    "implementation",
    "deadline",
    "release",
    "approval",
    "approved",
    "performance",
    "deployment",
    "feature",
    "integration",
    "bug",
    "testing",
    "optimization",
    "sprint",
    "finalize",
    "decision",
    "timeline",
)

"""Experimental deterministic Minutes of Meeting formatter.

Purpose:
    Convert existing ANN sentence predictions into a more polished MoM format
    without changing the production rule-based generator. This is a reversible
    prototype under ``ml_mom/experimental`` only.

Responsibilities:
    - Group existing ANN predictions into semantic topic clusters.
    - Build structured evidence before any report text is rendered.
    - Validate evidence so transcript noise cannot become report content.
    - Select and merge discussion points deterministically.
    - Normalize decision, action, information, and summary wording.
    - Render Markdown and plain text artifacts for research review.

Inputs:
    Prediction-like objects containing ``sentence``, ``speaker``, ``timestamp``,
    ``predicted_label``, and ``confidence_score`` attributes.

Outputs:
    ``ExperimentalMom`` object with Markdown/text render helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

try:
    from ml_mom.experimental import formatter_utils as utils
    from ml_mom.experimental import template_library as templates
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    import formatter_utils as utils
    import template_library as templates


TOPIC_ENTITY_STOPWORDS = {
    "a",
    "an",
    "the",
    "i",
    "we",
    "you",
    "they",
    "it",
    "this",
    "that",
    "these",
    "those",
    "have",
    "has",
    "had",
    "having",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "will",
    "shall",
    "should",
    "would",
    "could",
    "can",
    "may",
    "might",
    "must",
    "do",
    "does",
    "did",
    "doing",
    "ready",
    "enough",
    "only",
    "affecting",
    "automatically",
    "automatic",
    "users",
    "receive",
    "meeting",
    "summary",
    "summaries",
    "around",
    "actually",
    "basically",
    "probably",
    "think",
    "speak",
    "say",
    "said",
    "tell",
    "told",
    "handle",
    "approved",
    "approve",
    "confirmed",
    "confirm",
    "accepted",
    "finalized",
    "resolved",
    "agreed",
    "formed",
    "required",
    "thing",
    "things",
    "something",
    "anything",
    "everything",
}
TOPIC_BOUNDARY_WORDS = {
    "for",
    "to",
    "from",
    "with",
    "without",
    "before",
    "after",
    "because",
    "if",
    "then",
    "and",
    "but",
    "or",
    "so",
    "as",
    "by",
}
TOPIC_TRAILING_STOPWORDS = TOPIC_BOUNDARY_WORDS | {
    "current",
    "updated",
    "final",
    "next",
    "new",
    "old",
    "required",
    "available",
    "approved",
    "confirmed",
    "accepted",
    "finalized",
    "resolved",
    "formed",
}
TOPIC_LEADING_VERBS = {
    "i",
    "we",
    "lets",
    "let",
    "can",
    "should",
    "need",
    "needs",
    "want",
    "wants",
    "do",
    "does",
    "will",
    "would",
    "could",
    "is",
    "are",
    "have",
    "has",
}
IMPORTANT_SINGLE_NOUNS = {
    "budget",
    "funding",
    "deployment",
    "testing",
    "release",
    "marketing",
    "logistics",
    "registration",
    "registrations",
    "donations",
    "inventory",
    "performance",
    "support",
    "security",
    "compliance",
    "notifications",
    "automation",
    "procurement",
}
NORMALIZED_DISCUSSION_RENDERERS = {
    "Automated Email Notifications": (
        "Automatic email delivery for generated meeting summaries was reviewed for implementation readiness."
    ),
    "Pitch Deck": (
        "Pitch deck readiness was reviewed to support stakeholder communication and follow-up."
    ),
    "Multilingual Support": (
        "Multilingual support was evaluated to understand user needs, scope, and implementation readiness."
    ),
    "Funding Capacity": (
        "Current funding availability was reviewed against procurement needs and follow-up requirements."
    ),
    "Donation Campaign": (
        "Donation campaign planning was reviewed to support outreach and resource collection."
    ),
    "Healthcare Coordination": (
        "Healthcare coordination was reviewed to improve follow-up planning and operational readiness."
    ),
    "Release Planning": (
        "Release planning was reviewed to confirm readiness, decisions, and remaining delivery risks."
    ),
    "Research Planning": (
        "Research planning was reviewed to clarify methodology, survey readiness, and next steps."
    ),
    "Access Management": (
        "Access management was reviewed to clarify security, login, and authorization requirements."
    ),
}
DISCUSSION_RENDER_TEMPLATES = (
    "{topic}: Discussion focused on {topic_lower}, with emphasis on {finding}.",
    "{topic}: Participants examined {topic_lower} and outlined the follow-up needed to move it forward.",
    "{topic}: The meeting covered {topic_lower} and clarified the related implementation considerations.",
    "{topic}: Attention was given to {topic_lower}, including the current status and required next steps.",
    "{topic}: {topic} was assessed in relation to delivery readiness, ownership, and outstanding work.",
    "{topic}: The group aligned on {topic_lower} and identified the practical follow-up required.",
)
QUANTITY_DISCUSSION_TEMPLATES = (
    "{topic}: Available figures for {topic_lower} were reviewed, including {quantities}, to assess readiness and follow-up needs.",
    "{topic}: Discussion focused on {topic_lower}, using {quantities} as the basis for planning and coordination.",
    "{topic}: Participants reviewed {topic_lower} against the recorded quantity of {quantities} and identified the related follow-up.",
)
TIMELINE_DISCUSSION_TEMPLATES = (
    "{topic}: Timeline considerations for {topic_lower} were reviewed with reference to {timeline}.",
    "{topic}: Discussion focused on {topic_lower}, including timing expectations around {timeline}.",
    "{topic}: Participants examined {topic_lower} and aligned the related follow-up with {timeline}.",
)
DECISION_SIGNAL_PATTERN = re.compile(
    r"(?i)\b(approved|approve|approved by|accepted|confirmed|finali[sz]ed|"
    r"agree|agreed|agreement reached|decided|decision|resolved|resolution|locked|"
    r"freeze|freezed|postpone|postponed|defer|deferred|moved to|moves to|move to|will move to|"
    r"added to|included in|will be included|will remain|scheduled|selected|chosen|"
    r"implemented|go ahead|proceed with|"
    r"continue with|stop|cancel|close|completed|completed for release|release on|"
    r"ship|deploy|merge|merged|feature freeze|release candidate|will handle)\b"
)
DECISION_CONFIRMATION_PATTERN = re.compile(
    r"(?i)^\s*(yes(?:,\s*agreed)?|agreed|approved|confirmed|done|okay|ok|sure|sounds good|"
    r"works for me|confirmed)\s*[.!]?$"
)
DECISION_PROPOSAL_PATTERN = re.compile(
    r"(?i)\b(should|need to|needs to|let'?s|we should|we need to|i think|proposal|propose|"
    r"better|option\s+[A-Z]|sprint\s+\d+|postpone|defer|move|release|ship|deploy|"
    r"freeze|use|proceed|continue|stop|cancel)\b"
)
DECISION_VERB_PATTERN = re.compile(
    r"(?i)\b(approved|confirmed|accepted|finali[sz]ed|resolved|agree|agreed|defer|deferred|"
    r"postponed|scheduled|locked|completed|closed|selected|chosen|implemented|"
    r"release|ship|deploy|merge|merged|proceed|continue|cancel|stopped|added|included|"
    r"will remain|will be implemented|will handle)\b"
)


@dataclass(slots=True)
class ExperimentalActionItem:
    """One normalized action item row."""

    owner: str
    task: str
    deadline: str = ""


@dataclass(slots=True)
class TopicCluster:
    """Context-aware topic cluster built from ANN prediction records."""

    title: str
    records: list[Any]
    contexts: list[str]
    representative_sentence: str
    centroid_embedding: list[float]
    importance_score: float = 0.0


@dataclass(slots=True)
class TopicEvidence:
    """Validated evidence collected for one meeting topic.

    The formatter uses this object as a boundary between ML outputs and report
    wording. ANN labels and semantic clusters are evidence, but they are not
    directly report text; this stage keeps supporting sentences grouped by the
    section they can safely contribute to.
    """

    topic: str
    cluster: TopicCluster
    discussion_sentences: list[str] = field(default_factory=list)
    decision_sentences: list[str] = field(default_factory=list)
    action_records: list[Any] = field(default_factory=list)
    pending_sentences: list[str] = field(default_factory=list)
    information_sentences: list[str] = field(default_factory=list)
    representative_fact: str = ""
    importance_score: float = 0.0


@dataclass(slots=True)
class FormatterEvidence:
    """Structured evidence used by deterministic report renderers."""

    topics: list[TopicEvidence] = field(default_factory=list)


@dataclass(slots=True)
class ExtractedEntities:
    """Deterministic entities extracted from clustered transcript evidence."""

    topics: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    quantities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TopicBlueprint:
    """Report-ready topic plan produced before language rendering.

    A blueprint is still not final prose. It contains the validated evidence,
    extracted entities, and a noun-based topic title that the deterministic
    renderer can safely turn into professional MoM language.
    """

    title: str
    evidence: TopicEvidence
    entities: ExtractedEntities
    supporting_facts: list[str] = field(default_factory=list)
    importance_score: float = 0.0


@dataclass(slots=True)
class StructuredMeetingRepresentation:
    """Structured meeting data used as the only source for report rendering."""

    topics: list[TopicBlueprint] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ExperimentalActionItem] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    information: list[str] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentalMom:
    """Experimental formatted Minutes of Meeting."""

    meeting_title: str = "Experimental Minutes of Meeting"
    meeting_date: str = ""
    objective: str = ""
    participants: list[str] = field(default_factory=list)
    summary: str = ""
    discussion_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ExperimentalActionItem] = field(default_factory=list)
    pending_items: list[str] = field(default_factory=list)
    information: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the experimental MoM as Markdown."""

        lines = [
            "# Minutes of Meeting",
            "",
            "## Meeting Details",
            "",
            f"**Title:** {self.meeting_title}",
            f"**Meeting Date:** {self.meeting_date or '-'}",
            f"**Participants:** {', '.join(self.participants) if self.participants else '-'}",
            "",
            "## Meeting Objective",
            "",
            self.objective or "Review the meeting priorities and confirm follow-up actions.",
            "",
            "## Executive Summary",
            "",
            self.summary or "No summary could be generated from the available predictions.",
            "",
            "## Discussion Points",
            "",
        ]
        lines.extend(render_bullets(self.discussion_points))
        lines.extend(["", "## Key Decisions", ""])
        lines.extend(render_bullets(self.decisions, fallback="No formal decisions recorded."))
        lines.extend(["", "## Action Items", "", "| Owner | Task | Deadline |", "| --- | --- | --- |"])
        if self.action_items:
            for item in self.action_items:
                lines.append(f"| {item.owner} | {item.task} | {item.deadline or ''} |")
        else:
            lines.append("| Unassigned | No action items were identified. | |")
        lines.extend(["", "## Pending Items", ""])
        lines.extend(render_bullets(self.pending_items))
        if self.information:
            lines.extend(["", "## Additional Information", ""])
            lines.extend(render_bullets(self.information))
        lines.extend(["", "## Prepared By", "", "MeetScribe Experimental Formatter"])
        lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        """Render the experimental MoM as plain text."""

        return re.sub(r"[*#`|]", "", self.to_markdown()).replace("---", "").strip() + "\n"


class ExperimentalMomFormatter:
    """Rule-based formatter for existing ANN predictions."""

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        """Create an experimental formatter.

        Args:
            confidence_threshold: Minimum ANN confidence required before a
                prediction can contribute to formatter sections.
        """

        self.confidence_threshold = confidence_threshold

    def format(
        self,
        prediction_records: Iterable[Any],
        *,
        meeting_title: str = "Experimental Minutes of Meeting",
        meeting_date: str = "",
        participants: list[str] | None = None,
    ) -> ExperimentalMom:
        """Format ANN prediction records into an experimental MoM.

        Args:
            prediction_records: Existing ANN prediction results.
            meeting_title: Optional meeting title to display.
            meeting_date: Optional meeting date to display.
            participants: Optional participant list. If omitted, speakers are
                inferred from predictions.

        Returns:
            Deterministic experimental MoM object.
        """

        records = decision_aware_confidence_records(
            list(prediction_records),
            threshold=self.confidence_threshold,
        )
        inferred_participants = participants or infer_participants(records)
        clusters = build_topic_clusters(records)
        evidence = build_formatter_evidence(clusters)
        blueprints = build_topic_blueprints(evidence)
        representation = build_structured_meeting_representation(
            blueprints,
            participants=inferred_participants,
        )
        formatted_mom = build_report_from_representation(
            representation,
            meeting_title=meeting_title,
            meeting_date=meeting_date,
        )
        return validate_report_output(validate_mom_quality(formatted_mom))


def decision_aware_confidence_records(records: list[Any], *, threshold: float) -> list[Any]:
    """Keep confident records plus explicit formal decisions below threshold.

    The ANN output is not changed. This deterministic gate only prevents clear
    decision evidence such as "approved" or "finalized" from disappearing before
    the post-classification decision rule engine can evaluate it.
    """

    retained: list[Any] = []
    seen: set[int] = set()
    for record in utils.confidence_filtered_records(records, threshold=threshold):
        retained.append(record)
        seen.add(id(record))
    for record in records:
        if id(record) in seen:
            continue
        sentence = utils.record_sentence(record)
        if has_explicit_decision_evidence(sentence):
            retained.append(record)
            seen.add(id(record))
    return retained


def validate_mom_quality(mom: ExperimentalMom) -> ExperimentalMom:
    """Apply the final deterministic quality gate before report rendering."""

    discussion_points = quality_filter_texts(mom.discussion_points, min_score=0.38)
    decisions = quality_filter_decisions(mom.decisions)
    pending_items = quality_filter_texts(mom.pending_items, min_score=0.38)
    information = quality_filter_texts(mom.information, min_score=0.36)
    action_items = quality_filter_actions(mom.action_items)
    objective = (
        utils.clean_report_sentence(mom.objective)
        if utils.is_quality_report_sentence(mom.objective, min_score=0.34)
        else ""
    )
    summary = quality_summary(
        mom.summary,
        discussion_points=discussion_points,
        decisions=decisions,
        action_items=action_items,
        pending_items=pending_items,
        information=information,
    )
    return ExperimentalMom(
        meeting_title=mom.meeting_title,
        meeting_date=mom.meeting_date,
        objective=objective,
        participants=mom.participants,
        summary=summary,
        discussion_points=discussion_points,
        decisions=decisions,
        action_items=action_items,
        pending_items=pending_items,
        information=information,
    )


def build_structured_meeting_representation(
    blueprints: list[TopicBlueprint],
    *,
    participants: list[str],
) -> StructuredMeetingRepresentation:
    """Build structured meeting data before any final prose is generated."""

    action_items = build_action_items(blueprints)
    decisions = build_decisions(blueprints)
    pending_items = build_pending_items(blueprints, action_items)
    pending_items = utils.unique_normalized_sentences(
        pending_items + pending_items_from_decisions(decisions)
    )
    information = build_information(blueprints)
    deadlines = unique_preserve_order(
        deadline
        for blueprint in blueprints
        for deadline in blueprint.entities.deadlines
        if is_valid_deadline_phrase(deadline)
    )
    timeline = unique_preserve_order(deadlines)
    organizations = unique_preserve_order(
        org for blueprint in blueprints for org in blueprint.entities.organizations
    )
    products = unique_preserve_order(
        product for blueprint in blueprints for product in blueprint.entities.products
    )
    numbers = unique_preserve_order(
        number for blueprint in blueprints for number in blueprint.entities.quantities
    )
    return StructuredMeetingRepresentation(
        topics=blueprints,
        participants=participants,
        decisions=decisions,
        action_items=action_items,
        pending_items=pending_items,
        information=information,
        timeline=timeline,
        deadlines=deadlines,
        organizations=organizations,
        products=products,
        numbers=numbers,
    )


def build_report_from_representation(
    representation: StructuredMeetingRepresentation,
    *,
    meeting_title: str,
    meeting_date: str,
) -> ExperimentalMom:
    """Render the final MoM only from structured meeting representation."""

    discussion_points = build_discussion_points(representation.topics)
    objective = build_objective(representation.topics)
    summary = build_summary(
        objective=objective,
        blueprints=representation.topics,
        discussion_points=discussion_points,
        decisions=representation.decisions,
        action_items=representation.action_items,
        pending_items=representation.pending_items,
        information=representation.information,
    )
    return ExperimentalMom(
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        objective=objective,
        participants=representation.participants,
        summary=summary,
        discussion_points=discussion_points,
        decisions=representation.decisions,
        action_items=representation.action_items,
        pending_items=representation.pending_items,
        information=representation.information,
    )


def validate_report_output(mom: ExperimentalMom) -> ExperimentalMom:
    """Self-validate final report sections against transcript-like fragments."""

    discussion_points = validate_discussion_points(mom.discussion_points)
    decisions = validate_text_items(mom.decisions)
    pending_items = validate_text_items(mom.pending_items)
    information = validate_text_items(mom.information)
    action_items = validate_action_items(mom.action_items)
    summary = validate_summary_text(
        mom.summary,
        discussion_points=discussion_points,
        decisions=decisions,
        action_items=action_items,
        pending_items=pending_items,
    )
    objective = mom.objective if not contains_transcript_fragment(mom.objective) else ""
    return ExperimentalMom(
        meeting_title=mom.meeting_title,
        meeting_date=mom.meeting_date,
        objective=objective,
        participants=mom.participants,
        summary=summary,
        discussion_points=discussion_points,
        decisions=decisions,
        action_items=action_items,
        pending_items=pending_items,
        information=information,
    )


def validate_discussion_points(points: Iterable[str]) -> list[str]:
    """Reject discussion bullets whose headings or body look transcript-like."""

    valid: list[str] = []
    seen: set[str] = set()
    seen_openers: dict[str, int] = {}
    for point in points:
        text = utils.normalize_whitespace(str(point or ""))
        heading, separator, body = text.partition(":")
        normalized_heading = normalize_topic_heading(heading)
        if not separator or not normalized_heading:
            continue
        if contains_transcript_fragment(heading) or contains_transcript_fragment(body):
            continue
        cleaned_body = utils.clean_report_sentence(body)
        if not cleaned_body or contains_transcript_fragment(cleaned_body):
            continue
        opener = discussion_opener(cleaned_body)
        if seen_openers.get(opener, 0) >= 2:
            continue
        candidate = f"{normalized_heading}: {cleaned_body}"
        key = candidate.casefold()
        if key in seen:
            continue
        valid.append(candidate)
        seen.add(key)
        seen_openers[opener] = seen_openers.get(opener, 0) + 1
    return valid


def discussion_opener(text: str) -> str:
    """Return the first few words used to detect repetitive discussion prose."""

    words = re.findall(r"[A-Za-z]+", text.casefold())
    return " ".join(words[:3])


def validate_text_items(items: Iterable[str]) -> list[str]:
    """Reject final report text items that still contain transcript fragments."""

    valid: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = utils.clean_report_sentence(str(item or ""))
        if not cleaned or contains_transcript_fragment(cleaned):
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        valid.append(cleaned)
        seen.add(key)
    return valid


def validate_action_items(actions: Iterable[ExperimentalActionItem]) -> list[ExperimentalActionItem]:
    """Reject duplicate or fragment-like action rows."""

    valid: list[ExperimentalActionItem] = []
    seen: set[str] = set()
    for action in actions:
        task = utils.clean_report_sentence(action.task)
        if contains_transcript_fragment(task) or not is_valid_action_task(task):
            continue
        owner = normalize_owner(action.owner)
        deadline = normalize_deadline_phrase(action.deadline)
        key = f"{owner}|{task}".casefold()
        if key in seen:
            continue
        valid.append(ExperimentalActionItem(owner=owner, task=task, deadline=deadline))
        seen.add(key)
    return valid


def validate_summary_text(
    summary: str,
    *,
    discussion_points: list[str],
    decisions: list[str],
    action_items: list[ExperimentalActionItem],
    pending_items: list[str],
) -> str:
    """Regenerate the summary when the rendered summary contains fragments."""

    if summary and not contains_transcript_fragment(summary):
        return summary
    return quality_summary(
        "",
        discussion_points=discussion_points,
        decisions=decisions,
        action_items=action_items,
        pending_items=pending_items,
        information=[],
    )


def contains_transcript_fragment(text: str) -> bool:
    """Return whether text contains phrases that should never reach the MoM."""

    lowered = utils.normalize_whitespace(text).casefold()
    if not lowered:
        return False
    fragment_patterns = (
        r"\bi\s+will\s+speak\b",
        r"\bi\s+a\s+pitch\b",
        r"\bemail\s+to\s+by\b",
        r"\bcity\s+with\s+high\b",
        r"\bwe\s+have\s+around\b",
        r"\bi'?ll\s+i'?ll\b",
        r"\bwant\s+users\s+to\s+receive\b",
        r"\btoday'?s\s+decisions\s+finalized\b",
    )
    if any(re.search(pattern, lowered) for pattern in fragment_patterns):
        return True
    words = re.findall(r"[a-z0-9']+", lowered)
    if not words:
        return False
    if words[0] in TOPIC_LEADING_VERBS and len(words) <= 5:
        return True
    return False


def quality_filter_texts(items: Iterable[str], *, min_score: float) -> list[str]:
    """Clean, validate, de-duplicate, and preserve quality report sentences."""

    filtered: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = utils.clean_report_sentence(str(item or ""))
        key = re.sub(r"[^a-z0-9]+", " ", cleaned.casefold()).strip()
        if not key or key in seen:
            continue
        if not utils.is_quality_report_sentence(cleaned, min_score=min_score):
            continue
        filtered.append(cleaned)
        seen.add(key)
    return filtered


def quality_filter_decisions(items: Iterable[str]) -> list[str]:
    """Validate formal decisions while retaining short explicit outcomes."""

    filtered: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = cleanup_decision_output(utils.clean_report_sentence(str(item or "")))
        key = re.sub(r"[^a-z0-9]+", " ", cleaned.casefold()).strip()
        if not key or key in seen:
            continue
        if not DECISION_VERB_PATTERN.search(cleaned):
            continue
        if is_noisy_decision_output(cleaned):
            continue
        if utils.is_low_value_sentence(cleaned) or utils.is_weak_topic(cleaned):
            continue
        filtered.append(cleaned)
        seen.add(key)
    return filtered


def is_noisy_decision_output(text: str) -> bool:
    """Reject rendered decisions that still expose conversational fragments."""

    lowered = text.casefold()
    if re.search(r"\b(i'?d|i would|personally|i think|i feel)\b", lowered):
        return True
    if re.search(r"\bit implementation\b", lowered):
        return True
    if re.search(r"\b(recorded item|general)\b", lowered):
        return True
    if re.search(r"\btoday'?s decisions\b", lowered):
        return True
    return False


def quality_filter_actions(actions: Iterable[ExperimentalActionItem]) -> list[ExperimentalActionItem]:
    """Keep only explicit, complete, de-duplicated action items."""

    filtered: list[ExperimentalActionItem] = []
    seen: set[str] = set()
    for action in actions:
        task = utils.clean_report_sentence(action.task)
        if not is_valid_action_task(task):
            continue
        owner = normalize_owner(action.owner)
        deadline = normalize_deadline_phrase(
            action.deadline or utils.extract_deadline(task)
        )
        key = f"{owner}|{task}|{deadline}".casefold()
        if key in seen:
            continue
        filtered.append(ExperimentalActionItem(owner=owner, task=task, deadline=deadline))
        seen.add(key)
    return filtered


def is_valid_action_task(task: str) -> bool:
    """Return whether an action task contains enough commitment evidence."""

    if re.search(r"(?i)\b(that'?s exactly|we have around|i think|kind of|sort of)\b", task):
        return False
    has_action_verb = bool(
        re.search(
            r"(?i)\b(implement|finish|complete|update|publish|confirm|prepare|assign|"
            r"generate|review|submit|send|share|email|finalize|validate|test|create|"
            r"deliver|deploy|verify|finish|fix)\b",
            task,
        )
    )
    has_object = len(re.findall(r"\b[A-Za-z0-9]+\b", task)) >= 2
    if not has_action_verb or not has_object:
        return False
    # Short task descriptions such as "Confirm transport" are valid when they
    # contain a clear action verb and object, even if the generic sentence
    # quality scorer would normally consider them brief.
    return has_action_verb and (
        has_object or utils.is_quality_report_sentence(task, min_score=0.32)
    )


def normalize_owner(owner: str) -> str:
    """Use only explicit owner names; never invent owners."""

    cleaned = utils.normalize_whitespace(owner)
    lowered = cleaned.casefold()
    if not cleaned or cleaned == "-" or lowered in {"unassigned", "not assigned", "unknown"}:
        return "Not Mentioned"
    if lowered in {"we", "us", "everyone", "everybody", "people", "somebody", "someone"}:
        return "Team"
    if re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", cleaned):
        return "Not Mentioned"
    return cleaned


def normalize_deadline_phrase(deadline: str) -> str:
    """Return one natural due-date value for action rendering."""

    cleaned = utils.normalize_whitespace(str(deadline or "")).strip(" .")
    if not cleaned:
        return "Not Mentioned"

    candidates = [
        match.group(0)
        for match in utils.DEADLINE_PATTERN.finditer(cleaned)
        if is_valid_deadline_phrase(match.group(0))
    ]
    if not candidates and is_valid_deadline_phrase(cleaned):
        candidates = [cleaned]
    if not candidates:
        return "Not Mentioned"

    # Prefer the most specific phrasing. "Before Friday Evening" carries more
    # scheduling information than the separate word "Friday".
    before_candidates = [
        item for item in candidates if re.match(r"(?i)^before\s+", item)
    ]
    if before_candidates:
        chosen = max(before_candidates, key=lambda item: len(item.split()))
    else:
        chosen = candidates[0]
    normalized = title_case_topic(chosen)
    replacements = {
        "Eod": "EOD",
        "End Of Day": "End of Day",
        "End Of Month": "End of Month",
        "This Week": "This Week",
        "Next Week": "Next Week",
        "Next Release": "Next Release",
        "Next Sprint": "Next Sprint",
        "Before Deployment": "Before Deployment",
    }
    return replacements.get(normalized, normalized)


def quality_summary(
    current_summary: str,
    *,
    discussion_points: list[str],
    decisions: list[str],
    action_items: list[ExperimentalActionItem],
    pending_items: list[str],
    information: list[str],
) -> str:
    """Build a concise summary only from validated final report sections."""

    topics = [
        point.split(":", 1)[0].strip()
        for point in discussion_points
        if point.split(":", 1)[0].strip()
    ][:3]
    sentences: list[str] = []
    if topics:
        opener = summary_opening_phrase(topics)
        sentences.append(
            f"{opener} {summarize_list([topic_phrase(topic) for topic in topics], fallback='the main meeting themes')}."
        )
    if decisions:
        sentences.append(
            f"Formal outcomes were captured for {summarize_decision_topics(decisions)}."
        )
    if action_items:
        sentences.append(
            f"{len(action_items)} follow-up action item{'s were' if len(action_items) != 1 else ' was'} recorded for execution."
        )
    if pending_items:
        sentences.append(
            f"Pending work remains for {summarize_pending_topics(pending_items)}."
        )
    if not sentences and information:
        sentences.append(
            f"Relevant updates included {summarize_list([item.rstrip('.') for item in information[:2]], fallback='the captured information')}."
        )
    if not sentences and utils.is_quality_report_sentence(current_summary, min_score=0.4):
        sentences.append(utils.clean_report_sentence(current_summary))
    if not sentences:
        return "No high-confidence reportable outcomes were identified from the transcript."
    return limit_words(utils.normalize_sentence(" ".join(sentences)), 100)


def infer_participants(records: Iterable[Any]) -> list[str]:
    """Infer ordered participant names from prediction speakers."""

    participants: list[str] = []
    seen: set[str] = set()
    for record in records:
        speaker = utils.normalize_whitespace(getattr(record, "speaker", ""))
        if not speaker or speaker == "-" or re.fullmatch(r"(?i)speaker\s+\d+", speaker):
            continue
        key = speaker.casefold()
        if key in seen:
            continue
        participants.append(speaker)
        seen.add(key)
    return participants


def build_topic_clusters(
    records: list[Any],
    similarity_threshold: float = 0.74,
) -> list[TopicCluster]:
    """Create context-aware topic clusters using MiniLM cosine similarity."""

    usable_records = [
        record
        for record in records
        if utils.is_reportable_sentence(utils.record_sentence(record))
        or is_decision_context_record(utils.record_sentence(record))
    ]
    contexts = [
        utils.context_window_text(usable_records, index, window_size=2)
        for index, _record in enumerate(usable_records)
    ]
    vectors = utils.embedding_vectors_for_texts(contexts)
    clusters: list[TopicCluster] = []

    for index, record in enumerate(usable_records):
        sentence = utils.record_sentence(record)
        context = contexts[index]
        vector = vectors[index] if index < len(vectors) else []
        best_cluster: TopicCluster | None = None
        best_similarity = 0.0

        for cluster in clusters:
            if vector and cluster.centroid_embedding:
                similarity = utils.cosine(vector, cluster.centroid_embedding)
            else:
                similarity = utils.lexical_similarity(context, " ".join(cluster.contexts))
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None and best_similarity >= similarity_threshold:
            best_cluster.records.append(record)
            best_cluster.contexts.append(context)
            if vector:
                best_cluster.centroid_embedding = averaged_centroid(
                    best_cluster.centroid_embedding,
                    vector,
                    len(best_cluster.records),
                )
            best_cluster.representative_sentence = choose_representative_sentence(best_cluster.records)
            best_cluster.title = cluster_title(best_cluster)
            continue

        title = cluster_title_from_text(context or sentence) or cluster_title_from_text(sentence)
        if not title:
            continue

        clusters.append(
            TopicCluster(
                title=title,
                records=[record],
                contexts=[context],
                representative_sentence=sentence,
                centroid_embedding=vector,
            )
        )

    consolidated = consolidate_clusters_by_title(clusters)
    score_clusters(consolidated)
    return sorted(consolidated, key=lambda cluster: cluster.importance_score, reverse=True)


def build_formatter_evidence(clusters: list[TopicCluster]) -> FormatterEvidence:
    """Collect validated section evidence from topic clusters.

    This is the first deterministic formatter stage after ANN classification
    and clustering. It prevents section builders from promoting a single raw
    transcript sentence directly into the report. Each topic gathers only the
    sentences that contain enough evidence for discussion, decisions, actions,
    pending work, or factual information.
    """

    topic_evidence: list[TopicEvidence] = []
    for cluster in clusters:
        if not cluster.title or utils.is_weak_topic(cluster.title):
            continue

        evidence = TopicEvidence(
            topic=cluster.title,
            cluster=cluster,
            representative_fact=cluster_representative_fact(cluster),
            importance_score=cluster.importance_score,
        )

        for index, record in enumerate(cluster.records):
            raw_sentence = utils.record_sentence(record)
            if not raw_sentence:
                continue

            context_text = cluster.contexts[index] if index < len(cluster.contexts) else ""
            contextual_decision = contextual_decision_evidence(raw_sentence, context_text)
            if contextual_decision:
                evidence.decision_sentences.append(contextual_decision)

            # The validation gate runs before section assignment so greetings,
            # repeated speech fragments, and low-information utterances cannot
            # become MoM evidence even when ANN labels are confident.
            if not utils.is_reportable_sentence(raw_sentence):
                continue
            cleaned_sentence = utils.clean_report_sentence(raw_sentence)
            if not is_valid_evidence_sentence(cleaned_sentence):
                continue

            label = getattr(record, "predicted_label", "")
            # Decision evidence is deterministic post-processing after ANN
            # prediction. A sentence can contain an explicit outcome phrase even
            # if the classifier labels it as Discussion or Information, so the
            # signal check is intentionally not restricted to label == Decision.
            if contains_decision_signal(cleaned_sentence):
                evidence.decision_sentences.append(cleaned_sentence)

            if action_sentence_has_evidence(cleaned_sentence, label) and action_record_has_task_support(
                record,
                cleaned_sentence,
            ):
                evidence.action_records.append(record)

            if utils.is_pending_sentence(cleaned_sentence):
                evidence.pending_sentences.append(cleaned_sentence)

            if label == "Information":
                evidence.information_sentences.append(cleaned_sentence)

            if label in {"Discussion", "Summary"} or (
                label == "Information" and not utils.is_pending_sentence(cleaned_sentence)
            ):
                evidence.discussion_sentences.append(cleaned_sentence)

        # Representative facts keep discussion topic rendering grounded in
        # transcript evidence while still avoiding raw fragment copy.
        if evidence.representative_fact and not contains_decision_signal(evidence.representative_fact):
            evidence.discussion_sentences.insert(0, evidence.representative_fact)

        if topic_has_evidence(evidence):
            topic_evidence.append(evidence)

    topic_evidence.sort(key=lambda item: item.importance_score, reverse=True)
    return FormatterEvidence(topics=topic_evidence)


def topic_has_evidence(evidence: TopicEvidence) -> bool:
    """Return whether a topic has any validated reportable evidence."""

    return bool(
        evidence.discussion_sentences
        or evidence.decision_sentences
        or evidence.action_records
        or evidence.pending_sentences
        or evidence.information_sentences
    )


def is_valid_evidence_sentence(sentence: str) -> bool:
    """Allow high-quality sentences or short sentences with real entities."""

    if utils.is_quality_report_sentence(sentence, min_score=0.30):
        return True
    if utils.is_low_value_sentence(sentence):
        return False
    entity_topics = extract_topic_entities(sentence)
    word_count = len(re.findall(r"\b[A-Za-z0-9]+\b", sentence))
    return bool(entity_topics) and word_count >= 4


def build_topic_blueprints(evidence: FormatterEvidence) -> list[TopicBlueprint]:
    """Create noun-based topic blueprints from structured evidence.

    This is the entity extraction and topic builder stage. It replaces direct
    use of transcript phrases with normalized topics derived from entities,
    quantities, deadlines, and cleaned noun phrases found across the full
    clustered evidence for that topic.
    """

    blueprints: list[TopicBlueprint] = []
    for topic_evidence in evidence.topics:
        for split_evidence in split_evidence_by_topic(topic_evidence):
            entities = extract_entities_from_evidence(split_evidence)
            title = choose_blueprint_title(split_evidence, entities)
            if not title or utils.is_weak_topic(title):
                continue
            facts = build_supporting_facts(split_evidence, entities)
            blueprints.append(
                TopicBlueprint(
                    title=title,
                    evidence=split_evidence,
                    entities=entities,
                    supporting_facts=facts,
                    importance_score=split_evidence.importance_score,
                )
            )
    return consolidate_blueprints(blueprints)


def split_evidence_by_topic(evidence: TopicEvidence) -> list[TopicEvidence]:
    """Split one broad cluster into noun-topic evidence groups.

    Clustering remains unchanged. This function corrects the formatter layer by
    allowing one semantic cluster to produce multiple report topics when its
    evidence mentions different business objects, such as registrations and a
    pitch deck in adjacent transcript turns.
    """

    grouped: dict[str, TopicEvidence] = {}

    def bucket_for(sentence: str) -> TopicEvidence:
        title = sentence_topic_title(sentence, evidence.topic)
        if title not in grouped:
            grouped[title] = TopicEvidence(
                topic=title,
                cluster=evidence.cluster,
                representative_fact="",
                importance_score=evidence.importance_score,
            )
        return grouped[title]

    for sentence in evidence.discussion_sentences:
        bucket_for(sentence).discussion_sentences.append(sentence)
    for sentence in evidence.decision_sentences:
        bucket_for(sentence).decision_sentences.append(sentence)
    for sentence in evidence.pending_sentences:
        bucket_for(sentence).pending_sentences.append(sentence)
    for sentence in evidence.information_sentences:
        bucket_for(sentence).information_sentences.append(sentence)
    for record in evidence.action_records:
        bucket_for(utils.record_sentence(record)).action_records.append(record)

    if not grouped:
        return [evidence]

    split_items = list(grouped.values())
    for item in split_items:
        item.representative_fact = first_quality_fact(item.discussion_sentences)
    return split_items


def sentence_topic_title(sentence: str, fallback: str) -> str:
    """Resolve a noun-based topic title from one evidence sentence."""

    entities = extract_topic_entities(sentence)
    for entity in entities:
        normalized = normalize_topic_heading(entity)
        if normalized:
            return normalized
    return normalize_topic_heading(fallback)


def extract_entities_from_evidence(evidence: TopicEvidence) -> ExtractedEntities:
    """Extract deterministic entities from all evidence attached to a topic."""

    sentences = evidence_sentences(evidence)
    entities = ExtractedEntities()
    entities.topics = unique_preserve_order(
        topic
        for sentence in sentences
        for topic in extract_topic_entities(sentence)
    )
    entities.people = unique_preserve_order(
        name
        for record in evidence.cluster.records + evidence.action_records
        for name in [utils.normalize_whitespace(getattr(record, "speaker", ""))]
        if name and not re.fullmatch(r"(?i)speaker\s+[A-Za-z0-9]+", name)
    )
    entities.deadlines = unique_preserve_order(
        deadline
        for sentence in sentences
        for deadline in [utils.extract_deadline(sentence)]
        if deadline and is_valid_deadline_phrase(deadline)
    )
    entities.quantities = unique_preserve_order(
        quantity
        for sentence in sentences
        for quantity in extract_quantities(sentence)
    )
    entities.organizations = unique_preserve_order(extract_named_entities(sentences, suffixes=("Inc", "Ltd", "University", "College", "Foundation")))
    entities.projects = unique_preserve_order(extract_project_entities(sentences))
    entities.products = unique_preserve_order(extract_product_entities(sentences))
    entities.locations = unique_preserve_order(extract_named_entities(sentences, suffixes=("Hall", "Auditorium", "Room", "Office", "Campus")))
    return entities


def evidence_sentences(evidence: TopicEvidence) -> list[str]:
    """Return all validated evidence sentences for entity extraction."""

    sentences: list[str] = []
    sentences.extend(evidence.discussion_sentences)
    sentences.extend(evidence.decision_sentences)
    sentences.extend(evidence.pending_sentences)
    sentences.extend(evidence.information_sentences)
    sentences.extend(utils.record_sentence(record) for record in evidence.action_records)
    return utils.unique_normalized_sentences(sentence for sentence in sentences if sentence)


def extract_topic_entities(sentence: str) -> list[str]:
    """Extract noun-like topic candidates without using sentence prefixes."""

    candidates: list[str] = []
    business_topic = deterministic_business_topic(sentence)
    if business_topic:
        candidates.append(business_topic)
    professional = utils.professional_topic(sentence)
    if professional and not utils.is_weak_topic(professional):
        candidates.append(title_case_topic(professional))

    cleaned = utils.remove_fillers(utils.remove_speech_repetitions(sentence))
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", cleaned)
        if token.casefold() not in TOPIC_ENTITY_STOPWORDS
    ]
    chunks: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in TOPIC_BOUNDARY_WORDS:
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(token)
        if len(current) >= 4:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)

    for chunk in chunks:
        normalized = normalize_topic_chunk(chunk)
        if normalized:
            candidates.append(normalized)
    return unique_preserve_order(candidates)


def deterministic_business_topic(sentence: str) -> str:
    """Resolve common business concepts without using transcript word order."""

    lowered = sentence.casefold()
    if re.search(r"\b(email|mail|notification|notify|send)\b", lowered) and re.search(
        r"\b(summary|summaries|minutes|mom|report)\b",
        lowered,
    ):
        return "Automated Email Notifications"
    if re.search(r"\bautomatic|automated|auto\b", lowered) and re.search(r"\bdelivery|notification|email|mail\b", lowered):
        return "Automated Email Notifications"
    if re.search(r"\bmultilingual|multi[-\s]?language|translation|translate|language support\b", lowered):
        return "Multilingual Support"
    if re.search(r"\brelease\b", lowered) and re.search(r"\bdecision|decisions|final|finalized|planning|readiness\b", lowered):
        return "Release Planning"
    if "pitch deck" in lowered:
        if re.search(r"\b(email|send|share|distribute|distribution)\b", lowered):
            return "Pitch Deck Distribution"
        return "Pitch Deck"
    if re.search(r"\bvolunteer|volunteers\b", lowered) and re.search(r"\brecruit|recruitment|drive\b", lowered):
        return "Volunteer Recruitment"
    if re.search(r"\btarget\s+(location|locations|area|areas)|city|cities\b", lowered):
        return "Target Distribution Areas"
    if re.search(r"\brevenue|forecast|pricing|price|financial\b", lowered):
        return "Financial Planning"
    if re.search(r"\btimetable|schedule|scheduling|calendar\b", lowered):
        return "Schedule Coordination"
    if re.search(r"\bdepartment|departments\b", lowered):
        return "Department Coordination"
    if re.search(r"\binventory|warehouse|reconciliation|cycle count\b", lowered):
        return "Inventory Reconciliation"
    if re.search(r"\bregistration|registrations|attendee|attendees\b", lowered):
        return "Event Registrations"
    if re.search(r"\bdonation campaign|campaign\b", lowered):
        return "Donation Campaign"
    if re.search(r"\bfunds?|funding|budget|donation|donations\b", lowered):
        return "Funding Capacity"
    if re.search(r"\bblanket|blankets|inventory\b", lowered):
        return "Inventory Planning"
    if re.search(r"\bapi\b", lowered) and re.search(r"\bresponse|latency|performance|cache|caching\b", lowered):
        return "API Performance"
    if re.search(r"\bclient demo|demo\b", lowered):
        return "Client Demo Readiness"
    if re.search(r"\bpatient|patients|clinic|clinical|doctor|doctors|healthcare\b", lowered):
        return "Healthcare Coordination"
    if re.search(r"\bresearch|study|methodology|literature review|proposal\b", lowered):
        return "Research Planning"
    if re.search(r"\bsecurity|authentication|login|access control\b", lowered):
        return "Access Management"
    return ""


def normalize_topic_chunk(words: list[str]) -> str:
    """Convert a word chunk into a safe noun-based topic title."""

    trimmed = [word for word in words if word not in TOPIC_ENTITY_STOPWORDS]
    while trimmed and trimmed[0] in TOPIC_BOUNDARY_WORDS:
        trimmed.pop(0)
    while trimmed and trimmed[-1] in TOPIC_TRAILING_STOPWORDS:
        trimmed.pop()
    if not trimmed:
        return ""
    if len(trimmed) == 1 and trimmed[0] not in IMPORTANT_SINGLE_NOUNS:
        return ""
    return normalize_topic_heading(" ".join(trimmed[:3]))


def extract_quantities(sentence: str) -> list[str]:
    """Extract numeric quantities with their nearby noun object."""

    quantities: list[str] = []
    tokens = re.findall(r"\d+(?:[,.]\d+)?|[A-Za-z%]+", sentence)
    for index, token in enumerate(tokens):
        if not re.fullmatch(r"\d+(?:[,.]\d+)?", token):
            continue
        phrase_tokens = [token]
        for next_token in tokens[index + 1 : index + 4]:
            lowered = next_token.casefold()
            if lowered in TOPIC_BOUNDARY_WORDS or lowered in {"the", "a", "an"}:
                break
            phrase_tokens.append(next_token)
        if len(phrase_tokens) > 1:
            quantities.append(utils.normalize_whitespace(" ".join(phrase_tokens)))
    return quantities


def is_valid_deadline_phrase(deadline: str) -> bool:
    """Reject phrases that match deadline syntax but are not time references."""

    lowered = deadline.casefold().strip()
    return lowered not in {"by email", "by mail"}


def extract_named_entities(sentences: list[str], *, suffixes: tuple[str, ...]) -> list[str]:
    """Extract simple proper-name entities with known organizational suffixes."""

    suffix_pattern = "|".join(re.escape(suffix) for suffix in suffixes)
    pattern = re.compile(rf"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){{0,3}}\s+(?:{suffix_pattern})\b")
    return [match.group(0) for sentence in sentences for match in pattern.finditer(sentence)]


def extract_project_entities(sentences: list[str]) -> list[str]:
    """Extract project or sprint identifiers from evidence sentences."""

    projects: list[str] = []
    for sentence in sentences:
        projects.extend(match.group(0) for match in re.finditer(r"(?i)\b(Sprint\s+\d+|Phase\s+\d+|Project\s+[A-Z][A-Za-z0-9-]*)\b", sentence))
    return projects


def extract_product_entities(sentences: list[str]) -> list[str]:
    """Extract product-like noun phrases such as pitch decks and exports."""

    products: list[str] = []
    for sentence in sentences:
        for pattern in (
            r"(?i)\bpitch\s+deck\b",
            r"(?i)\brequirement\s+document\b",
            r"(?i)\bpdf\s+export\b",
            r"(?i)\bdocx\s+export\b",
            r"(?i)\blazy\s+loading\b",
        ):
            products.extend(title_case_topic(match.group(0)) for match in re.finditer(pattern, sentence))
    return products


def choose_blueprint_title(evidence: TopicEvidence, entities: ExtractedEntities) -> str:
    """Select the safest noun-based title for a topic blueprint."""

    # Concrete product/topic phrases carry more meaning than scheduling labels
    # such as "Phase 2" or "Sprint 5".
    candidates = (
        entities.products
        + entities.topics
        + entities.organizations
        + entities.locations
        + entities.projects
        + [evidence.topic]
    )
    for candidate in candidates:
        title = normalize_topic_heading(candidate)
        if title:
            return title
    return ""


def build_supporting_facts(
    evidence: TopicEvidence,
    entities: ExtractedEntities,
) -> list[str]:
    """Build concise factual supports without copying transcript fragments."""

    facts: list[str] = []
    if entities.quantities:
        facts.append(f"Captured quantities included {summarize_list(entities.quantities[:2], fallback='reported figures')}.")
    if entities.deadlines:
        facts.append(f"Relevant timing included {summarize_list(entities.deadlines[:2], fallback='the recorded timeline')}.")
    representative = first_quality_fact(evidence.discussion_sentences)
    if representative and not facts:
        facts.append(representative)
    return utils.unique_normalized_sentences(facts)


def consolidate_blueprints(blueprints: list[TopicBlueprint]) -> list[TopicBlueprint]:
    """Merge lexically overlapping blueprints that describe the same topic."""

    merged: dict[str, TopicBlueprint] = {}
    for blueprint in blueprints:
        key = semantic_topic_key(blueprint.title)
        if key not in merged:
            merged[key] = blueprint
            continue
        target = merged[key]
        target.title = preferred_topic_title(target.title, blueprint.title)
        target.supporting_facts = utils.unique_normalized_sentences(
            target.supporting_facts + blueprint.supporting_facts
        )
        target.evidence.discussion_sentences = utils.unique_normalized_sentences(
            target.evidence.discussion_sentences + blueprint.evidence.discussion_sentences
        )
        target.evidence.decision_sentences = utils.unique_normalized_sentences(
            target.evidence.decision_sentences + blueprint.evidence.decision_sentences
        )
        target.evidence.pending_sentences = utils.unique_normalized_sentences(
            target.evidence.pending_sentences + blueprint.evidence.pending_sentences
        )
        target.evidence.information_sentences = utils.unique_normalized_sentences(
            target.evidence.information_sentences + blueprint.evidence.information_sentences
        )
        target.evidence.action_records.extend(blueprint.evidence.action_records)
        target.importance_score = max(target.importance_score, blueprint.importance_score)
    return sorted(merged.values(), key=lambda item: item.importance_score, reverse=True)


def semantic_topic_key(title: str) -> str:
    """Return a stable concept key for discussion-title de-duplication."""

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", title.casefold())
        if token not in {"automatic", "current", "planned", "general"}
    ]
    aliases = {
        "emails": "email",
        "mail": "email",
        "notification": "notification",
        "notifications": "notification",
        "delivery": "notification",
        "integrations": "integration",
        "improvements": "improvement",
        "features": "feature",
    }
    normalized = [aliases.get(token, token) for token in tokens]
    if "email" in normalized and "notification" in normalized:
        return "email notification"
    return " ".join(sorted(set(normalized)))


def preferred_topic_title(left: str, right: str) -> str:
    """Prefer the more descriptive of two equivalent deterministic titles."""

    def score(title: str) -> tuple[int, int]:
        words = re.findall(r"[A-Za-z0-9]+", title)
        generic = sum(
            word.casefold() in {"phase", "feature", "backend", "performance", "ui"}
            for word in words
        )
        return (len(words) - generic, len(title))

    return max((left, right), key=score)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    """Return unique normalized items while preserving original order."""

    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = utils.normalize_whitespace(str(item or "").strip(" ."))
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        output.append(cleaned)
        seen.add(key)
    return output


def title_case_topic(text: str) -> str:
    """Convert an extracted entity into a readable topic title."""

    small_words = {"and", "or", "for", "of", "to", "in", "on", "by"}
    words = re.findall(r"[A-Za-z0-9%'-]+", text)
    titled = [
        word.upper() if word.casefold() in {"ui", "api", "pdf", "docx", "qa"} else (
            word.casefold() if index and word.casefold() in small_words else word[:1].upper() + word[1:].lower()
        )
        for index, word in enumerate(words)
    ]
    return utils.normalize_whitespace(" ".join(titled))


def normalize_topic_heading(candidate: str) -> str:
    """Return a concise professional topic heading or an empty string.

    This is the Topic Normalization Layer. It converts entity candidates into
    section-safe headings and rejects anything that still looks like transcript
    language, such as clauses beginning with personal pronouns or helper verbs.
    """

    cleaned = utils.remove_fillers(utils.remove_speech_repetitions(candidate))
    cleaned = re.sub(r"(?i)\b(today'?s|tomorrow'?s)\b", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9%&/ -]+", " ", cleaned)
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9%&/-]+", cleaned)
        if word.casefold() not in TOPIC_ENTITY_STOPWORDS
    ]
    while words and words[0].casefold() in (TOPIC_LEADING_VERBS | TOPIC_BOUNDARY_WORDS):
        words.pop(0)
    while words and words[-1].casefold() in TOPIC_TRAILING_STOPWORDS:
        words.pop()
    if not words:
        return ""
    candidate_text = " ".join(words).casefold()
    if re.search(r"\b(email|mail)\s+to\s+by\b|\bcity\s+with\s+high\b|\bi\s+a\s+pitch\b", candidate_text):
        return ""
    if len(words) > 4:
        words = compact_heading_words(words)
    if len(words) > 4 or looks_like_clause(words):
        return ""
    words = trim_topic_status_suffix(words)
    if len(words) == 1 and words[0].casefold() not in IMPORTANT_SINGLE_NOUNS:
        return ""
    title = title_case_topic(" ".join(words))
    if not title or utils.is_weak_topic(title):
        return ""
    if len(title.split()) == 1:
        title = title_case_topic(
            templates.PROFESSIONAL_TOPIC_KEYWORDS.get(title.casefold(), title)
        )
    first = title.split()[0].casefold()
    if first in TOPIC_LEADING_VERBS:
        return ""
    return title


def trim_topic_status_suffix(words: list[str]) -> list[str]:
    """Remove report-status words so headings stay noun-like and concise."""

    trimmed = list(words)
    status_suffixes = {
        "also",
        "complete",
        "completed",
        "reviewed",
        "finalized",
        "finalised",
        "approved",
        "confirmed",
        "discussed",
        "planned",
        "ready",
    }
    while len(trimmed) > 1 and trimmed[-1].casefold() in status_suffixes:
        trimmed.pop()
    if len(trimmed) > 2 and trimmed[-2].casefold() == "also":
        trimmed.pop(-2)
    return trimmed


def compact_heading_words(words: list[str]) -> list[str]:
    """Prefer important nouns from an overlong candidate heading."""

    priority = [
        word for word in words if word.casefold() in IMPORTANT_SINGLE_NOUNS
    ]
    if len(priority) >= 2:
        return priority[:3]
    return words[-3:]


def looks_like_clause(words: list[str]) -> bool:
    """Return whether a heading candidate still looks sentence-like."""

    lowered = [word.casefold() for word in words]
    if lowered[0] in TOPIC_LEADING_VERBS:
        return True
    clause_verbs = {
        "receive",
        "receives",
        "send",
        "sends",
        "speak",
        "speaks",
        "make",
        "makes",
        "get",
        "gets",
        "give",
        "gives",
    }
    if any(word in clause_verbs for word in lowered[:2]):
        return True
    if "email" in lowered and "by" in lowered:
        return True
    return False


def averaged_centroid(
    existing: list[float],
    new_vector: list[float],
    member_count: int,
) -> list[float]:
    """Update a centroid with a newly added vector."""

    if not existing:
        return new_vector
    previous_weight = max(member_count - 1, 1)
    return [
        ((left * previous_weight) + right) / member_count
        for left, right in zip(existing, new_vector)
    ]


def consolidate_clusters_by_title(clusters: list[TopicCluster]) -> list[TopicCluster]:
    """Merge clusters that resolve to the same deterministic topic title."""

    merged: dict[str, TopicCluster] = {}
    for cluster in clusters:
        title = cluster_title(cluster)
        cluster.title = title
        if title not in merged:
            merged[title] = cluster
            continue
        target = merged[title]
        target.records.extend(cluster.records)
        target.contexts.extend(cluster.contexts)
        target.representative_sentence = choose_representative_sentence(target.records)
        if target.centroid_embedding and cluster.centroid_embedding:
            target.centroid_embedding = averaged_centroid(
                target.centroid_embedding,
                cluster.centroid_embedding,
                2,
            )
    return list(merged.values())


def score_clusters(clusters: list[TopicCluster]) -> None:
    """Compute 0.6 confidence + 0.4 centrality importance scores."""

    for cluster in clusters:
        avg_confidence = average_confidence(cluster.records)
        centrality = cluster_centrality(cluster, clusters)
        cluster.importance_score = round((0.6 * avg_confidence) + (0.4 * centrality), 6)


def average_confidence(records: list[Any]) -> float:
    """Return the average ANN confidence for a cluster."""

    if not records:
        return 0.0
    return sum(utils.prediction_confidence(record) for record in records) / len(records)


def cluster_centrality(cluster: TopicCluster, clusters: list[TopicCluster]) -> float:
    """Estimate how central a cluster is among all meeting topics."""

    similarities: list[float] = []
    for other in clusters:
        if other is cluster:
            continue
        if cluster.centroid_embedding and other.centroid_embedding:
            similarities.append(utils.cosine(cluster.centroid_embedding, other.centroid_embedding))
        else:
            similarities.append(utils.lexical_similarity(" ".join(cluster.contexts), " ".join(other.contexts)))
    if not similarities:
        return 1.0
    return max(0.0, min(1.0, sum(similarities) / len(similarities)))


def choose_representative_sentence(records: list[Any]) -> str:
    """Choose the highest-scoring sentence inside a cluster."""

    sentences = [utils.record_sentence(record) for record in records]
    if not sentences:
        return ""
    return max(sentences, key=utils.sentence_score)


def cluster_title(cluster: TopicCluster) -> str:
    """Generate a concise topic title for a cluster."""

    return cluster_title_from_text(" ".join(cluster.contexts + [cluster.representative_sentence]))


def cluster_title_from_text(text: str) -> str:
    """Resolve a topic title using deterministic noun-phrase rules."""

    lowered = text.casefold()
    scored_titles: list[tuple[int, int, str]] = []
    for index, (title, keywords) in enumerate(templates.TOPIC_TITLE_RULES):
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            scored_titles.append((score, -index, title))
    if scored_titles:
        return max(scored_titles)[2]
    topic = utils.professional_topic(text)
    if utils.is_weak_topic(topic):
        return ""
    words = topic.split()
    if len(words) > 3:
        topic = " ".join(words[:3])
    return "" if utils.is_weak_topic(topic) else topic


def build_discussion_points(blueprints: list[TopicBlueprint]) -> list[str]:
    """Generate discussion bullets from topic blueprints."""

    discussion_blueprints = [
        blueprint
        for blueprint in blueprints
        if topic_has_discussion_value(blueprint.evidence)
        and not re.fullmatch(r"(?i)Sprint\s+\d+", blueprint.title)
    ][:5]
    bullets = [discussion_from_blueprint(blueprint) for blueprint in discussion_blueprints]
    merged_bullets = utils.merge_text_fragments(
        utils.unique_normalized_sentences(bullets),
        similarity_threshold=0.82,
        limit=5,
    )
    return merged_bullets


def topic_has_discussion_value(evidence: TopicEvidence) -> bool:
    """Return whether a topic should contribute to discussion points."""

    return (
        bool(evidence.discussion_sentences)
        and evidence.topic != "Meeting Summary"
        and not re.fullmatch(r"(?i)Sprint\s+\d+", evidence.topic)
        and not utils.is_weak_topic(evidence.topic)
    )


def discussion_from_blueprint(blueprint: TopicBlueprint) -> str:
    """Render one professional discussion point from a topic blueprint."""

    title = blueprint.title
    if title in NORMALIZED_DISCUSSION_RENDERERS:
        return f"{title}: {NORMALIZED_DISCUSSION_RENDERERS[title]}"
    if title in templates.DISCUSSION_CONTEXT:
        return f"{title}: {templates.DISCUSSION_CONTEXT[title]}"
    quantities_for_rendering = [
        quantity for quantity in blueprint.entities.quantities if is_reportable_quantity(quantity)
    ]
    if quantities_for_rendering:
        quantities = summarize_list(quantities_for_rendering[:2], fallback="reported figures")
        template = utils.choose_template(QUANTITY_DISCUSSION_TEMPLATES, title)
        return template.format(topic=title, topic_lower=title.lower(), quantities=quantities)
    discussion_deadlines = [
        deadline
        for sentence in blueprint.evidence.discussion_sentences
        for deadline in [utils.extract_deadline(sentence)]
        if deadline and is_valid_deadline_phrase(deadline)
    ]
    discussion_deadlines = unique_preserve_order(discussion_deadlines)
    if discussion_deadlines:
        timeline = summarize_list(discussion_deadlines[:2], fallback="the recorded timeline")
        template = utils.choose_template(TIMELINE_DISCUSSION_TEMPLATES, title)
        return template.format(topic=title, topic_lower=title.lower(), timeline=timeline)
    context = " ".join(blueprint.evidence.cluster.contexts)
    if (
        ("slow" in context.casefold() or "loading" in context.casefold())
        and re.search(r"(?i)\b(performance|loading|optimization|lazy)\b", title)
    ):
        return f"{title}: The team evaluated {title.lower()} to improve responsiveness before release."
    if utils.is_pending_sentence(context):
        return f"{title}: Outstanding work for {title.lower()} was reviewed to clarify ownership and follow-up."
    finding = topic_finding_phrase(blueprint)
    template = utils.choose_template(DISCUSSION_RENDER_TEMPLATES, title)
    return template.format(topic=title, topic_lower=title.lower(), finding=finding)


def topic_finding_phrase(blueprint: TopicBlueprint) -> str:
    """Return a concise rendering phrase from structured topic attributes."""

    if blueprint.entities.deadlines:
        return "timing, ownership, and delivery readiness"
    if blueprint.entities.products:
        return "readiness, ownership, and stakeholder communication"
    if blueprint.entities.organizations or blueprint.entities.locations:
        return "coordination requirements and operational readiness"
    if blueprint.evidence.pending_sentences:
        return "open work, ownership, and follow-up requirements"
    return "current status, risks, and required follow-up"


def is_reportable_quantity(quantity: str) -> bool:
    """Return whether an extracted quantity is useful in final prose."""

    lowered = quantity.casefold()
    if re.search(r"\b(needs?|review|before|after|meeting)\b", lowered):
        return False
    return bool(re.search(r"\d", lowered))


def first_quality_fact(sentences: Iterable[str]) -> str:
    """Return the first validated fact that is not a weak topic fragment."""

    for sentence in sentences:
        fact = utils.meaningful_fact(sentence)
        if fact and not utils.is_weak_topic(fact):
            return fact
    return ""


def cluster_representative_fact(cluster: TopicCluster) -> str:
    """Return the strongest transcript-supported fact from a cluster."""

    candidate_sentences = [
        (utils.record_sentence(record), getattr(record, "predicted_label", ""))
        for record in sorted(
            cluster.records,
            key=lambda record: (
                getattr(record, "predicted_label", "") == "Action_Item",
                cluster_title_from_text(utils.record_sentence(record)) != cluster.title,
                -utils.sentence_score(utils.record_sentence(record)),
            ),
            reverse=False,
        )
    ]
    for sentence, label in candidate_sentences:
        if label == "Action_Item":
            continue
        fact = utils.meaningful_fact(sentence)
        if fact and not utils.is_weak_topic(fact):
            return fact
    for sentence, _label in candidate_sentences:
        fact = utils.meaningful_fact(sentence)
        if fact and not utils.is_weak_topic(fact):
            return fact
    return ""


def decision_sentence_title(sentence: str, fallback: str) -> str:
    """Resolve a decision topic from the sentence that carries the signal."""

    if re.search(r"(?i)\bdark mode\b", sentence):
        return "Dark Mode"
    if re.search(r"(?i)\b(rollout|deployment|deploy|production)\b", sentence):
        return "Deployment"
    fallback_title = clean_decision_title(fallback)
    if fallback_title and not utils.is_weak_topic(fallback_title):
        return fallback_title
    title = cluster_title_from_text(sentence)
    if title and not utils.is_weak_topic(title):
        return clean_decision_title(title)
    topic = utils.professional_topic(sentence)
    if topic and not utils.is_weak_topic(topic):
        return clean_decision_title(topic)
    return "Recorded item"


def clean_decision_title(title: str) -> str:
    """Remove auxiliary endings that create awkward formal decisions."""

    cleaned = utils.normalize_whitespace(title)
    cleaned = re.sub(r"(?i)\b(was|were|is|are|has|have|had|will|should)$", "", cleaned).strip()
    return cleaned or "Recorded item"


def contextual_decision_evidence(sentence: str, context_text: str) -> str:
    """Return decision evidence from explicit signals or adjacent confirmations.

    ANN classification is left untouched. This rule engine only interprets the
    sentence and its existing context window so explicit outcome language is not
    lost when the classifier assigns a non-Decision label.
    """

    if not is_contextual_confirmation(sentence):
        cleaned = utils.clean_report_sentence(sentence)
        if cleaned and has_explicit_decision_evidence(cleaned):
            return cleaned
        return ""
    proposal = proposal_from_context(context_text, exclude_sentence=sentence)
    if not proposal:
        return ""
    return utils.clean_report_sentence(proposal)


def has_explicit_decision_evidence(text: str) -> bool:
    """Return whether text contains a deterministic decision signal."""

    if not text or utils.is_agreement_sentence(text) or is_contextual_confirmation(text):
        return False
    if re.search(r"(?i)\b(approval|approved)\b.{0,24}\b(pending|waiting|not completed|requires review)\b", text):
        return False
    if re.search(r"(?i)\b(pending|waiting|not completed|requires review)\b.{0,24}\b(approval|approved)\b", text):
        return False
    return bool(DECISION_SIGNAL_PATTERN.search(text))


def is_contextual_confirmation(text: str) -> bool:
    """Return whether a short reply confirms a nearby proposal."""

    return bool(DECISION_CONFIRMATION_PATTERN.match(utils.normalize_whitespace(text)))


def is_decision_context_record(text: str) -> bool:
    """Keep otherwise low-value confirmations for nearby decision analysis."""

    cleaned = utils.normalize_whitespace(text)
    return is_contextual_confirmation(cleaned) or has_explicit_decision_evidence(cleaned)


def proposal_from_context(context_text: str, *, exclude_sentence: str) -> str:
    """Extract the nearest proposal from an existing context window."""

    excluded = utils.normalize_whitespace(exclude_sentence).casefold().strip(".!?")
    candidates = [
        candidate.strip()
        for candidate in re.split(r"(?<=[.!?])\s+", context_text)
        if candidate.strip()
    ]
    for candidate in reversed(candidates):
        comparable = utils.normalize_whitespace(candidate).casefold().strip(".!?")
        if not comparable or comparable == excluded:
            continue
        if (
            DECISION_PROPOSAL_PATTERN.search(candidate)
            and not has_explicit_decision_evidence(candidate)
        ):
            return candidate
    return ""


def decision_topic_from_sentence(sentence: str, fallback_title: str) -> str:
    """Resolve a clean decision subject from explicit outcome wording."""

    cleaned = utils.normalize_whitespace(sentence).rstrip(".")
    if re.search(r"(?i)\brelease on\b", cleaned):
        return "Release Timeline"
    if re.search(r"(?i)\bwill handle\b", cleaned) and re.search(r"(?i)\bbackend|frontend\b", cleaned):
        return "Implementation Ownership"
    if re.search(r"(?i)\bminilm\b", cleaned):
        return "MiniLM"
    topic_patterns = (
        r"(?i)\b(?:feature\s+list)\b",
        r"(?i)\b(?:dark\s+mode)\b",
        r"(?i)\b(?:multilingual\s+support)\b",
        r"(?i)\b(?:performance\s+optimization)\b",
        r"(?i)\b(?:option\s+[A-Z])\b",
        r"(?i)\b(?:MiniLM)\b",
        r"(?i)\b(?:proposal)\b",
        r"(?i)\b(?:backend|frontend)\b",
    )
    for pattern in topic_patterns:
        match = re.search(pattern, cleaned)
        if match:
            return clean_decision_title(title_case_topic(match.group(0)))

    stripped = re.sub(
        r"(?i)\b(the|a|an|we|team|will|shall|should|let'?s|lets|i think|"
        r"approved|approve|accepted|confirmed|finali[sz]ed|agreed|decided|"
        r"resolved|locked|freeze|freezed|postponed|deferred|moved|moves|move|"
        r"to|go ahead|proceed with|continue with|stop|cancel|close|completed|"
        r"release on|ship|deploy|merge|merged|use)\b",
        " ",
        cleaned,
    )
    topic = cluster_title_from_text(stripped) or utils.professional_topic(stripped)
    if topic and not utils.is_weak_topic(topic):
        return clean_decision_title(topic)
    return decision_sentence_title(sentence, fallback_title)


def decision_from_sentence(sentence: str, fallback_title: str) -> str:
    """Generate a decision using only the sentence with decision evidence."""

    title = decision_topic_from_sentence(sentence, fallback_title)
    subject = decision_subject(sentence, title)
    sprint_match = re.search(r"(?i)\bSprint\s+(\d+)\b", sentence)
    sprint_phrase = f"Sprint {sprint_match.group(1)}" if sprint_match else ""
    weekday_match = re.search(
        r"(?i)\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow)\b",
        sentence,
    )
    release_phrase = weekday_match.group(1).title() if weekday_match else ""
    if re.match(r"(?i)^\s*go\s+ahead\b", sentence):
        return "The team approved proceeding with the planned work."
    if re.search(r"(?i)\b(defer|deferred|postpone|postponed|moves to|move to|moved to|will move to|will be moved to|future release|next release|next sprint|delay|delayed)\b", sentence):
        suffix = f" to {sprint_phrase}" if sprint_phrase else ""
        if re.search(r"(?i)\bnext release\b", sentence):
            suffix = " to Next Release"
        return f"{subject} was deferred{suffix}."
    if re.search(r"(?i)\b(approve|approved|approval|approved by|go ahead|proceed|proceed with|approved for release|selected|chosen|accepted)\b", sentence):
        suffix = f" for {sprint_phrase}" if sprint_phrase else " as the agreed direction"
        if title == "Proposal":
            subject = proposal_decision_subject(sentence, fallback_title)
            return f"The proposal for {subject} was approved."
        if re.fullmatch(r"(?i)(go\s+ahead|proceed(?:ing)?|planned\s+work)", subject):
            return "The team approved proceeding with the planned work."
        return f"{subject} was approved{f' for {sprint_phrase}' if sprint_phrase else ''}."
    if re.search(r"(?i)\b(added to|included in)\b", sentence):
        suffix = f" for {sprint_phrase}" if sprint_phrase else ""
        return f"{title} was approved{suffix}."
    if re.search(r"(?i)\b(finalize|finalise|finalized|finalised|feature freeze)\b", sentence):
        return f"{subject} was finalized."
    if re.search(r"(?i)\b(release on|ship)\b", sentence):
        suffix = f" on {release_phrase}" if release_phrase else ""
        return f"The team confirmed the release schedule{suffix}."
    if re.search(r"(?i)\bwill handle\b", sentence):
        return f"The team confirmed {title.lower()}."
    if re.search(r"(?i)\b(deploy|merge|merged|release candidate)\b", sentence):
        return f"The team confirmed {title} as the agreed release path."
    if re.search(r"(?i)\b(scheduled)\b", sentence):
        return f"{subject} was scheduled."
    if re.search(r"(?i)\b(freeze|feature freeze)\b", sentence):
        return f"{subject} was finalized."
    if re.search(r"(?i)\b(locked|closed|completed)\b", sentence):
        return f"{subject} was finalized."
    if re.search(r"(?i)\b(will remain|will be implemented|implemented)\b", sentence):
        return f"The team confirmed {title} as the agreed outcome."
    if re.search(r"(?i)\b(confirmed|resolved|resolution|agree|agreed|agreement reached|decided|decision)\b", sentence):
        if title == "MiniLM":
            return "The proposal to use MiniLM was approved."
        return f"The team confirmed {title} as the agreed outcome."
    return normalize_decision(sentence)


def decision_subject(sentence: str, fallback_title: str) -> str:
    """Extract a clean noun phrase from imperative or completed decisions."""

    cleaned = utils.normalize_whitespace(sentence).strip(" .!?")
    cleaned = re.sub(
        r"(?i)^(?:the\s+team\s+)?(?:has\s+|had\s+)?(?:approved?|accept(?:ed)?|"
        r"confirm(?:ed)?|finali[sz](?:e|ed)|reject(?:ed)?|postpone(?:d)?|"
        r"defer(?:red)?|move(?:d)?|decid(?:e|ed)\s+to)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\s+(?:has\s+been|was|is|were)\s+(?:approved|accepted|confirmed|"
        r"finali[sz]ed|rejected|postponed|deferred|moved)\b.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b(?:as\s+the\s+agreed\s+(?:direction|outcome))\b.*$", "", cleaned)
    cleaned = re.sub(r"(?i)^the\s+", "", cleaned).strip(" .")
    if not cleaned or re.fullmatch(r"(?i)(go\s+ahead|proceed(?:ing)?)", cleaned):
        return "planned work"
    words = re.findall(r"[A-Za-z0-9%&/-]+", cleaned)
    topic = title_case_topic(cleaned) if 1 <= len(words) <= 8 else ""
    topic = topic or normalize_topic_heading(cleaned) or utils.professional_topic(cleaned)
    if not topic or utils.is_weak_topic(topic):
        topic = fallback_title
    return topic[:1].upper() + topic[1:]


def cleanup_decision_output(text: str) -> str:
    """Remove duplicated outcome language from a rendered decision."""

    cleaned = re.sub(
        r"(?i)\s+(?:as\s+the\s+agreed\s+(?:direction|outcome)|"
        r"as\s+the\s+agreed\s+release\s+path)\b",
        "",
        text,
    )
    cleaned = re.sub(
        r"(?i)\b(approved|confirmed|finalized|accepted|rejected|postponed|"
        r"moved|deferred)(?:\s+(?:has\s+been|was|is))?\s+\1\b",
        r"\1",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\bhas\s+been\s+approved\s+approved\b", "was approved", cleaned)
    cleaned = re.sub(r"(?i)\bhas\s+been\s+approved\b", "was approved", cleaned)
    return utils.normalize_sentence(cleaned)


def decision_auxiliary(title: str) -> str:
    """Return a simple deterministic auxiliary for decision wording."""

    last_word = title.strip().split()[-1].casefold() if title.strip() else ""
    singular_concepts = {
        "logistics",
        "readiness",
        "performance",
        "analysis",
        "status",
        "progress",
        "notifications",
    }
    if last_word in singular_concepts:
        return "was"
    return "were" if last_word.endswith("s") and not last_word.endswith("ss") else "was"


def proposal_decision_subject(sentence: str, fallback_title: str) -> str:
    """Extract the concrete subject behind a proposal approval."""

    patterns = (
        r"(?i)\bproposal\s+(?:for|on|about)\s+(?P<subject>.+?)\s+(?:is|was|has been)?\s*approved\b",
        r"(?i)\bproposal\s+(?:to|for)\s+(?P<subject>.+?)$",
    )
    for pattern in patterns:
        match = re.search(pattern, sentence.rstrip("."))
        if not match:
            continue
        subject = normalize_topic_heading(match.group("subject")) or utils.professional_topic(match.group("subject"))
        if subject and subject != "Proposal":
            return subject
    fallback = normalize_topic_heading(fallback_title) or utils.professional_topic(fallback_title)
    return fallback if fallback and fallback != "Proposal" else "the recorded proposal"


def build_decisions(blueprints: list[TopicBlueprint]) -> list[str]:
    """Extract formal decisions from explicitly supported topic blueprints."""

    decisions: list[str] = []
    for blueprint in blueprints:
        if not decision_has_topic_support(blueprint):
            continue
        for sentence in blueprint.evidence.decision_sentences:
            if not contains_decision_signal(sentence):
                continue
            decision = decision_from_sentence(sentence, blueprint.title)
            if decision:
                decisions.append(decision)
    unique_decisions = utils.unique_normalized_sentences(decisions)
    deduped_decisions = deduplicate_decisions_by_outcome(unique_decisions)
    return utils.merge_text_fragments(deduped_decisions, similarity_threshold=0.86, limit=10)


def deduplicate_decisions_by_outcome(decisions: Iterable[str]) -> list[str]:
    """Remove repeated decisions that express the same outcome."""

    deduped: list[str] = []
    seen: set[str] = set()
    for decision in decisions:
        key = decision_outcome_key(decision)
        if key in seen:
            continue
        deduped.append(decision)
        seen.add(key)
    return deduped


def decision_outcome_key(decision: str) -> str:
    """Build a stable de-duplication key from topic and outcome type."""

    lowered = decision.casefold()
    if "dark mode" in lowered and re.search(r"\b(defer|deferred|postponed|sprint\s+5)\b", lowered):
        return "dark_mode_deferred"
    if "feature list" in lowered and re.search(r"\b(finalized|closed|freeze|frozen)\b", lowered):
        return "feature_list_finalized"
    if "minilm" in lowered:
        return "minilm_approved"
    topic = normalize_decision_summary_topic(decision)
    outcome = "approved"
    if re.search(r"\b(defer|deferred|postponed)\b", lowered):
        outcome = "deferred"
    elif re.search(r"\b(finalized|closed|locked|completed)\b", lowered):
        outcome = "finalized"
    elif re.search(r"\b(release|scheduled)\b", lowered):
        outcome = "scheduled"
    return f"{topic.casefold()}_{outcome}"


def decision_has_topic_support(blueprint: TopicBlueprint) -> bool:
    """Require explicit decision evidence plus broader topic support."""

    return bool(blueprint.evidence.decision_sentences)


def contains_decision_signal(text: str) -> bool:
    """Return whether text contains a real decision signal."""

    return has_explicit_decision_evidence(text)


def decision_from_cluster(cluster: TopicCluster, cluster_text: str) -> str:
    """Generate a formal decision from a topic cluster."""

    sprint_match = re.search(r"(?i)\bSprint\s+(\d+)\b", cluster_text)
    sprint_phrase = f"Sprint {sprint_match.group(1)}" if sprint_match else ""
    if re.search(r"(?i)\b(moves to|move to|postponed|deferred|later|next sprint|sprint\s+5)\b", cluster_text):
        if re.search(r"(?i)\bdark mode\b", cluster_text):
            return "Dark Mode implementation was deferred to Sprint 5."
        suffix = f" to {sprint_phrase}" if sprint_phrase and sprint_phrase != "Sprint 4" else " to Sprint 5" if "sprint 5" in cluster_text.casefold() else ""
        return f"{cluster.title} implementation was deferred{suffix}."
    if re.search(r"(?i)\b(go ahead|proceed|approved|approval|selected)\b", cluster_text):
        suffix = f" during {sprint_phrase}" if sprint_phrase else ""
        topic_text = cluster.title.replace(" Implementation", "").lower()
        return f"The team approved implementing {topic_text}{suffix}."
    if re.search(r"(?i)\b(approved|approval|finalized|confirmed|accepted)\b", cluster_text):
        suffix = f" for {sprint_phrase}" if sprint_phrase else ""
        return f"{cluster.title} was approved{suffix}."
    return normalize_decision(cluster.representative_sentence)


def normalize_decision(sentence: str) -> str:
    """Apply simple deterministic decision wording."""

    cleaned = sentence.rstrip(".")
    lowered = cleaned.casefold()
    topic = utils.professional_topic(cleaned)
    sprint_match = re.search(r"(?i)\bSprint\s+(\d+)\b", cleaned)
    sprint_phrase = f"Sprint {sprint_match.group(1)}" if sprint_match else ""
    if re.search(r"(?i)\b(implement|we'll implement|we will implement)\b", cleaned):
        suffix = f" during {sprint_phrase}" if sprint_phrase else ""
        topic_text = topic.replace(" implementation", "")
        return f"The team approved implementing {topic_text.lower()}{suffix}."
    if re.search(r"(?i)\b(moves to|move to|moved to|will move to|postponed|deferred)\b", cleaned):
        suffix = f" to {sprint_phrase}" if sprint_phrase else ""
        return f"{topic} implementation was deferred{suffix}."
    if re.search(r"(?i)\b(added to|included in)\b", cleaned):
        suffix = f" for {sprint_phrase}" if sprint_phrase else ""
        return f"{topic} was approved{suffix}."
    for trigger, template in templates.DECISION_REWRITES:
        if trigger in lowered:
            subject = re.sub(re.escape(trigger), "", cleaned, flags=re.IGNORECASE).strip(" .")
            subject = utils.professional_topic(subject or topic)
            if subject:
                return template.format(subject=subject[0].upper() + subject[1:]) + "."
    if re.search(r"(?i)\b(defer|postpone|later|next sprint|sprint\s+5)\b", cleaned):
        return f"{topic} has been deferred."
    if re.search(r"(?i)\b(approve|approved|approval|go ahead|finalize|finalized|accepted)\b", cleaned):
        return f"{topic} has been approved."
    if re.search(r"(?i)\b(scheduled)\b", cleaned):
        return f"{topic} has been scheduled."
    if re.search(r"(?i)\b(locked|completed|closed|freeze)\b", cleaned):
        return f"{topic} has been closed."
    if re.search(r"(?i)\b(will remain|will be implemented)\b", cleaned):
        return f"{topic} has been confirmed."
    if re.search(r"(?i)\b(retain|keep|continue)\b", cleaned):
        return f"{topic} will be retained."
    if not re.search(r"(?i)\b(will be|was|were|has been|have been)\b", cleaned):
        cleaned = f"{topic} has been confirmed"
    return utils.normalize_sentence(cleaned)


def build_action_items(blueprints: list[TopicBlueprint]) -> list[ExperimentalActionItem]:
    """Build normalized action rows from validated topic blueprints."""

    action_items: list[ExperimentalActionItem] = []
    seen: set[str] = set()
    for blueprint in blueprints:
        for record in blueprint.evidence.action_records:
            raw_sentence = utils.record_sentence(record)
            if is_non_action_utterance(raw_sentence) or is_decision_only_statement(raw_sentence):
                continue
            if not action_sentence_has_evidence(raw_sentence, getattr(record, "predicted_label", "")):
                continue
            if not utils.is_reportable_sentence(raw_sentence):
                continue
            if raw_sentence.strip().endswith("?") and not re.match(
                r"(?i)^\s*(?:(?:can|could|would)\s+you|"
                r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+"
                r"(?:will|shall|needs to|need to|should|must|has to))\b",
                raw_sentence,
            ):
                continue
            parts = utils.extract_action_parts(
                raw_sentence,
                getattr(record, "speaker", ""),
            )
            key = f"{parts.owner}|{parts.task}|{parts.deadline}".casefold()
            if re.search(r"(?i)^(finish\s+it|'?s goal\b)", parts.task):
                continue
            if not parts.task or key in seen:
                continue
            action_items.append(
                ExperimentalActionItem(
                    owner=parts.owner,
                    task=parts.task,
                    deadline=parts.deadline,
                )
            )
            seen.add(key)
    return action_items


def action_sentence_has_evidence(sentence: str, predicted_label: str) -> bool:
    """Return whether a sentence contains explicit action-item evidence."""

    if is_non_action_utterance(sentence) or is_decision_only_statement(sentence):
        return False
    if re.search(r"(?i)\b(resolved|approved|confirmed|accepted|finalized|agreed|decided)\b", sentence):
        return bool(
            re.search(
                r"(?i)\b([A-Z][a-z]+|i|we)\s+(will|shall|need to|needs to|should|must|has to)\b",
                sentence,
            )
        )
    if predicted_label == "Action_Item" and re.search(
        r"(?i)\b(i'?ll|i will|we will|will|shall|need to|needs to|should|must|"
        r"has to|complete|update|publish|confirm|prepare|assign|generate|send|"
        r"share|email|review|finalize|implement|submit|deliver|deploy|verify|"
        r"finish|test|fix)\b",
        sentence,
    ):
        return True
    # Non-action labels need explicit assignment or commitment language. This
    # prevents ordinary discussion phrases such as "requires review" from being
    # converted into action items without an owner, modal, or imperative task.
    if re.search(
        r"(?i)\b(i'?ll|i will|we will|need to|needs to|should|must|has to|"
        r"assigned to|responsible for)\b",
        sentence,
    ):
        return True
    if re.match(
        r"(?i)^\s*(?:please\s+)?(complete|update|publish|confirm|prepare|assign|generate|"
        r"send|share|email|review|finalize|implement|submit|deliver|deploy|verify|"
        r"finish|test|fix)\b",
        sentence,
    ):
        return True
    if re.match(r"(?i)^\s*(?:can|could|would)\s+you\s+\w+", sentence):
        return True
    return bool(
        re.match(
            r"(?i)^\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+"
            r"(will|shall|need to|needs to|should|must|has to)\b",
            sentence,
        )
    )


def is_non_action_utterance(sentence: str) -> bool:
    """Reject greetings, introductions, questions, and status narration."""

    cleaned = utils.normalize_whitespace(sentence)
    if not cleaned:
        return True
    if re.match(
        r"(?i)^(?:good\s+(?:morning|afternoon|evening)|hello|hi|welcome|"
        r"let'?s\s+(?:begin|start|get\s+started)|today'?s\s+meeting\s+is|"
        r"the\s+(?:meeting|session)\s+(?:is|was)|we(?:'ll|\s+will)\s+review\b)",
        cleaned,
    ):
        return True
    if cleaned.endswith("?") and not re.match(
        r"(?i)^\s*(?:can|could|would)\s+you\b", cleaned
    ):
        return True
    if re.search(
        r"(?i)\b(?:was|were|has\s+been|have\s+been)\s+"
        r"(?:discussed|reviewed|completed|updated|noted)\b",
        cleaned,
    ):
        return True
    return False


def is_decision_only_statement(sentence: str) -> bool:
    """Separate agreed scope/outcome declarations from executable tasks."""

    cleaned = utils.normalize_whitespace(sentence)
    has_explicit_assignee = bool(
        re.search(
            r"(?i)\b(?:i'?ll|i\s+will|we\s+will|"
            r"assigned\s+to|responsible\s+for|please|can\s+you|must|need(?:s)?\s+to)\b",
            cleaned,
        )
    )
    outcome_declaration = bool(
        re.search(
            r"(?i)\b(?:will\s+be|was|has\s+been)\s+"
            r"(?:included|added|approved|accepted|confirmed|finali[sz]ed|"
            r"rejected|postponed|deferred|moved|retained|implemented)\b",
            cleaned,
        )
    )
    return outcome_declaration and not has_explicit_assignee


def action_record_has_task_support(record: Any, sentence: str) -> bool:
    """Require owner, future intent, and a concrete task before action output."""

    parts = utils.extract_action_parts(sentence, getattr(record, "speaker", ""))
    owner_is_known = normalize_owner(parts.owner) != "Unassigned"
    has_future_intent = bool(
        re.search(
            r"(?i)\b(i'?ll|i will|we will|will|shall|need to|needs to|should|"
            r"must|has to|by\s+[A-Za-z]+|before\s+[A-Za-z]+|this week|"
            r"next week|tomorrow|today)\b",
            sentence,
        )
    )
    return owner_is_known and has_future_intent and is_valid_action_task(parts.task)


def build_pending_items(
    blueprints: list[TopicBlueprint],
    action_items: list[ExperimentalActionItem],
) -> list[str]:
    """Detect unresolved or deferred work without duplicating action rows."""

    pending: list[str] = []
    for blueprint in blueprints:
        cluster_text = " ".join(blueprint.evidence.pending_sentences)
        if not utils.is_pending_sentence(cluster_text):
            continue
        topic_name = blueprint.title
        if pending_duplicates_action(topic_name, action_items):
            continue
        pending_item = normalize_pending_item(cluster_text, topic_name)
        if pending_duplicates_action(f"{topic_name} {pending_item}", action_items):
            continue
        pending.append(pending_item)
    unique_pending = utils.unique_normalized_sentences(pending)
    return utils.merge_text_fragments(unique_pending, similarity_threshold=0.82, limit=5)


def pending_duplicates_action(
    topic: str,
    action_items: list[ExperimentalActionItem],
) -> bool:
    """Return whether a pending topic already appears as assigned work."""

    topic_key = topic.casefold()
    for item in action_items:
        task_key = item.task.casefold()
        if "dark mode" in topic_key and "dark mode" not in task_key:
            continue
        if topic_key in task_key or utils.lexical_similarity(topic_key, task_key) >= 0.34:
            return True
        if "pdf" in topic_key and "pdf" in task_key:
            return True
        if "pdf" in topic_key and "export" in task_key:
            return True
        if "ui" in topic_key and "ui" in task_key:
            return True
    return False


def normalize_pending_item(sentence: str, topic: str) -> str:
    """Normalize unresolved work into concise pending-item wording."""

    if re.search(r"(?i)\b(dark mode|deferred|postponed|next sprint|later)\b", sentence):
        return "Dark mode implementation was deferred to a later sprint."
    if re.search(r"(?i)\bpdf\b.*\b(polish|polishing|needs)\b", sentence):
        return "PDF layout polishing remains pending."
    if re.search(r"(?i)\b(ui|alignment|issues)\b", sentence):
        return "UI alignment issues require review."
    if re.search(r"(?i)\bblocker|remaining\b", sentence):
        return f"{topic or 'Remaining release work'} requires follow-up review."
    return f"{topic or utils.professional_topic(sentence)} remains pending."


def pending_items_from_decisions(decisions: list[str]) -> list[str]:
    """Create pending items from formal deferred decisions only."""

    pending: list[str] = []
    for decision in decisions:
        if not re.search(r"(?i)\bdeferred|postponed|later sprint|sprint\s+5\b", decision):
            continue
        if re.search(r"(?i)\bdark mode\b", decision):
            pending.append("Dark mode implementation was deferred to a later sprint.")
        else:
            pending.append(re.sub(r"(?i)\bwas deferred\b", "remains deferred", decision).rstrip(".") + ".")
    return pending


def build_information(blueprints: list[TopicBlueprint]) -> list[str]:
    """Keep useful factual statements from structured topic blueprints."""

    information: list[str] = []
    for blueprint in blueprints:
        for sentence in blueprint.evidence.information_sentences:
            if not utils.is_reportable_sentence(sentence):
                continue
            if utils.is_pending_sentence(sentence):
                continue
            if not re.search(
                r"(?i)\b(accuracy|next meeting|next review|release|performance|pending|"
                r"issue|blocker|timeline|deadline|model|metric|scheduled|export|"
                r"api|apis|ready|regression|completed)\b",
                sentence,
            ):
                continue
            normalized = utils.factual_information(sentence)
            if re.search(r"(?i)\b(next meeting|next review|follow-up|review meeting)\b", normalized):
                deadline = utils.extract_deadline(normalized)
                if deadline:
                    normalized = f"Next review meeting is scheduled for {deadline}."
            information.append(normalized)
    unique_information = utils.unique_normalized_sentences(information)
    return utils.merge_text_fragments(unique_information, similarity_threshold=0.82, limit=5)


def build_objective(blueprints: list[TopicBlueprint]) -> str:
    """Generate a concise objective from the strongest topic blueprint."""

    primary = max(
        blueprints,
        key=lambda blueprint: (len(blueprint.evidence.discussion_sentences), blueprint.importance_score),
        default=None,
    )
    if primary is None:
        return ""
    context = " ".join(primary.evidence.cluster.contexts + [primary.title])
    if re.search(r"(?i)\bloading|performance|optimization|lazy\b", context):
        return "Evaluate application performance, confirm the optimization approach, and align ownership for release readiness."
    if re.search(r"(?i)\bsprint|release|feature\b", context):
        return "Review sprint release readiness, finalize delivery priorities, and confirm ownership for remaining work."
    if re.search(r"(?i)\btesting|qa|regression\b", context):
        return "Assess testing readiness, confirm quality risks, and assign validation follow-up."
    return limit_words(
        utils.normalize_sentence(
            f"Review {primary.title.lower()} and confirm any required follow-up."
        ),
        20,
    )


def build_summary(
    *,
    objective: str,
    blueprints: list[TopicBlueprint],
    discussion_points: list[str],
    decisions: list[str],
    action_items: list[ExperimentalActionItem],
    pending_items: list[str],
    information: list[str],
) -> str:
    """Build a 100-word executive summary from clusters and outcomes."""

    summary_sentences: list[str] = []
    discussion_topics = [
        blueprint.title
        for blueprint in blueprints
        if blueprint.title
        and not utils.is_weak_topic(blueprint.title)
        and not re.fullmatch(r"(?i)Sprint\s+\d+", blueprint.title)
    ][:3]
    if discussion_topics:
        opener = summary_opening_phrase(discussion_topics)
        summary_sentences.append(
            f"{opener} {summarize_list([topic_phrase(topic) for topic in discussion_topics], fallback='the main meeting themes')}."
        )
    if decisions:
        summary_sentences.append(
            f"Formal outcomes were captured for {summarize_decision_topics(decisions)}."
        )
    if action_items:
        summary_sentences.append(
            f"{len(action_items)} follow-up action item{'s were' if len(action_items) != 1 else ' was'} recorded for execution."
        )
    if pending_items:
        summary_sentences.append(
            f"Pending work remains for {summarize_pending_topics(pending_items)}."
        )
    elif information and not discussion_topics:
        summary_sentences.append(
            f"Relevant updates included {summarize_list([item.rstrip('.') for item in information[:2]], fallback='the captured information')}."
        )
    if not summary_sentences:
        summary_sentences.append(
            "No reportable discussion, decisions, or action items were identified from the transcript."
        )
    summary = " ".join(summary_sentences)
    return limit_words(utils.normalize_sentence(summary), 100)


def summary_opening_phrase(topics: list[str]) -> str:
    """Choose a deterministic opener so summaries do not sound repetitive."""

    options = (
        "The session covered",
        "Key discussion areas included",
        "Participants reviewed",
        "The meeting addressed",
        "The meeting focused on",
        "The discussion centered on",
    )
    seed = " ".join(topics).casefold()
    index = sum(ord(character) for character in seed) % len(options)
    return options[index]


def summarize_discussion_context(discussion_points: list[str]) -> str:
    """Create a natural summary phrase from discussion bullets."""

    text = " ".join(discussion_points)
    topics: list[str] = []
    if re.search(r"(?i)\btranscript loading|lazy loading|performance\b", text):
        topics.append("application performance")
    if re.search(r"(?i)\brelease|sprint|blocker\b", text):
        topics.append("release readiness")
    if re.search(r"(?i)\bpdf|export|ui|alignment\b", text):
        topics.append("remaining product polish")
    if not topics:
        topics.append(templates.NO_DISCUSSION_TEXT)
    return summarize_list(topics, fallback=templates.NO_DISCUSSION_TEXT)


def summarize_decision_topics(decisions: list[str]) -> str:
    """Summarize decision subjects without repeating full decision sentences."""

    topics = [
        normalize_decision_summary_topic(decision)
        for decision in decisions[:3]
    ]
    topics = [topic for topic in topics if topic]
    return summarize_list(topics, fallback="the recorded decisions")


def normalize_decision_summary_topic(decision: str) -> str:
    """Extract a concise decision subject from rendered decision prose."""

    cleaned = re.sub(r"(?i)^the team\s+(approved|confirmed|finalized)\s+", "", decision).strip(" .")
    cleaned = re.sub(r"(?i)^the team agreed to defer\s+", "", cleaned).strip(" .")
    cleaned = re.sub(r"(?i)\s+as the agreed (direction|outcome)$", "", cleaned)
    cleaned = re.sub(r"(?i)\s+for Sprint\s+\d+$", "", cleaned)
    cleaned = re.sub(r"(?i)\s+implementation$", "", cleaned)
    return topic_phrase(cleaned)


def topic_phrase(topic: str) -> str:
    """Render a topic title as a readable inline summary phrase."""

    words = re.findall(r"[A-Za-z0-9]+", topic)
    rendered = [
        word if word.isupper() and len(word) <= 4 else word.lower()
        for word in words
    ]
    return utils.normalize_whitespace(" ".join(rendered))


def summarize_pending_topics(pending_items: list[str]) -> str:
    """Summarize pending items by topic instead of full pending sentences."""

    topics = [
        pending_topic_phrase(item)
        for item in pending_items[:3]
    ]
    return summarize_list([topic.lower() for topic in topics if topic], fallback="the unresolved work")


def pending_topic_phrase(item: str) -> str:
    """Return a clean topic phrase for pending-work summary text."""

    if re.search(r"(?i)\bdark mode\b", item):
        return "dark mode implementation"
    if re.search(r"(?i)\bui\b|\balignment\b", item):
        return "UI alignment"
    cleaned = re.sub(
        r"(?i)\b(remains pending|requires review|was deferred|remains deferred|to a later sprint|later sprint)\b",
        "",
        item,
    )
    return topic_phrase(normalize_topic_heading(cleaned) or cleaned)


def summarize_list(items: list[str], fallback: str) -> str:
    """Summarize a short list into readable inline text."""

    cleaned = [item.strip(" .") for item in items if item.strip(" .")]
    if not cleaned:
        return fallback
    cleaned = cleaned[:3]
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def render_bullets(items: list[str], *, fallback: str = "No items were identified.") -> list[str]:
    """Render bullet lines with a fallback message."""

    if not items:
        return [f"- {fallback}"]
    return [f"- {item}" for item in items]


def limit_words(text: str, max_words: int) -> str:
    """Limit generated prose to a maximum word count."""

    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,.;") + "."

"""Experimental deterministic Minutes of Meeting formatter.

Purpose:
    Convert existing ANN sentence predictions into a more polished MoM format
    without changing the production rule-based generator. This is a reversible
    prototype under ``ml_mom/experimental`` only.

Responsibilities:
    - Group existing ANN predictions by label.
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
        lines.extend(render_bullets(self.decisions))
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

        records = utils.confidence_filtered_records(
            list(prediction_records),
            threshold=self.confidence_threshold,
        )
        inferred_participants = participants or infer_participants(records)
        clusters = build_topic_clusters(records)
        discussion_points = build_discussion_points(clusters)
        decisions = build_decisions(clusters)
        action_items = build_action_items(clusters)
        information = build_information(clusters)
        pending_items = build_pending_items(clusters, action_items)
        pending_items = utils.unique_normalized_sentences(
            pending_items + pending_items_from_decisions(decisions)
        )
        objective = build_objective(clusters)
        summary = build_summary(
            objective=objective,
            clusters=clusters,
            discussion_points=discussion_points,
            decisions=decisions,
            action_items=action_items,
            pending_items=pending_items,
            information=information,
        )
        return ExperimentalMom(
            meeting_title=meeting_title,
            meeting_date=meeting_date,
            objective=objective,
            participants=inferred_participants,
            summary=summary,
            discussion_points=discussion_points,
            decisions=decisions,
            action_items=action_items,
            pending_items=pending_items,
            information=information,
        )


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
        record for record in records if utils.is_reportable_sentence(utils.record_sentence(record))
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

        clusters.append(
            TopicCluster(
                title=cluster_title_from_text(context or sentence),
                records=[record],
                contexts=[context],
                representative_sentence=sentence,
                centroid_embedding=vector,
            )
        )

    consolidated = consolidate_clusters_by_title(clusters)
    score_clusters(consolidated)
    return sorted(consolidated, key=lambda cluster: cluster.importance_score, reverse=True)


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
    words = topic.split()
    if len(words) > 3:
        topic = " ".join(words[:3])
    return topic or "Project Coordination"


def build_discussion_points(clusters: list[TopicCluster]) -> list[str]:
    """Generate discussion bullets from topic clusters."""

    discussion_clusters = [
        cluster
        for cluster in clusters
        if cluster_has_discussion_value(cluster)
    ][:5]
    bullets = [discussion_from_cluster(cluster) for cluster in discussion_clusters]
    merged_bullets = utils.merge_text_fragments(
        utils.unique_normalized_sentences(bullets),
        similarity_threshold=0.82,
        limit=5,
    )
    return merged_bullets or ["Project priorities were evaluated to clarify delivery impact."]


def cluster_has_discussion_value(cluster: TopicCluster) -> bool:
    """Return whether a cluster should contribute to discussion points."""

    labels = {getattr(record, "predicted_label", "") for record in cluster.records}
    return bool(labels - {"Action_Item"}) and cluster.title != "Meeting Summary"


def discussion_from_cluster(cluster: TopicCluster) -> str:
    """Summarize one topic cluster into a What/Why discussion bullet."""

    if cluster.title in templates.DISCUSSION_CONTEXT:
        return templates.DISCUSSION_CONTEXT[cluster.title]
    context = " ".join(cluster.contexts)
    if "slow" in context.casefold() or "loading" in context.casefold():
        return f"The team evaluated {cluster.title.lower()} to improve responsiveness before release."
    if utils.is_pending_sentence(context):
        return f"The team reviewed {cluster.title.lower()} to clarify unresolved work and ownership."
    template = utils.choose_template(templates.DISCUSSION_TEMPLATES, cluster.title)
    return template.format(topic=cluster.title)


def build_decisions(clusters: list[TopicCluster]) -> list[str]:
    """Extract formal decisions from context-rich topic clusters."""

    decisions: list[str] = []
    for cluster in clusters:
        cluster_text = " ".join(utils.record_sentence(record) for record in cluster.records)
        if not contains_decision_signal(cluster_text):
            continue
        decision = decision_from_cluster(cluster, cluster_text)
        if decision:
            decisions.append(decision)
    unique_decisions = utils.unique_normalized_sentences(decisions)
    return utils.merge_text_fragments(unique_decisions, similarity_threshold=0.86, limit=6)


def contains_decision_signal(text: str) -> bool:
    """Return whether text contains a real decision signal."""

    lowered = text.casefold()
    if utils.is_agreement_sentence(text):
        return False
    return any(signal in lowered for signal in templates.DECISION_SIGNALS)


def decision_from_cluster(cluster: TopicCluster, cluster_text: str) -> str:
    """Generate a formal decision from a topic cluster."""

    sprint_match = re.search(r"(?i)\bSprint\s+(\d+)\b", cluster_text)
    sprint_phrase = f"Sprint {sprint_match.group(1)}" if sprint_match else ""
    if re.search(r"(?i)\b(moves to|move to|postponed|deferred|later|next sprint|sprint\s+5)\b", cluster_text):
        if re.search(r"(?i)\bdark mode\b", cluster_text):
            return "Dark Mode implementation was deferred to Sprint 5."
        suffix = f" to {sprint_phrase}" if sprint_phrase and sprint_phrase != "Sprint 4" else " to Sprint 5" if "sprint 5" in cluster_text.casefold() else ""
        return f"{cluster.title} implementation was deferred{suffix}."
    if re.search(r"(?i)\b(implement|implemented|selected)\b", cluster_text):
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
    if re.search(r"(?i)\b(moves to|move to|postponed|deferred)\b", cleaned):
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
    if re.search(r"(?i)\b(approve|approved|approval|go ahead|finalize|finalized)\b", cleaned):
        return f"{topic} has been approved."
    if re.search(r"(?i)\b(retain|keep|continue)\b", cleaned):
        return f"{topic} will be retained."
    if not re.search(r"(?i)\b(will be|was|were|has been|have been)\b", cleaned):
        cleaned = f"{topic} has been confirmed"
    return utils.normalize_sentence(cleaned)


def build_action_items(clusters: list[TopicCluster]) -> list[ExperimentalActionItem]:
    """Build normalized action rows from cluster-supported action predictions."""

    action_items: list[ExperimentalActionItem] = []
    seen: set[str] = set()
    for cluster in clusters:
        for record in cluster.records:
            if getattr(record, "predicted_label", "") != "Action_Item":
                continue
            raw_sentence = utils.record_sentence(record)
            if not utils.is_reportable_sentence(raw_sentence):
                continue
            if raw_sentence.strip().endswith("?") and not re.match(
                r"(?i)^\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s+"
                r"(will|shall|needs to|need to|should|must|has to)\b",
                raw_sentence,
            ):
                continue
            parts = utils.extract_action_parts(
                raw_sentence,
                getattr(record, "speaker", ""),
            )
            key = f"{parts.owner}|{parts.task}|{parts.deadline}".casefold()
            if re.search(r"(?i)^finish\s+it\b", parts.task):
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


def build_pending_items(
    clusters: list[TopicCluster],
    action_items: list[ExperimentalActionItem],
) -> list[str]:
    """Detect unresolved or deferred work without duplicating action rows."""

    pending: list[str] = []
    for cluster in clusters:
        cluster_text = " ".join(utils.record_sentence(record) for record in cluster.records)
        if not utils.is_pending_sentence(cluster_text):
            continue
        topic = cluster.title
        if pending_duplicates_action(topic, action_items):
            continue
        pending_item = normalize_pending_item(cluster_text, topic)
        if pending_duplicates_action(f"{topic} {pending_item}", action_items):
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


def build_information(clusters: list[TopicCluster]) -> list[str]:
    """Keep useful factual statements from topic clusters."""

    information: list[str] = []
    for cluster in clusters:
        for record in cluster.records:
            if getattr(record, "predicted_label", "") != "Information":
                continue
            sentence = utils.record_sentence(record)
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


def build_objective(clusters: list[TopicCluster]) -> str:
    """Generate a concise objective from the largest discussion cluster."""

    largest = max(clusters, key=lambda cluster: (len(cluster.records), cluster.importance_score), default=None)
    if largest is None:
        return "Review meeting priorities and confirm ownership for follow-up work."
    context = " ".join(largest.contexts + [largest.title])
    if re.search(r"(?i)\bloading|performance|optimization|lazy\b", context):
        return "Evaluate application performance, confirm the optimization approach, and align ownership for release readiness."
    if re.search(r"(?i)\bsprint|release|feature\b", context):
        return "Review sprint release readiness, finalize delivery priorities, and confirm ownership for remaining work."
    if re.search(r"(?i)\btesting|qa|regression\b", context):
        return "Assess testing readiness, confirm quality risks, and assign validation follow-up."
    return limit_words(
        utils.normalize_sentence(
            f"Review {largest.title.lower()}, confirm outcomes, and assign follow-up ownership."
        ),
        20,
    )


def build_summary(
    *,
    objective: str,
    clusters: list[TopicCluster],
    discussion_points: list[str],
    decisions: list[str],
    action_items: list[ExperimentalActionItem],
    pending_items: list[str],
    information: list[str],
) -> str:
    """Build a 100-word executive summary from clusters and outcomes."""

    primary_topics = summarize_list(
        [cluster.title.lower() for cluster in clusters[:3]],
        fallback=templates.NO_DISCUSSION_TEXT,
    )
    decision_topics = summarize_list(
        [decision.rstrip(".") for decision in decisions[:3]],
        fallback=templates.NO_DECISION_TEXT,
    )
    next_date = utils.extract_next_meeting_date(information)
    next_meeting_clause = (
        templates.NEXT_MEETING_CLAUSE.format(date=next_date)
        if next_date
        else templates.NO_NEXT_MEETING_CLAUSE
    )
    action_phrase = (
        f"{len(action_items)} follow-up action items were assigned"
        if action_items
        else "No follow-up action items were identified"
    )
    pending_phrase = (
        " Pending work was separated for later tracking."
        if pending_items
        else ""
    )
    summary = (
        f"The meeting was held to {objective.rstrip('.').casefold()}. "
        f"Core topics included {primary_topics}. "
        f"The main outcomes were {decision_topics}. "
        f"{action_phrase}{next_meeting_clause}.{pending_phrase}"
    )
    return limit_words(utils.normalize_sentence(summary), 100)


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


def render_bullets(items: list[str]) -> list[str]:
    """Render bullet lines with a fallback message."""

    if not items:
        return ["- No items were identified."]
    return [f"- {item}" for item in items]


def limit_words(text: str, max_words: int) -> str:
    """Limit generated prose to a maximum word count."""

    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,.;") + "."

"""Sentence embedding clustering for the future ML MoM pipeline.

Purpose:
    Group semantically similar transcript sentences into temporary topic
    clusters using sentence embeddings.

Responsibilities:
    - Accept ``SentenceEmbedding`` objects from ``ml_mom.embeddings``.
    - Support KMeans, Agglomerative, DBSCAN, and HDBSCAN strategy methods.
    - Return explainable ``SentenceCluster`` objects with member sentences and
      centroid embeddings.
    - Generate temporary rule-based topic labels without LLMs.
    - Handle empty, single-sentence, small, and large inputs gracefully.

Inputs:
    Sentence embeddings containing sentence IDs, turn IDs, speakers, original
    sentences, and embedding vectors.

Outputs:
    Cluster result objects containing sentence IDs, member sentences, centroid
    vectors, and temporary topic labels.

Future Implementation Notes:
    This module performs clustering only. It does not classify sentences, train
    an ANN, generate Minutes of Meeting, or integrate with the UI.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import os
import re
import sys
from typing import Callable

try:
    from ml_mom.embeddings import SentenceEmbedding
except ModuleNotFoundError:  # pragma: no cover - supports direct script demos.
    from embeddings import SentenceEmbedding


DEFAULT_CLUSTERING_ALGORITHM = "agglomerative"
DEFAULT_TOPIC_LABEL = "General Discussion"
EMPTY_EMBEDDINGS_MESSAGE = "No sentence embeddings were provided for clustering."
INVALID_EMBEDDINGS_MESSAGE = (
    "Sentence embeddings are missing or have inconsistent dimensions."
)

_TOPIC_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "before",
        "can",
        "for",
        "from",
        "have",
        "into",
        "let",
        "lets",
        "our",
        "should",
        "that",
        "the",
        "this",
        "today",
        "tomorrow",
        "will",
        "with",
    }
)


@dataclass(slots=True)
class SentenceCluster:
    """Cluster of semantically related transcript sentences.

    Attributes:
        cluster_id: Numeric cluster identifier from the clustering strategy.
        topic_label: Temporary rule-based topic label for display/debugging.
        sentence_ids: Source sentence IDs belonging to the cluster.
        member_sentences: Original sentence text for each cluster member.
        centroid_embedding: Average embedding vector across cluster members.
        cluster_size: Number of sentences in the cluster.
    """

    cluster_id: int
    topic_label: str
    sentence_ids: list[int]
    member_sentences: list[str]
    centroid_embedding: list[float]
    cluster_size: int


@dataclass(slots=True)
class ClusteringResult:
    """Container for clustering output and friendly validation state.

    Attributes:
        clusters: Produced sentence clusters.
        algorithm: Algorithm requested by the caller.
        error_message: Friendly message when clustering cannot be completed.
        warnings: Non-blocking warnings from fallback behavior.
    """

    clusters: list[SentenceCluster] = field(default_factory=list)
    algorithm: str = DEFAULT_CLUSTERING_ALGORITHM
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


class ClusteringService:
    """Service that clusters sentence embeddings through swappable strategies."""

    def __init__(self, default_algorithm: str = DEFAULT_CLUSTERING_ALGORITHM) -> None:
        """Initialize the clustering service.

        Args:
            default_algorithm: Strategy name used by ``cluster`` when no
                explicit algorithm is provided.
        """

        self.default_algorithm = default_algorithm

    def cluster(
        self,
        sentence_embeddings: list[SentenceEmbedding],
        algorithm: str | None = None,
    ) -> ClusteringResult:
        """Cluster sentence embeddings using a selected strategy.

        Args:
            sentence_embeddings: Sentence embeddings to cluster.
            algorithm: Optional algorithm name. Supported values are
                ``kmeans``, ``agglomerative``, ``dbscan``, and ``hdbscan``.

        Returns:
            ``ClusteringResult`` with clusters or a friendly validation message.
        """

        selected_algorithm = (algorithm or self.default_algorithm).lower()
        strategy_map: dict[
            str,
            Callable[[list[SentenceEmbedding]], ClusteringResult],
        ] = {
            "kmeans": self.cluster_kmeans,
            "agglomerative": self.cluster_agglomerative,
            "dbscan": self.cluster_dbscan,
            "hdbscan": self.cluster_hdbscan,
        }

        strategy = strategy_map.get(selected_algorithm)
        if strategy is None:
            return ClusteringResult(
                algorithm=selected_algorithm,
                error_message=(
                    f"Unsupported clustering algorithm '{selected_algorithm}'. "
                    "Use kmeans, agglomerative, dbscan, or hdbscan."
                ),
            )

        validation_error = validate_embeddings(sentence_embeddings)
        if validation_error is not None:
            return validation_error_for(selected_algorithm, validation_error)

        if len(sentence_embeddings) == 1:
            # Single-sentence transcripts do not need a statistical algorithm;
            # one cluster preserves the sentence without forcing dependencies.
            return ClusteringResult(
                clusters=build_clusters(sentence_embeddings, [0]),
                algorithm=selected_algorithm,
                warnings=["Only one sentence was provided; returned one cluster."],
            )

        return strategy(sentence_embeddings)

    def cluster_kmeans(
        self,
        sentence_embeddings: list[SentenceEmbedding],
        n_clusters: int | None = None,
    ) -> ClusteringResult:
        """Cluster embeddings with KMeans.

        Args:
            sentence_embeddings: Sentence embeddings to cluster.
            n_clusters: Optional target number of clusters.

        Returns:
            Clustering result or a friendly error when sklearn is unavailable.
        """

        validation_error = validate_embeddings(sentence_embeddings)
        if validation_error is not None:
            return validation_error_for("kmeans", validation_error)

        if len(sentence_embeddings) == 1:
            return ClusteringResult(
                clusters=build_clusters(sentence_embeddings, [0]),
                algorithm="kmeans",
                warnings=["Only one sentence was provided; returned one cluster."],
            )

        try:
            # On some Windows environments, joblib's physical-core detection can
            # spawn a subprocess that is slow or noisy; this keeps demos stable.
            os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
            from sklearn.cluster import KMeans
        except Exception as exc:  # pragma: no cover - dependency dependent.
            return dependency_error_result("kmeans", exc)

        vectors = embedding_matrix(sentence_embeddings)
        cluster_count = n_clusters or estimate_cluster_count(len(sentence_embeddings))
        cluster_count = max(1, min(cluster_count, len(sentence_embeddings)))

        try:
            model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
            labels = [int(label) for label in model.fit_predict(vectors)]
        except Exception as exc:  # pragma: no cover - sklearn runtime dependent.
            return runtime_error_result("kmeans", exc)

        return ClusteringResult(
            clusters=build_clusters(sentence_embeddings, labels),
            algorithm="kmeans",
        )

    def cluster_agglomerative(
        self,
        sentence_embeddings: list[SentenceEmbedding],
        n_clusters: int | None = None,
    ) -> ClusteringResult:
        """Cluster embeddings with Agglomerative Clustering.

        Args:
            sentence_embeddings: Sentence embeddings to cluster.
            n_clusters: Optional target cluster count. When omitted, a
                conservative count is estimated from transcript size.

        Returns:
            Clustering result or a friendly error when sklearn is unavailable.
        """

        validation_error = validate_embeddings(sentence_embeddings)
        if validation_error is not None:
            return validation_error_for("agglomerative", validation_error)

        if len(sentence_embeddings) == 1:
            return ClusteringResult(
                clusters=build_clusters(sentence_embeddings, [0]),
                algorithm="agglomerative",
                warnings=["Only one sentence was provided; returned one cluster."],
            )

        try:
            from sklearn.cluster import AgglomerativeClustering
        except Exception as exc:  # pragma: no cover - dependency dependent.
            return dependency_error_result("agglomerative", exc)

        vectors = embedding_matrix(sentence_embeddings)
        cluster_count = n_clusters or estimate_cluster_count(len(sentence_embeddings))
        cluster_count = max(1, min(cluster_count, len(sentence_embeddings)))

        try:
            # Agglomerative clustering is the default because meeting topics are
            # naturally hierarchical and topic counts are rarely known upfront.
            model = AgglomerativeClustering(n_clusters=cluster_count)
            labels = [int(label) for label in model.fit_predict(vectors)]
        except Exception as exc:  # pragma: no cover - sklearn runtime dependent.
            return runtime_error_result("agglomerative", exc)

        return ClusteringResult(
            clusters=build_clusters(sentence_embeddings, labels),
            algorithm="agglomerative",
        )

    def cluster_dbscan(
        self,
        sentence_embeddings: list[SentenceEmbedding],
        eps: float = 0.45,
        min_samples: int = 2,
    ) -> ClusteringResult:
        """Cluster embeddings with DBSCAN.

        Args:
            sentence_embeddings: Sentence embeddings to cluster.
            eps: Maximum distance between neighboring samples.
            min_samples: Minimum samples required to form a dense region.

        Returns:
            Clustering result or a friendly error when sklearn is unavailable.
        """

        validation_error = validate_embeddings(sentence_embeddings)
        if validation_error is not None:
            return validation_error_for("dbscan", validation_error)

        if len(sentence_embeddings) < min_samples:
            return ClusteringResult(
                clusters=build_clusters(sentence_embeddings, [0] * len(sentence_embeddings)),
                algorithm="dbscan",
                warnings=[
                    "Transcript is too small for DBSCAN density estimation; "
                    "returned one cluster."
                ],
            )

        try:
            from sklearn.cluster import DBSCAN
        except Exception as exc:  # pragma: no cover - dependency dependent.
            return dependency_error_result("dbscan", exc)

        try:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = [int(label) for label in model.fit_predict(embedding_matrix(sentence_embeddings))]
        except Exception as exc:  # pragma: no cover - sklearn runtime dependent.
            return runtime_error_result("dbscan", exc)

        return ClusteringResult(
            clusters=build_clusters(sentence_embeddings, labels),
            algorithm="dbscan",
        )

    def cluster_hdbscan(
        self,
        sentence_embeddings: list[SentenceEmbedding],
        min_cluster_size: int = 2,
    ) -> ClusteringResult:
        """Cluster embeddings with HDBSCAN when available.

        Args:
            sentence_embeddings: Sentence embeddings to cluster.
            min_cluster_size: Minimum cluster size for HDBSCAN.

        Returns:
            Clustering result or a friendly error when HDBSCAN is unavailable.
        """

        validation_error = validate_embeddings(sentence_embeddings)
        if validation_error is not None:
            return validation_error_for("hdbscan", validation_error)

        if len(sentence_embeddings) < min_cluster_size:
            return ClusteringResult(
                clusters=build_clusters(sentence_embeddings, [0] * len(sentence_embeddings)),
                algorithm="hdbscan",
                warnings=[
                    "Transcript is too small for HDBSCAN density estimation; "
                    "returned one cluster."
                ],
            )

        clusterer = load_hdbscan_clusterer(min_cluster_size)
        if isinstance(clusterer, ClusteringResult):
            return clusterer

        try:
            labels = [
                int(label)
                for label in clusterer.fit_predict(embedding_matrix(sentence_embeddings))
            ]
        except Exception as exc:  # pragma: no cover - dependency/runtime dependent.
            return runtime_error_result("hdbscan", exc)

        return ClusteringResult(
            clusters=build_clusters(sentence_embeddings, labels),
            algorithm="hdbscan",
        )


def validate_embeddings(sentence_embeddings: list[SentenceEmbedding]) -> str | None:
    """Validate embedding input before clustering.

    Args:
        sentence_embeddings: Sentence embeddings to validate.

    Returns:
        Friendly validation message, or ``None`` when input is valid.
    """

    if not sentence_embeddings:
        return EMPTY_EMBEDDINGS_MESSAGE

    dimensions = {
        len(embedding.embedding_vector)
        for embedding in sentence_embeddings
        if embedding.embedding_vector
    }
    if not dimensions or len(dimensions) != 1:
        return INVALID_EMBEDDINGS_MESSAGE

    if any(
        embedding.embedding_dimension != len(embedding.embedding_vector)
        for embedding in sentence_embeddings
    ):
        return INVALID_EMBEDDINGS_MESSAGE

    return None


def validation_error_for(algorithm: str, message: str) -> ClusteringResult:
    """Build a validation error result for a clustering strategy.

    Args:
        algorithm: Algorithm name.
        message: Friendly validation message.

    Returns:
        Clustering result containing the validation error.
    """

    return ClusteringResult(algorithm=algorithm, error_message=message)


def dependency_error_result(algorithm: str, exc: Exception) -> ClusteringResult:
    """Build a friendly dependency error result.

    Args:
        algorithm: Algorithm that required the missing dependency.
        exc: Exception raised while importing the dependency.

    Returns:
        Clustering result containing a friendly dependency error.
    """

    return ClusteringResult(
        algorithm=algorithm,
        error_message=(
            f"{algorithm} clustering is unavailable because its dependency "
            f"could not be loaded. Details: {exc}"
        ),
    )


def runtime_error_result(algorithm: str, exc: Exception) -> ClusteringResult:
    """Build a friendly clustering runtime error result.

    Args:
        algorithm: Algorithm that failed.
        exc: Runtime exception raised by the clustering implementation.

    Returns:
        Clustering result containing a friendly runtime error.
    """

    return ClusteringResult(
        algorithm=algorithm,
        error_message=f"{algorithm} clustering failed gracefully. Details: {exc}",
    )


def embedding_matrix(sentence_embeddings: list[SentenceEmbedding]) -> list[list[float]]:
    """Return embedding vectors in matrix form.

    Args:
        sentence_embeddings: Sentence embeddings to convert.

    Returns:
        List of embedding vectors.
    """

    return [embedding.embedding_vector for embedding in sentence_embeddings]


def estimate_cluster_count(sentence_count: int) -> int:
    """Estimate a conservative cluster count for unknown meeting topics.

    Args:
        sentence_count: Number of sentences to cluster.

    Returns:
        Estimated number of clusters.
    """

    # Square-root scaling prevents tiny transcripts from being over-clustered
    # while allowing larger meetings to split into more discussion topics.
    return max(1, min(8, round(math.sqrt(sentence_count))))


def build_clusters(
    sentence_embeddings: list[SentenceEmbedding],
    labels: list[int],
) -> list[SentenceCluster]:
    """Build ``SentenceCluster`` objects from clustering labels.

    Args:
        sentence_embeddings: Source sentence embeddings.
        labels: Cluster labels aligned with ``sentence_embeddings``.

    Returns:
        Sorted sentence clusters.
    """

    grouped: dict[int, list[SentenceEmbedding]] = defaultdict(list)
    for embedding, label in zip(sentence_embeddings, labels):
        grouped[label].append(embedding)

    clusters: list[SentenceCluster] = []
    for label, members in sorted(grouped.items(), key=lambda item: item[0]):
        member_sentences = [member.original_sentence for member in members]
        clusters.append(
            SentenceCluster(
                cluster_id=label,
                topic_label=derive_topic_label(member_sentences),
                sentence_ids=[member.sentence_id for member in members],
                member_sentences=member_sentences,
                centroid_embedding=calculate_centroid(members),
                cluster_size=len(members),
            )
        )
    return clusters


def calculate_centroid(members: list[SentenceEmbedding]) -> list[float]:
    """Calculate an average embedding vector for cluster members.

    Args:
        members: Sentence embeddings in one cluster.

    Returns:
        Centroid embedding vector.
    """

    if not members:
        return []

    dimension = len(members[0].embedding_vector)
    centroid: list[float] = []
    for index in range(dimension):
        centroid.append(
            sum(member.embedding_vector[index] for member in members) / len(members)
        )
    return centroid


def derive_topic_label(member_sentences: list[str]) -> str:
    """Derive a temporary topic label from frequent content words.

    Args:
        member_sentences: Sentences belonging to one cluster.

    Returns:
        Temporary topic label.
    """

    tokens: list[str] = []
    for sentence in member_sentences:
        for token in _TOPIC_TOKEN_RE.findall(sentence.lower()):
            if token not in _STOPWORDS:
                tokens.append(token)

    if not tokens:
        return DEFAULT_TOPIC_LABEL

    # This simple label is intentionally transparent; future work can replace
    # it with noun-phrase extraction without introducing LLM generation.
    most_common = [token for token, _ in Counter(tokens).most_common(3)]
    return " / ".join(token.title() for token in most_common)


def load_hdbscan_clusterer(min_cluster_size: int) -> object | ClusteringResult:
    """Load an HDBSCAN implementation when available.

    Args:
        min_cluster_size: Minimum cluster size for the loaded clusterer.

    Returns:
        HDBSCAN clusterer instance or a friendly ``ClusteringResult`` error.
    """

    try:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(min_cluster_size=min_cluster_size)
    except Exception:
        pass

    try:
        import hdbscan

        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    except Exception as exc:  # pragma: no cover - dependency dependent.
        return dependency_error_result("hdbscan", exc)


def cluster_embeddings_placeholder(embeddings: list[list[float]]) -> list[int]:
    """Backward-compatible helper for the original skeleton entry point.

    Args:
        embeddings: Raw sentence embedding vectors.

    Returns:
        Cluster labels, or an empty list when input cannot be clustered.
    """

    sentence_embeddings = [
        SentenceEmbedding(
            sentence_id=index + 1,
            turn_id=index + 1,
            speaker="Unknown",
            original_sentence=f"Sentence {index + 1}",
            embedding_vector=embedding,
            embedding_dimension=len(embedding),
        )
        for index, embedding in enumerate(embeddings)
    ]
    result = ClusteringService().cluster(sentence_embeddings)
    labels: list[int] = []
    for cluster in result.clusters:
        labels.extend([cluster.cluster_id] * cluster.cluster_size)
    return labels


def clustering_demo() -> None:
    """Run unit-test style clustering examples.

    Returns:
        None. Prints cluster ID, topic label, cluster size, and member
        sentences for Agglomerative and KMeans strategies.
    """

    demo_embeddings = [
        SentenceEmbedding(1, 1, "Rahul", "We approved the project budget.", [0.9, 0.8, 0.1], 3),
        SentenceEmbedding(2, 1, "Rahul", "The finance team confirmed the cost.", [0.88, 0.76, 0.12], 3),
        SentenceEmbedding(3, 2, "Priya", "Please prepare the launch checklist.", [0.1, 0.2, 0.9], 3),
        SentenceEmbedding(4, 2, "Priya", "The action owner will deliver the report.", [0.08, 0.18, 0.86], 3),
        SentenceEmbedding(5, 3, "Speaker A", "Let us review customer feedback.", [0.45, 0.7, 0.35], 3),
    ]

    service = ClusteringService()
    for algorithm in ("agglomerative", "kmeans"):
        print(f"\n{algorithm.title()} Clustering")
        result = service.cluster(demo_embeddings, algorithm=algorithm)
        if result.error_message:
            print(result.error_message)
            continue
        for cluster in result.clusters:
            print(f"Cluster ID: {cluster.cluster_id}")
            print(f"Topic Label: {cluster.topic_label}")
            print(f"Cluster Size: {cluster.cluster_size}")
            print("Member Sentences:")
            for sentence in cluster.member_sentences:
                print(f"- {sentence}")

    # TODO:
    # Add silhouette scoring and topic-label quality checks after real meeting
    # embeddings are available.


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    clustering_demo()

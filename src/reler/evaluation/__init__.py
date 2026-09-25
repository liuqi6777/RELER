"""Standalone evaluation utilities for RELER checkpoints."""

ENGLISH_V1_SUBSET_TASKS = (
    "SciFact",
    "ArguAna",
    "NFCorpus",
    "StackOverflowDupQuestions",
    "SciDocsRR",
    "BiorxivClusteringS2S",
    "MedrxivClusteringS2S",
    "TwentyNewsgroupsClustering",
    "SprintDuplicateQuestions",
    "Banking77Classification",
    "EmotionClassification",
    "MassiveIntentClassification",
    "STS17",
    "SICK-R",
    "STSBenchmark",
    "SummEval",
)


def get_tasks(
    names: list[str] | None,
    languages: list[str] | None = None,
    benchmark: str | None = None,
):
    """Resolve an MTEB benchmark or the repository's stable v1 subset."""
    import mteb

    if benchmark == "MTEB(eng, v1, subset)":
        return mteb.get_tasks(languages=languages, tasks=list(ENGLISH_V1_SUBSET_TASKS))
    if benchmark:
        return mteb.get_benchmark(benchmark).tasks
    return mteb.get_tasks(languages=languages, tasks=names)

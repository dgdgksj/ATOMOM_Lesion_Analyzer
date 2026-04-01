from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def artifacts_path(*parts: str) -> str:
    return repo_path("artifacts", *parts)


def runtime_cache_path(*parts: str) -> str:
    return artifacts_path("runtime_cache", *parts)


def experiment_results_path(*parts: str) -> str:
    return artifacts_path("experiment_results", *parts)

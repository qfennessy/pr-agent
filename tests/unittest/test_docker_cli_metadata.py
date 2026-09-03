import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_stage_copies_declared_project_metadata_before_install():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    required_metadata = {project["readme"], project["license"]["file"]}
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    cli_stage = dockerfile.split("FROM base AS cli", 1)[1]
    before_install = cli_stage.split("RUN uv sync --frozen --no-dev", 1)[0]
    copied_tokens = {
        token
        for line in before_install.splitlines()
        if line.startswith(("ADD ", "COPY "))
        for token in line.split()[1:-1]
    }

    assert required_metadata <= copied_tokens

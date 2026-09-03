import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_pinecone_dependency_matches_legacy_module_api():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependency = next(
        entry
        for entry in project["dependency-groups"]["similar-issue"]
        if Requirement(entry).name == "pinecone-client"
    )
    specifier = Requirement(dependency).specifier

    assert specifier.contains("2.2.4")
    assert not specifier.contains("3.0.0")

    locked = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    package = next(package for package in locked["package"] if package["name"] == "pinecone-client")
    assert package["version"] == "2.2.4"

"""Every deploy workflow must refuse to run from any ref but main.

Covers the engine deploy (deploy.yml) and the query service deploy
(deploy-query-service.yml, D37 #105). Parses each workflow's YAML and checks
that the first step of every job that can deploy is an explicit main-only
check, that it needs no checkout or credentials to run, and that the ref is
passed through env (never interpolated in run:).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOWS = [WORKFLOWS_DIR / "deploy.yml", WORKFLOWS_DIR / "deploy-query-service.yml"]
DEPLOY_COMMANDS = ("adk deploy", "gcloud run deploy")

pytestmark = pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)


def _load(workflow: Path) -> dict:
    return yaml.safe_load(workflow.read_text())


def _jobs(workflow: Path) -> dict:
    return _load(workflow)["jobs"]


def _deploys(job: dict) -> bool:
    return any(cmd in (s.get("run") or "") for s in job["steps"] for cmd in DEPLOY_COMMANDS)


def test_workflow_has_a_deploying_job(workflow):
    assert any(_deploys(j) for j in _jobs(workflow).values())


def test_deploying_job_runs_in_an_environment(workflow):
    """Per-environment values come from a GitHub Environment (D37)."""
    for name, job in _jobs(workflow).items():
        if _deploys(job):
            assert job.get("environment"), f"{name}: deploying job has no environment"


def test_guard_is_first_step_of_every_deploying_job(workflow):
    for name, job in _jobs(workflow).items():
        if not _deploys(job):
            continue
        first = job["steps"][0]
        assert "uses" not in first, f"{name}: first step must be the guard, not an action"
        run = first.get("run", "")
        assert first["env"]["DEPLOY_REF"] == "${{ github.ref }}", name
        for ref, code in [
            ("refs/heads/main", 0),
            ("refs/heads/feature", 1),
            ("refs/heads/main2", 1),
            ("refs/tags/x", 1),
            ("", 1),
        ]:
            r = subprocess.run(
                ["bash", "-c", run], env={"DEPLOY_REF": ref, "PATH": "/usr/bin:/bin"}
            )
            assert r.returncode == code, f"{name}: ref={ref!r}"


def test_no_expression_interpolated_in_guard_script(workflow):
    for job in _jobs(workflow).values():
        assert "${{" not in job["steps"][0].get("run", "")


def test_no_job_needs_gate_bypass(workflow):
    for name, job in _jobs(workflow).items():
        assert "if" not in job, f"{name}: job-level if could skip the guard"
        assert "if" not in job["steps"][0], f"{name}: guard must be unconditional"


def test_guard_cannot_be_neutralised(workflow):
    wf = _load(workflow)
    assert set(wf.get(True, wf.get("on"))) == {"workflow_dispatch"}
    for name, job in wf["jobs"].items():
        for key in ("if", "needs", "continue-on-error"):
            assert key not in job, f"{name}: job-level {key}"
        for key in ("if", "continue-on-error"):
            assert key not in job["steps"][0], f"{name}: guard step {key}"

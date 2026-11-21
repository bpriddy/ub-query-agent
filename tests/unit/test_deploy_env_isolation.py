"""
deploy env files — the sandbox flag is a property of the deploy, not of callers.

There is one production engine; the Chat bot and the Gemini Enterprise
registration point at it. Its isolation from experiments rests on a single
fact: the env file the production deploy bakes in has SANDBOX_ENABLED off, and
only the sandbox deploy's env file has it on. Both files are plain text that
anyone can edit, and production's used to be called `deploy-dev.env` — a name
that invited exactly the mistake this pins (adding a flag "to the dev file").
So the check lives in CI: read the workflow, find the file it deploys, and
refuse a build that would arm the sandbox in production, name the file "dev"
again, or pass a relative --env_file that would be silently skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY_WORKFLOW = REPO / ".github" / "workflows" / "deploy.yml"
SANDBOX_SCRIPT = REPO / "deployment" / "deploy-sandbox.sh"
TEARDOWN_SCRIPT = REPO / "deployment" / "teardown-sandbox.sh"
# Pinned here, in the test, on purpose: it is what CI checks the scripts and
# env files against. The scripts read it from GitHub Environment prod.
PROD_ENGINE_ID = "9136379226620952576"

TRUTHY = ("1", "true", "yes")  # mirrors config.py's parsing


def _env(path: Path) -> dict[str, str]:
    """Minimal dotenv parse — the files are deliberately KEY=VALUE only."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _flag_on(values: dict[str, str], key: str) -> bool:
    return values.get(key, "false").lower() in TRUTHY


def _env_file_named_in(text: str) -> Path:
    found = re.findall(r"--env_file=(\S+)", text)
    assert len(found) == 1, f"expected exactly one --env_file in {text[:60]!r}..., got {found}"
    named = found[0].strip("'\"")
    # The workflow anchors the path on $GITHUB_WORKSPACE (see the absolute-path
    # test below); resolve it against the repo the same way the runner would.
    return REPO / named.removeprefix("$GITHUB_WORKSPACE/")


def _prod_env_file() -> Path:
    return _env_file_named_in(DEPLOY_WORKFLOW.read_text())


def _sandbox_env_file() -> Path:
    text = SANDBOX_SCRIPT.read_text()
    match = re.search(r'^ENV_FILE="([^"]+)"', text, re.MULTILINE)
    assert match, 'deploy-sandbox.sh must declare ENV_FILE="..."'
    # The script anchors the path on $REPO_ROOT (adk deploy chdir()s before it
    # reads --env_file, so a relative path is silently skipped). Strip it here.
    return REPO / match.group(1).removeprefix("$REPO_ROOT/")


async def test_sandbox_script_passes_an_absolute_env_file_path():
    """A relative --env_file is looked up inside adk's temp folder and silently
    dropped — the engine comes up with no env and is not a sandbox at all."""
    text = SANDBOX_SCRIPT.read_text()
    match = re.search(r'^ENV_FILE="([^"]+)"', text, re.MULTILINE)
    assert match and match.group(1).startswith(("$REPO_ROOT/", "/")), match


async def test_production_deploy_env_has_the_sandbox_explicitly_off():
    """Explicit, not defaulted: a missing key would also be off today, but the
    line is the record of intent the next editor reads."""
    path = _prod_env_file()
    values = _env(path)
    assert "SANDBOX_ENABLED" in values, f"{path.name}: SANDBOX_ENABLED must be set explicitly"
    assert not _flag_on(values, "SANDBOX_ENABLED"), (
        f"{path.name} is baked into the PRODUCTION engine by deploy.yml and "
        "would arm the sandbox for live users"
    )


async def test_sandbox_deploy_env_has_the_sandbox_on():
    path = _sandbox_env_file()
    assert _flag_on(_env(path), "SANDBOX_ENABLED"), (
        f"{path.name}: a sandbox engine without SANDBOX_ENABLED ignores every override"
    )


async def test_production_and_sandbox_deploys_use_different_env_files():
    """The two deploys must not share a file, or one edit flips both engines."""
    assert _prod_env_file().resolve() != _sandbox_env_file().resolve()


async def test_sandbox_env_never_names_the_production_engine():
    values = _env(_sandbox_env_file())
    assert PROD_ENGINE_ID not in values.get("SANDBOX_AGENT_ENGINE_ID", "")


async def test_sandbox_script_does_not_register_with_gemini_enterprise():
    """The sandbox engine must not appear in the Gemini Enterprise agent list —
    registration is what routes real users to an engine."""
    text = SANDBOX_SCRIPT.read_text()
    invoking = [
        line
        for line in text.splitlines()
        if "register_agent" in line and not line.lstrip().startswith("#")
    ]
    assert invoking == [], f"deploy-sandbox.sh invokes register_agent: {invoking}"


def _run_guarded(script: Path, tmp_path: Path, args=(), **env: str):
    """Run a sandbox script with stub `gh`, `adk` and `curl` first on PATH.

    The stubs record that they were called; `gh` fails, as it does for an
    operator who is not logged in, unless a test overrides the engine id.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "calls"
    for tool in ("gh", "adk", "curl"):
        stub = bin_dir / tool
        stub.write_text(f'#!/bin/sh\necho {tool} >> "{calls}"\nexit 1\n')
        stub.chmod(0o755)
    run_env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "GCP_PROJECT_ID": "test-project", **env}
    result = subprocess.run(
        ["bash", str(script), *args], env=run_env, capture_output=True, text=True, cwd=REPO
    )
    called = calls.read_text().split() if calls.exists() else []
    return result, called


@pytest.mark.parametrize("script", [SANDBOX_SCRIPT, TEARDOWN_SCRIPT], ids=lambda p: p.name)
def test_sandbox_scripts_fail_closed_without_the_production_engine_id(script, tmp_path):
    """The prod engine id is no longer a literal in the scripts (D37, #105): it
    comes from GitHub Environment prod. If it cannot be read, the guard cannot
    run, so the script must stop before it deploys or deletes anything."""
    result, called = _run_guarded(script, tmp_path)
    assert result.returncode != 0
    assert "cannot resolve the PRODUCTION engine id" in result.stderr
    assert "adk" not in called and "curl" not in called


def test_sandbox_script_refuses_the_production_engine_id(tmp_path):
    """Deploying the sandbox build onto the production engine is refused before
    `adk deploy` runs."""
    sandbox_id = _env(_sandbox_env_file())["SANDBOX_AGENT_ENGINE_ID"]
    result, called = _run_guarded(SANDBOX_SCRIPT, tmp_path, PROD_AGENT_ENGINE_ID=sandbox_id)
    assert result.returncode != 0
    assert "PRODUCTION engine" in result.stderr
    assert "adk" not in called


def test_teardown_script_refuses_the_production_engine_id(tmp_path):
    result, called = _run_guarded(
        TEARDOWN_SCRIPT, tmp_path, args=[PROD_ENGINE_ID], PROD_AGENT_ENGINE_ID=PROD_ENGINE_ID
    )
    assert result.returncode != 0
    assert "PRODUCTION engine" in result.stderr
    assert "curl" not in called


def test_no_deploy_script_hardcodes_the_production_engine_id():
    """One source for the prod engine id: GitHub Environment prod."""
    for script in (REPO / "deployment").glob("*.sh"):
        assert PROD_ENGINE_ID not in script.read_text(), script.name


async def test_both_deploy_envs_set_the_ub_url():
    """config.py has no GUB_BASE_URL default any more (D37, #105), so an engine
    whose env file lacks it refuses to start. Both files must set it."""
    for path in (_prod_env_file(), _sandbox_env_file()):
        assert _env(path).get("GUB_BASE_URL", "").startswith("https://"), (
            f"{path.name}: set GUB_BASE_URL — the engine will not start without it"
        )


def test_config_refuses_to_start_without_the_ub_url(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "GUB_BASE_URL"}
    result = subprocess.run(
        [sys.executable, "-c", "import gub_agent.config"],
        env=env,
        cwd=tmp_path,  # no .env here for load_dotenv to find
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "GUB_BASE_URL is not set" in result.stderr


async def test_production_workflow_passes_an_absolute_env_file_path():
    """The bug this pins is silent, not loud: `adk deploy` chdir()s into its
    staging folder before reading --env_file, so a relative path resolves in
    the wrong directory and is skipped with no error. The engine then runs on
    config.py defaults and looks like a successful deploy — which is what
    happened to production for a month. Absolute, or the build fails."""
    text = DEPLOY_WORKFLOW.read_text()
    found = re.findall(r"--env_file=(\S+)", text)
    assert len(found) == 1, found
    named = found[0].strip("'\"")
    assert named.startswith(("$GITHUB_WORKSPACE/", "/")), (
        f"--env_file={named} is relative and would be silently dropped"
    )


async def test_production_env_file_is_not_named_dev():
    """The name is load-bearing documentation: for a month this file was called
    deploy-dev.env while being the file the PRODUCTION deploy bakes in."""
    name = _prod_env_file().name
    assert "dev" not in name.lower(), (
        f"{name} is deployed to production ({PROD_ENGINE_ID}) — a 'dev' name "
        "invites edits meant for a test engine"
    )


async def test_production_deploy_declares_emit_thinking_explicitly():
    """EMIT_THINKING sat at 1 in this file for its whole life and never reached
    the engine, so production has always run with thought summaries off. The
    key stays explicit so that whoever changes it is choosing, not discovering."""
    values = _env(_prod_env_file())
    assert "EMIT_THINKING" in values, (
        f"{_prod_env_file().name}: keep EMIT_THINKING explicit — the fix to the "
        "env_file path means its value now actually reaches production"
    )


async def test_both_deploy_envs_declare_the_file_search_flag():
    """search-01's kill switch, stated rather than defaulted — in BOTH files.

    Its value is a rollout decision (the rollout enables it only after the GUB
    it calls has FILE_SEARCH on, because a disabled backend answers `200 []`
    and the tool reads that as "no such file"), so this pins the DECLARATION,
    not the setting: an operator turning it on should be editing a line that is
    already in front of them, and a reader should not have to open config.py to
    learn which way an engine is running.
    """
    for path in (_prod_env_file(), _sandbox_env_file()):
        assert "FILE_SEARCH_ENABLED" in _env(path), (
            f"{path.name}: keep FILE_SEARCH_ENABLED explicit — it is the kill "
            "switch for the file-search tool, and its default is off"
        )


async def test_both_deploy_envs_state_the_latency_switches():
    """Each latency change has its own rollback, and each rollback is a line in
    BOTH files — the same value, so a sandbox run measures what production
    runs. Pins the declaration and the parity, not the setting."""
    switches = (
        "CRITIC_PARALLEL",
        "CRITIC_SKIP_FINAL_PASS",
        "ROUTER_THINKING_OFF",
        "FORMATTER_THINKING_OFF",
        "SPECULATIVE_DEEP",
        "SPECULATIVE_FAST_PRECHECK",
        "EAGER_ADK_IMPORTS",
        "EVIDENCE_ROWS_EVENT",
    )
    prod, sandbox = _env(_prod_env_file()), _env(_sandbox_env_file())
    for key in switches:
        assert key in prod and key in sandbox, (
            f"keep {key} explicit in both deploy env files — it is a rollback, and "
            "an operator should be editing a line that is already there"
        )
        assert _flag_on(prod, key) == _flag_on(sandbox, key), (
            f"{key} differs between production and the sandbox"
        )


async def test_both_deploy_envs_state_the_format_gate_early_abort():
    """FORMAT_GATE_EARLY_ABORT is a rollback like the switches above, but NOT
    one of them: its values differ on purpose (the sandbox runs `contract`,
    production stays 0 until the residual is accepted), and `_flag_on` reads
    `contract` as off, so the parity assert above would pass vacuously. This
    pins the declaration in both files and a value config.py understands."""
    for path in (_prod_env_file(), _sandbox_env_file()):
        values = _env(path)
        assert "FORMAT_GATE_EARLY_ABORT" in values, (
            f"{path.name}: keep FORMAT_GATE_EARLY_ABORT explicit — it is a rollback, "
            "and an operator should be editing a line that is already there"
        )
        assert values["FORMAT_GATE_EARLY_ABORT"] in ("0", "contract"), (
            f"{path.name}: FORMAT_GATE_EARLY_ABORT={values['FORMAT_GATE_EARLY_ABORT']} "
            "is neither 0 nor contract — config.py would silently read it as 0"
        )


async def test_production_deploy_verifies_the_env_landed():
    """A deploy that cannot fail on a skipped env file will hide the next one."""
    text = DEPLOY_WORKFLOW.read_text()
    assert "verify-engine-env.sh" in text, (
        "deploy.yml must read the env back off the engine after deploying"
    )
    assert (REPO / "deployment" / "verify-engine-env.sh").exists()


# The Gemini Enterprise app the production Chat bot federates over. The /ask
# engine must use its own environment's app (ub-query-agent#90, D37), never
# this one: a dev turn would otherwise run on prod's app and connectors.
PROD_WORKSPACE_APP = "gub-and-federated-blend"


async def test_workspace_is_configured_per_engine_and_never_on_prods_app():
    """Workspace's app and data stores come only from the engine's env file
    (config.py has no defaults). The sandbox engine, which UB dev's /ask
    calls, must never name the production Chat bot's app; the stores and the
    app are set together or not at all."""
    values = _env(_sandbox_env_file())
    app = values.get("WORKSPACE_ENGINE", "")
    assert PROD_WORKSPACE_APP not in app
    assert bool(app) == bool(values.get("WORKSPACE_DATA_STORE_IDS")), (
        "set WORKSPACE_ENGINE and WORKSPACE_DATA_STORE_IDS together"
    )
    if app:
        assert re.fullmatch(
            r"projects/[^/]+/locations/[^/]+/collections/[^/]+/engines/[^/]+", app
        ), app

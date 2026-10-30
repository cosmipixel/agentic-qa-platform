"""The selected pilots' inputs, each admitted before any of them is used
(ADR-0023, #51). Manifest answers never enter."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from aqa_core.compiled import CompiledScript
from aqa_core.config import ProjectConfig
from aqa_core.project import (
    SpecError,
    load_compiled,
    load_project,
    path_on_origin,
    start_origin,
)
from aqa_core.schema import parse_origin
from aqa_core.spec import Spec
from aqa_runner.bound_secrets import MissingSecretError, SecretLoggedError
from aqa_runner.executor import admitted


class UnusableSecretError(ValueError):
    """`<source>: unusable test secret`: a referenced test secret is unset,
    too short or unencodable, protocol logging is on, or a bound value is in
    another variable (API.md §7's 12)."""


@dataclass(frozen=True)
class ResetRequest:
    """The spec's reset hook: a POST to this URL on the run's start origin."""

    url: str


@dataclass(frozen=True)
class PilotInput:
    """One selected pilot, admitted for replay."""

    spec: Spec
    config: ProjectConfig
    script: CompiledScript
    start: str
    reset: ResetRequest | None


def load_pilots(
    qa_root: Path, compiled_dir: Path, selected_ids: Sequence[str], *, origin: str
) -> tuple[PilotInput, ...]:
    """Every selected pilot, or every pilot by ID when none is selected, each
    from `<compiled_dir>/<id>.json` and admitted to run on `origin` alone
    (`validate_pilot`). Unless all are admitted, raises ValueError naming
    `qa_root` or the script's path and a fixed category, never the problem's
    text; an unusable test secret only once no selected input has another
    problem, so the exit code doesn't depend on their order."""
    project = _quietly(partial(load_project, qa_root), OSError, SpecError)
    if project is None:
        raise ValueError(f"{qa_root}: invalid pilot input")
    ids = tuple(selected_ids) or tuple(sorted(project.specs))
    if not ids or len(set(ids)) < len(ids) or not set(ids) <= project.specs.keys():
        raise ValueError(f"{qa_root}: invalid selection")
    root = compiled_dir.resolve()
    pilots: list[PilotInput] = []
    unusable: UnusableSecretError | None = None
    for spec_id in ids:
        source = compiled_dir / f"{spec_id}.json"
        # Read where the check looked, so a link can't lead outside the root.
        path = source.resolve()
        script = (
            _quietly(partial(load_compiled, path, project.config), OSError, SpecError)
            if path.is_relative_to(root)
            else None
        )
        if script is None:
            raise ValueError(f"{source}: invalid compiled input")
        admit = partial(validate_pilot, source=source, origin=origin)
        try:
            pilots.append(admit(project.specs[spec_id], project.config, script))
        except UnusableSecretError as error:
            unusable = unusable or error
    if unusable is not None:
        raise unusable
    return tuple(pilots)


def validate_pilot(
    spec: Spec,
    config: ProjectConfig,
    script: CompiledScript,
    *,
    source: Path,
    origin: str,
) -> PilotInput:
    """`script`, already read strictly, admitted for `spec`: covering each of
    its expectations, reset by a valid hook when a step has side effects, and
    admitted as replay admits it (`aqa_runner.executor.admitted`): compiled
    from the spec as it is now, agreeing with the project's subject
    contracts, runnable by the executor, spelling no bound value where the
    run would use it, and with every test secret the spec references bound
    for this run; values are read to check, never kept. The run may reach
    `origin`, the app's stack, alone: base_url must be that origin, compared
    as origins, and neither the spec nor the config may add another.
    Otherwise raises ValueError naming `source` and a fixed category,
    checking the test secrets last (UnusableSecretError, the environment's)."""
    prepared = _quietly(partial(_prepare, spec, config, script), SpecError)
    if prepared is None or not _covered(spec, script):
        raise ValueError(f"{source}: invalid pilot input")
    start, reset, usable = prepared
    if start != parse_origin(origin):
        raise ValueError(f"{source}: base_url is not the stack's origin")
    if (
        spec.frontmatter.allowed_origins
        or config.egress.private_origins
        or config.egress.subresource_hosts
    ):
        raise ValueError(f"{source}: declares an origin beyond the stack's")
    if not usable:
        raise UnusableSecretError(f"{source}: unusable test secret")
    return PilotInput(spec, config, script, start, reset)


def _quietly[T](read: Callable[[], T], *errors: type[Exception]) -> T | None:
    """What `read` returns, or None when it raises one of `errors`. The error
    is dropped here, so a refusal raised later carries neither its text nor
    the error itself on `__context__`, where it could hold a secret value."""
    try:
        return read()
    except errors:
        return None


def _prepare(
    spec: Spec, config: ProjectConfig, script: CompiledScript
) -> tuple[str, ResetRequest | None, bool]:
    """The run's start origin, the spec's reset hook on it, and whether each
    test secret the spec references binds for it once replay admits `script`
    (`_admitted`)."""
    start = start_origin(None, config)
    return start, _reset(spec, start), _admitted(spec, config, script, start)


def _admitted(
    spec: Spec, config: ProjectConfig, script: CompiledScript, start: str
) -> bool:
    """Whether each test secret `spec` references binds for a run from
    `start`, once the executor admits `script` as replay does, before
    anything is opened (`aqa_runner.executor.admitted`, #239). Anything else
    it refuses is a SpecError, a binding the run doesn't allow included. A
    value with surrogates fails to encode with a UnicodeEncodeError holding
    it, so the binding's errors are dropped; any other ValueError is a
    defect and propagates."""
    environment = (MissingSecretError, SecretLoggedError, UnicodeEncodeError)
    bound = _quietly(partial(admitted, script, spec, config, start), *environment)
    return bound is not None


def _reset(spec: Spec, start: str) -> ResetRequest | None:
    """The spec's reset hook on `start`, joined as text, as replay's reset is:
    loading the spec held it to POST and a path (`Reset.http`)."""
    hook = spec.frontmatter.preconditions.reset
    return None if hook is None else ResetRequest(path_on_origin(start, hook.path))


def _covered(spec: Spec, script: CompiledScript) -> bool:
    """Whether `script` names `spec`, lists each expectation once, in order,
    with exactly the assertions that name it, realizes every required
    condition, and declares a reset hook if a step has side effects, as the
    pilot repeats every script: what admission asks beyond replay's."""
    frontmatter = spec.frontmatter
    expect = range(len(frontmatter.expect))
    rows = script.coverage.expectations
    listed = [set(row.assertions) for row in rows]
    named = [{a.id for a in script.assertions if a.expect_index == i} for i in expect]
    satisfied = {name for step in script.steps for name in step.satisfies}
    return (
        script.spec_id == frontmatter.id
        and [row.expect_index for row in rows] == list(expect)
        and listed == named
        and {condition.id for condition in script.coverage.requires} <= satisfied
        and (
            frontmatter.preconditions.reset is not None
            or not any(step.side_effect for step in script.steps)
        )
    )

"""What a replay admits before anything is opened (ADR-0024's #46 and #216
amendments; ADR-0026's #53 P9+P10 amendment): the steps and checks M1 can run,
each secret the spec references with a value, and no bound value spelled where
the run would write or use it. The executor runs this, `aqa replay` and
`aqa explore` call it, and a repair's script passes `accepted` too
(`aqa_runner.repairs`)."""

from collections.abc import Mapping, Sequence

from aqa_core.compiled import (
    Assertion,
    Click,
    CompiledScript,
    Fill,
    FillSecret,
    Navigate,
    Press,
    ProbeEquals,
    ProbeEqualsBaseline,
    Reload,
    Select,
    Target,
    VisibleUnoccluded,
)
from aqa_core.config import ProjectConfig
from aqa_core.project import SpecError, contract_problems
from aqa_core.spec import Spec, secret_references

from aqa_runner.bound_secrets import BoundSecret, bound_secrets
from aqa_runner.browser_session import one_key
from aqa_runner.redaction import Redactor

# The steps the executor runs: those that act on a target, and the rest.
type Targeted = Click | Fill | FillSecret | Select
type Untargeted = Navigate | Reload | Press


class BoundValueSpelledError(SpecError):
    """A bound value is spelled where the run would write or use it
    (`_spelling_bound_values`): a spec error found once the spec's test
    secrets are bound, with that binding, so whoever prints the refusal
    can scan it with them."""

    def __init__(
        self, problems: Sequence[str], bound: Mapping[str, BoundSecret]
    ) -> None:
        super().__init__(problems)
        self.bound = bound


def admitted(
    script: CompiledScript, spec: Spec, config: ProjectConfig, start: str
) -> tuple[
    list[Targeted | Untargeted], list[Assertion], Mapping[str, BoundSecret], Redactor
]:
    """`script`'s steps and assertions, and the test secrets a run of `spec`
    from the start origin `start` binds with their scan, once nothing stops
    `script` before anything is opened: it is compiled from the spec as it
    is, agrees with `config`'s subject contracts and runs in M1
    (`accepted`); every secret the spec references has a value
    (`bound_secrets`); and no bound value is spelled where the run would
    write or use it (`_spelling_bound_values`), which raises
    `BoundValueSpelledError`. Otherwise `SpecError`, or `bound_secrets`'
    own error. The script's own problems come first, so a spec error wins
    over a missing value (exit 5 before 12), and a stale script's other
    problems are moot. Public so `aqa replay` admits every spec before any
    run's reset (#239)."""
    if script.spec_hash != spec.spec_hash:
        raise SpecError(["compiled from another version of the spec: explore it again"])
    if problems := contract_problems(script, config):
        raise SpecError(problems)
    runnable, checks = accepted(script, spec)
    bound = bound_secrets(spec, start)
    redactor = Redactor(bound.values())
    if problems := _spelling_bound_values(spec, runnable, script.targets, redactor):
        raise BoundValueSpelledError(problems, bound)
    return runnable, checks, bound, redactor


def _spelling_bound_values(
    spec: Spec,
    steps: list[Targeted | Untargeted],
    targets: Mapping[str, Target],
    redactor: Redactor,
) -> list[str]:
    """Each place, among what replay writes to the steps journal or hands
    the browser to find a target with, holding a spelling of a value
    `redactor` binds, as written or JSON-escaped, as the journal writes it:
    the start path, each step's action as its intent records it, and each
    target's name, every string of its locators, scopes included, and its
    subject contract's region and part (`Redactor.spells`). A place is named
    by its position, never by its text (SECURITY §5: keeping values out of
    `steps.jsonl` is the caller's preflight)."""
    places: list[tuple[str, object]] = [
        ("preconditions.start_url", spec.frontmatter.preconditions.start_url)
    ]
    places.extend(
        (f"steps[{index}] (seq {step.seq})", intent_action(step))
        for index, step in enumerate(steps)
    )
    places.extend(
        (
            f"targets[{index}]",
            [name, target.model_dump(mode="json", exclude={"semantic"})],
        )
        for index, (name, target) in enumerate(targets.items())
    )
    return [
        f"{place}: a bound value's spelling is in it, so the replay doesn't start"
        for place, value in places
        if redactor.spells(value)
    ]


def intent_action(step: Targeted | Untargeted) -> dict[str, object]:
    """`step` as its intent records it: as compiled, less its seq, which the
    line holds, its side-effect fields and its links to the coverage plan."""
    return step.model_dump(
        mode="json", exclude={"seq", "side_effect", "side_effect_basis", "satisfies"}
    )


def accepted(
    script: CompiledScript, spec: Spec
) -> tuple[list[Targeted | Untargeted], list[Assertion]]:
    """`script`'s steps and assertions, or `SpecError` naming each step and
    check M1 can't run, before anything is opened (ADR-0024's #46
    amendment), and each `fill_secret` naming a secret `spec` doesn't
    reference, which has no binding in the run. Probe assertions and baseline
    definitions must name probes declared by the spec. Public because a
    script replay would refuse is never up to date (`aqa explore`, #216)."""
    referenced = {name for _, name in secret_references(spec.frontmatter)}
    runnable: list[Targeted | Untargeted] = []
    checks: list[Assertion] = []
    problems: list[str] = []
    for index, step in enumerate(script.steps):
        where = f"steps[{index}] (seq {step.seq})"
        if isinstance(step, FillSecret) and step.secret not in referenced:
            problems.append(
                f"{where}: fill_secret names {step.secret}, which the spec doesn't "
                "reference, so this run has no binding for it"
            )
        elif isinstance(step, Press) and not one_key(step.key):
            problems.append(
                f"{where}: press takes one key, with only modifiers held before "
                f"it: {step.key[:40]!r}"
            )
        else:
            runnable.append(step)
    declared = spec.frontmatter.preconditions.probes
    for index, assertion in enumerate(script.assertions):
        if (
            isinstance(assertion, ProbeEquals | ProbeEqualsBaseline)
            and assertion.probe not in declared
        ):
            problems.append(
                f"assertions[{index}] ({assertion.id}): probe {assertion.probe} is not declared by the spec"
            )
        if isinstance(assertion, VisibleUnoccluded) and not assertion.in_viewport:
            problems.append(
                f"assertions[{index}] ({assertion.id}): "
                "visible_unoccluded requires in_viewport: true"
            )
        else:
            checks.append(assertion)
    problems.extend(
        f"probe_baselines[{name}]: probe {name} is not declared by the spec"
        for name in script.probe_baselines
        if name not in declared
    )
    if problems:
        raise SpecError(problems)
    return runnable, checks

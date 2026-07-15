"""Conservative goal-state prefilter for pyactr conflict resolution."""

from __future__ import annotations

from typing import Any, Callable

from pyactr import productions, utilities


def _plain(value: Any) -> str | None:
    try:
        split = utilities.splitting(value)
    except Exception:
        return None
    if split.variables or split.negvariables or split.negvalues:
        return None
    if split.values is None:
        return None
    return str(split.values)


def _goal_signature(self, rulename: str) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    cache = getattr(self, "_experimental_perf_goal_signatures", None)
    if cache is None:
        cache = {}
        self._experimental_perf_goal_signatures = cache
    rule_callable = self.rules[rulename]["rule"]
    identity = id(rule_callable)
    cached = cache.get(rulename)
    if cached is not None and cached[0] == identity:
        return cached[1]
    try:
        lhs = next(rule_callable())
        query = lhs.get("=g")
    except Exception:
        query = None
    if query is None:
        signature = (None, ())
    else:
        typename = str(getattr(query, "typename", "")) or None
        constants: list[tuple[str, str]] = []
        try:
            for slot_name, raw in query.removeunused():
                value = _plain(raw)
                if value is not None:
                    constants.append((str(slot_name), value))
        except Exception:
            constants = []
        signature = (typename, tuple(constants))
    cache[rulename] = (identity, signature)
    return signature


def _goal_definitely_incompatible(self, rulename: str) -> bool:
    typename, constants = _goal_signature(self, rulename)
    if typename is None and not constants:
        return False
    goal = self.buffers.get("g")
    if not goal:
        return True
    current = next(iter(goal), None)
    if current is None:
        return True
    if typename and str(getattr(current, "typename", "")) != typename:
        return True
    if not constants:
        return False
    current_values: dict[str, str] = {}
    try:
        for slot_name, raw in current:
            value = _plain(raw)
            if value is not None:
                current_values[str(slot_name)] = value
    except Exception:
        return False
    return any(current_values.get(slot) != expected for slot, expected in constants)


def indexed_procedural_process(self, start_time=0):
    """pyactr 0.3.2 procedure that skips rules with impossible goal constants."""
    time = start_time
    self.procs.append(self._PROCEDURAL,)
    self._ProductionRules__actrvariables = {}
    yield productions.Event(productions.roundtime(time), self._PROCEDURAL, "CONFLICT RESOLUTION")

    max_utility = float("-inf")
    used_rulename = None
    self.used_rulename = None
    self.extra_tests = {}
    self.last_rule_slotvals = self.current_slotvals.copy()

    rule_names = self.ordered_rulenames
    if self.model_parameters.get("utility_learning"):
        rule_names = sorted(
            tuple(self.rules.keys()),
            key=lambda name: self.rules[name]["utility"],
            reverse=True,
        )
        self.ordered_rulenames = list(rule_names)

    for rulename in rule_names:
        self.used_rulename = rulename
        utility = self.rules[rulename]["utility"]
        if self.model_parameters["subsymbolic"]:
            # Consume exactly one noise sample for every rule as upstream does.
            utility += utilities.calculate_instantaneous_noise(
                self.model_parameters["utility_noise"]
            )
        if _goal_definitely_incompatible(self, rulename):
            continue
        production = self.rules[rulename]["rule"]()
        lhs = next(production)
        if max_utility <= utility and self.LHStest(
            lhs, self._ProductionRules__actrvariables.copy()
        ):
            max_utility = utility
            used_rulename = rulename
            if not self.model_parameters["subsymbolic"] or not self.model_parameters["utility_noise"]:
                break

    if used_rulename:
        self.used_rulename = used_rulename
        production = self.rules[used_rulename]["rule"]()
        self.rules.used_rulenames.setdefault(used_rulename, []).append(time)
        yield productions.Event(
            productions.roundtime(time), self._PROCEDURAL, f"RULE SELECTED: {used_rulename}"
        )
        time += self.model_parameters["rule_firing"]
        yield productions.Event(productions.roundtime(time), self._PROCEDURAL, self._UNKNOWN)

        lhs = next(production)
        if not self.LHStest(lhs, self._ProductionRules__actrvariables.copy(), True):
            yield productions.Event(
                productions.roundtime(time), self._PROCEDURAL,
                f"RULE STOPPED FROM FIRING: {used_rulename}",
            )
        else:
            if self.model_parameters["utility_learning"] and self.rules[used_rulename]["reward"] is not None:
                utilities.modify_utilities(
                    time,
                    self.rules[used_rulename]["reward"],
                    self.rules.used_rulenames,
                    self.rules,
                    self.model_parameters,
                )
                self.rules.used_rulenames = {}
            compiled_rulename, re_created = self.compile_rules()
            self.compile = []
            if re_created:
                yield productions.Event(
                    productions.roundtime(time), self._PROCEDURAL,
                    f"RULE {re_created}: {compiled_rulename}",
                )
            self.current_slotvals = {key: None for key in self.buffers}
            yield productions.Event(
                productions.roundtime(time), self._PROCEDURAL, f"RULE FIRED: {used_rulename}"
            )
            try:
                yield from self.update(next(production), time)
            except utilities.ACTRError as exc:
                raise utilities.ACTRError(
                    "The following rule is not defined correctly according to ACT-R: "
                    f"'{self.used_rulename}'. The following error occurred: {exc}"
                ) from exc
            if self.last_rule and self.last_rule != used_rulename:
                self.compile = [self.last_rule, used_rulename, self.last_rule_slotvals.copy()]
                self.last_rule_slotvals = {key: None for key in self.buffers}
            self.last_rule = used_rulename
    else:
        self.procs.remove(self._PROCEDURAL,)
        yield productions.Event(productions.roundtime(time), self._PROCEDURAL, "NO RULE FOUND")
    yield self.procs


def apply(remember: Callable[[str, Any, str], None]) -> None:
    remember("ProductionRules.procedural_process", productions.ProductionRules, "procedural_process")
    productions.ProductionRules.procedural_process = indexed_procedural_process


def clear_caches() -> None:
    return None

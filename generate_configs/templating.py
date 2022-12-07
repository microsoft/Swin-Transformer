import copy
import dataclasses
import enum
import random
import re
import typing
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union #Protocol
from typing_extensions import Protocol
import orjson
import preface
import scipy.stats

import logger, config

# from . import config, logging

Primitive = Union[str, int, float, bool]
StringDict = Dict[str, object]

logger = logger.init(__name__)


TEMPLATE_PATTERN = re.compile(r"\((\d+)\.\.\.(\d+)\)")

CONTINUOUS_DISTRIBUTION_SUFFIX = "__random-sample-distribution"


class Strategy(preface.SumType):
    grid = enum.auto()
    paired = enum.auto()
    random = enum.auto()


@dataclasses.dataclass(frozen=True)
class DiscreteDistribution:
    values: Sequence[Primitive]


class HasRandomVariates(Protocol):
    def rvs(self, **kwargs: Dict[str, Any]) -> float:
        ...


# def parse_dist(
#     raw_dist: object, raw_params: object
# ) -> Tuple[HasRandomVariates, Dict[str, Any]]:
#     try:
#         # We type ignore because we catch any type errors
#         parameter_list = list(raw_params)  # type: ignore
#     except TypeError:
#         raise ValueError(f"{raw_params} should be a sequence!")

#     dist_choices = typing.get_args(config.Distribution)
#     if raw_dist not in dist_choices:
#         raise ValueError(f"{raw_dist} must be one of {', '.join(dist_choices)}")
#     raw_dist = typing.cast(config.Distribution, raw_dist)

#     params = {}

#     if raw_dist == "uniform":
#         dist = scipy.stats.uniform
#         assert len(parameter_list) == 2

#         low, high = parameter_list
#         assert low < high
#         params = {"loc": low, "scale": high - low}
#     elif raw_dist == "normal":
#         dist = scipy.stats.norm
#         assert len(parameter_list) == 2

#         mean, std = parameter_list
#         params = {"loc": mean, "scale": std}
#     elif raw_dist == "loguniform":
#         dist = scipy.stats.loguniform
#         assert len(parameter_list) == 2

#         low, high = parameter_list
#         assert low < high
#         params = {"a": low, "b": high}
#     else:
#         preface.never(raw_dist)

#     return dist, params


@dataclasses.dataclass(frozen=True)
class ContinuousDistribution:
    fn: HasRandomVariates
    params: Dict[str, Any]

    @classmethod
    def parse(cls, value: object) -> Optional["ContinuousDistribution"]:
        assert isinstance(value, dict)

        assert "dist" in value
        assert "params" in value

        dist, params = parse_dist(value["dist"], value["params"])

        return cls(dist, params)


@dataclasses.dataclass(frozen=True)
class Hole:
    path: str
    distribution: Union[DiscreteDistribution, ContinuousDistribution]

    def __post_init__(self) -> None:
        assert isinstance(self.distribution, DiscreteDistribution) or isinstance(
            self.distribution, ContinuousDistribution
        ), f"self.distribution ({self.distribution}) is {type(self.distribution)}!"


def makehole(key: str, value: object) -> Optional[Hole]:
    if isinstance(value, list):
        values: List[Union[str, int]] = []

        for item in value:
            if isinstance(item, str):
                values += expand_numbers(item)
            else:
                values.append(item)

        return Hole(key, DiscreteDistribution(values))

    if isinstance(value, str):
        numbers = expand_numbers(value)

        if len(numbers) > 1:
            return Hole(key, DiscreteDistribution(numbers))

    if key.endswith(CONTINUOUS_DISTRIBUTION_SUFFIX):
        key = key.removesuffix(CONTINUOUS_DISTRIBUTION_SUFFIX)

        dist = ContinuousDistribution.parse(value)
        assert dist

        return Hole(key, dist)

    return None


def find_holes(template: StringDict) -> List[Hole]:
    """
    Arguments:
        template (StringDict): Template with potential holes
        no_expand (Set[str]): Fields to not treat as holes, even if we would otherwise.
    """
    holes = []

    # Make it a list so we can modify template during iteration.
    for key, value in list(template.items()):
        # Have to check makehole first because value might be a dict, but
        # if key ends with CONTINUOUS_DISTRIBUTION_SUFFIX, then we want to
        # parse that dict as a continuous distribution
        hole = makehole(key, value)
        if hole:
            holes.append(hole)
            template.pop(key)
        elif isinstance(value, dict):
            holes.extend(
                Hole(f"{key}.{hole.path}", hole.distribution)
                for hole in find_holes(value)
            )

    return holes


def sort_by_json(dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(
        sorted(dicts, key=lambda d: orjson.dumps(d, option=orjson.OPT_SORT_KEYS))
    )


# region FILLING


def grid_fill(filled: StringDict, holes: List[Hole]) -> List[StringDict]:
    if not holes:
        return [filled]

    experiments = []
    first, rest = holes[0], holes[1:]

    if not isinstance(first.distribution, DiscreteDistribution):
        raise RuntimeError(
            f"Must sample from DiscreteDistribution with strategy grid, not {type(first.distribution)}!"
        )

    for value in first.distribution.values:

        experiment = copy.deepcopy(filled)
        
        preface.dict.set(experiment, first.path, value)
        experiments.extend(grid_fill(experiment, rest))


    return sort_by_json(experiments)


def paired_fill(holes: List[Hole]) -> List[StringDict]:
    experiments = []

    assert all(isinstance(hole.distribution, DiscreteDistribution) for hole in holes)

    # We can type ignore because we assert that all distributions are discrete
    shortest_hole = min(holes, key=lambda h: len(h.distribution.values))  # type: ignore

    assert isinstance(shortest_hole.distribution, DiscreteDistribution)

    for i in range(len(shortest_hole.distribution.values)):
        experiment: StringDict = {}
        for hole in holes:
            assert isinstance(hole.distribution, DiscreteDistribution)
            preface.dict.set(experiment, hole.path, hole.distribution.values[i])

        experiments.append(experiment)

    return sort_by_json(experiments)


def random_fill(holes: List[Hole], count: int) -> List[StringDict]:
    experiments = []

    for _ in range(count):
        experiment: StringDict = {}
        for hole in holes:
            if isinstance(hole.distribution, DiscreteDistribution):
                preface.dict.set(
                    experiment, hole.path, random.choice(hole.distribution.values)
                )
            elif isinstance(hole.distribution, ContinuousDistribution):
                preface.dict.set(
                    experiment,
                    hole.path,
                    float(hole.distribution.fn.rvs(**hole.distribution.params)),
                )
            else:
                preface.never(hole.distribution)

        experiments.append(experiment)

    return experiments


# endregion


def generate(
    template: StringDict,
    strategy: Strategy,
    count: int = 0,
    *,
    no_expand: Optional[Set[str]] = None,
) -> List[StringDict]:
    """
    Turns a template (a dictionary with lists as values) into a list of experiments (dictionaries with no lists).

    If strategy is Strategy.Grid, returns an experiment for each possible combination of each value in each list. Strategy.Paired returns an experiment for sequential pair of values in each list.

    An example makes this clearer. If the template had 3 lists with lengths 5, 4, 10, respectively:

    Grid would return 5 x 4 x 10 = 200 experiments.

    Paired would return min(5, 4, 10) = 4 experiments

    Random would return <count> experiments

    Experiments are returned sorted by the JSON value.
    """
    ignored = {}
    if no_expand is not None:
        # print (no_expand)
        # print (template)

        # for field in no_expand:
        #     print (field)
        # exit()

        ignored = {field: preface.dict.pop(template, field) for field in no_expand}
    # print (template)
    # exit()

    template = copy.deepcopy(template)
    # print ("\n\n\n")
    # print (template)
    # print ("\n\n\n")
    holes = find_holes(template)
    # print (holes)
    
    if not holes:
        # We can return this directly because there are no holes.
        return [template]


    logger.info("Found all holes. [count: %d]", len(holes))

    experiments: List[StringDict] = []

    if strategy is Strategy.grid:
        filled = grid_fill({}, holes)
    elif strategy is Strategy.paired:
        filled = paired_fill(holes)
    elif strategy is Strategy.random:
        filled = random_fill(holes, count)
    else:
        preface.never(strategy)

    logger.info("Filled all holes. [count: %d]", len(filled))

    without_holes: StringDict = {}
    for key, value in preface.dict.flattened(template).items():
        if makehole(key, value):
            continue

        without_holes[key] = value

    for key, value in ignored.items():
        without_holes[key] = value

    experiments = [preface.dict.merge(exp, without_holes) for exp in filled]

    logger.info("Merged all experiment configs. [count: %d]", len(experiments))

    return sort_by_json(experiments)


def expand_numbers(obj: str) -> Union[List[str], List[int]]:
    """
    Given a string that potentially has the digits expander:

    "(0...34)"

    Returns a list of strings with each value in the range (inclusive, exclusive)

    ["0", "1", ..., "33"]
    """
    splits = re.split(TEMPLATE_PATTERN, obj, maxsplit=1)

    # One split means the pattern isn't present.
    if len(splits) == 1:
        return [obj]

    if len(splits) != 4:
        raise ValueError("Can't parse strings with more than one ( ... )")

    front, start_s, end_s, back = splits

    front_list = expand_numbers(front)
    back_list = expand_numbers(back)

    start = int(start_s)
    end = int(end_s)

    if start < end:
        spread = range(start, end)
    else:
        spread = range(start, end, -1)

    expanded = []
    for f in front_list:
        for i in spread:
            for b in back_list:
                expanded.append(f"{f}{i}{b}")

    try:
        return [int(i) for i in expanded]
    except ValueError:
        return expanded

# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from dataclasses import FrozenInstanceError, fields
from contextlib import nullcontext as does_not_raise

import pytest

from ai4rag.search_space.src.parameter import Parameter, ParameterValueError


class TestMethods:
    """all member test various methods defined on the `Parameter` class"""

    @pytest.mark.parametrize(
        "param, exp_res",
        [
            # TODO hashable repr of list of dicts
            (Parameter(name="cat", param_type="C", values=[0.5, 0.1]), (0.1, 0.5)),
            (Parameter(name="cat", param_type="C", values=["simple", "recursive"]), ("recursive", "simple")),
            (Parameter("int_param", param_type="I", v_min=0, v_max=10), None),
            (Parameter("real_param", param_type="R", v_min=0.0, v_max=0.10), None),
            (Parameter("bool_param", param_type="B", values=[True, False]), (False, True)),
        ],
    )
    def test__hashable_field_repr_categorical_values(self, param, exp_res):
        assert param._hashable_field_repr_categorical_values() == exp_res

    @pytest.mark.parametrize(
        "param,expected",
        [
            (Parameter("int_param", param_type="I", v_min=0, v_max=10), list(range(0, 11))),
            (Parameter("real_param", param_type="R", v_min=0, v_max=10), None),  # error
            (Parameter("bool_param", param_type="B", values=[False]), [True, False]),
            (Parameter("cat_param", param_type="C", values=["first", "second"]), ["first", "second"]),
            (Parameter("cat_param", param_type="C", values=[1, {"a": "b"}]), [1, {"a": "b"}]),
        ],
    )
    def test_all_values(self, param, expected):
        if param.param_type in ("B", "I", "C"):
            assert param.all_values() == expected
        else:
            with pytest.raises(ParameterValueError):
                param.all_values()

    @pytest.mark.parametrize(
        "kwargs,expectation",
        [
            ({"param_type": "invalid_type"}, pytest.raises(ParameterValueError)),
            ({"param_type": "I"}, pytest.raises(ParameterValueError)),
            ({"param_type": "R"}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "v_min": 0}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "v_max": 4}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "values": []}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "v_min": 0, "v_max": 4, "values": []}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "v_min": 0.0, "v_max": 4}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "v_min": 0, "v_max": 4.2}, pytest.raises(ParameterValueError)),
            ({"param_type": "I", "v_min": 0, "v_max": 4}, does_not_raise()),
            ({"param_type": "R", "v_min": 0.0}, pytest.raises(ParameterValueError)),
            ({"param_type": "R", "v_max": 0.4}, pytest.raises(ParameterValueError)),
            ({"param_type": "R", "values": []}, pytest.raises(ParameterValueError)),
            ({"param_type": "R", "v_min": 0.0, "v_max": 0.5}, does_not_raise()),
            ({"param_type": "C"}, pytest.raises(ParameterValueError)),
            ({"param_type": "C", "v_min": 0}, pytest.raises(ParameterValueError)),
            ({"param_type": "C", "values": []}, pytest.raises(ParameterValueError)),
            ({"param_type": "C", "values": [10]}, does_not_raise()),
            ({"param_type": "B"}, pytest.raises(ParameterValueError)),
            ({"param_type": "B", "values": [False], "v_min": 0}, pytest.raises(ParameterValueError)),
            ({"param_type": "B", "values": [False]}, does_not_raise()),
        ],
    )
    def test___post_init__(self, kwargs, expectation):
        with expectation:
            Parameter("name", **kwargs)

    @pytest.mark.parametrize(
        "other,expectation",
        [
            ([1, 2, 3], False),
            ((1,), False),
            (1, False),
            (0.5, False),
            (None, False),
            ("this is string", False),
            (Parameter(name="chunk_overlap", param_type="C", values=[0.5]), True),
            (Parameter(name="chunk_overlap", param_type="C", values=["0.5"]), False),
            (Parameter(name="chunk_overlap", param_type="C", values=[1.0]), False),
            (Parameter(name="CHUNK_OVERLAP", param_type="C", values=[0.5]), False),
            (Parameter(name="chunking", param_type="C", values=[{"method": "recursive"}]), False),
            (Parameter(name="bool_param", param_type="B", values=[False]), False),
            (Parameter(name="int_param", param_type="I", v_min=0, v_max=4), False),
            (Parameter(name="chunk_method", param_type="C", values=["recursive"]), False),
        ],
    )
    def test_compare_with_different_objects(self, other, expectation):  # TODO!
        param = Parameter(name="chunk_overlap", param_type="C", values=[0.5])
        res = param == other

        assert res == expectation

    @pytest.mark.parametrize(
        "param",
        [
            (Parameter("bool_param", param_type="B", values=[True])),
            (Parameter("categorical_param", param_type="C", values=["first", "second"])),
            (Parameter("categorical_param", param_type="C", values=[{"first": 1}, {"second": "second"}])),
            (Parameter("int_param", param_type="I", v_min=0, v_max=4)),
            (Parameter("real_param", param_type="R", v_min=0.1, v_max=0.9)),
        ],
    )
    def test_hashability(self, param):
        hash(param)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"param_type": "B", "values": [True, False]},
        {"param_type": "C", "values": ["a", "b"]},
        {"param_type": "R", "v_min": 1.00, "v_max": 10.00},
        {"param_type": "I", "v_min": 1, "v_max": 10},
    ],
)
def test_parameter_instance_creation(kwargs):
    p = Parameter(name="param", **kwargs)

    assert p.name == "param"

    for kw, val in kwargs.items():
        assert getattr(p, kw) == val

    for f in fields(p):  # default values for unspecified fields should be `None`
        if f.name not in kwargs and f.name != "name":
            assert getattr(p, f.name) == None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v_min": 0},
        {"v_max": 4},
        {"v_min": 0, "values": []},
    ],
)
def test_positional_parameters(kwargs):
    patt_prefix = r"required\s+positional\s+arguments.*"
    error_patter = rf"(?={patt_prefix}'name')(?={patt_prefix}'param_type')"
    with pytest.raises(TypeError, match=error_patter):
        Parameter(**kwargs)


def tests_only_expected_fields_present():
    assert ("name", "param_type", "v_min", "v_max", "values") == tuple(f.name for f in fields(Parameter))


class DummyField:
    name = "nonExistingName"


@pytest.mark.parametrize("field", [*fields(Parameter), DummyField])
def test_is_immutable(field):
    p1 = Parameter(name="chunk_overlap", param_type="C", values=[0.5])

    with pytest.raises(FrozenInstanceError):
        p1.__setattr__(field.name, None)

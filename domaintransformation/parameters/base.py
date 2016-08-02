from __future__ import absolute_import, division, print_function

from pymor.parameters.base import ParameterType


class ProductParameterType(ParameterType):
    """Product of two |ParameterTypes|. Uniqueness of keys is ensured.
    Parameters
    ----------
    parameter_type_1
        |ParameterType|
    parameter_type_2
        |ParameterType|
    Attributes
    ----------
    parameter_type_1
    parameter_type_2
    """

    def __init__(self, parameter_type_1, parameter_type_2):
        assert isinstance(parameter_type_1, ParameterType) or parameter_type_1 is None
        assert isinstance(parameter_type_2, ParameterType) or parameter_type_2 is None

        assert parameter_type_1 is not None or parameter_type_2 is not None, \
            "One of the ParameterTypes must not be None"

        param_names_1 = set(parameter_type_1.keys()) if parameter_type_1 else set()
        param_names_2 = set(parameter_type_2.keys()) if parameter_type_2 else set()

        assert not param_names_1.intersection(param_names_2), "Names of Parameters must be unique"

        if parameter_type_1 is None:
            parameter_type = parameter_type_2
        elif parameter_type_2 is None:
            parameter_type = parameter_type_1
        else:
            params = {k: v for k, v in parameter_type_1.items()}
            params.update({k: v for k, v in parameter_type_2.items()})
            parameter_type = ParameterType(params)
        super(ProductParameterType, self).__init__(parameter_type)

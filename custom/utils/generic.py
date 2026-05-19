import inspect
import tempfile
import warnings
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, ContextManager, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.utils._pytree as _torch_pytree
from packaging import version

from .import_utils import (
    get_torch_version,
    is_torch_available,
)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        if is_torch_available():
            if version.parse(get_torch_version()) >= version.parse("2.2"):
                _torch_pytree.register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                    serialized_type_name=f"{cls.__module__}.{cls.__name__}",
                )
            else:
                _torch_pytree._register_pytree_node(
                    cls,
                    _model_output_flatten,
                    partial(_model_output_unflatten, output_type=cls),
                )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not isinstance(first_field, torch.Tensor):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
    

def _model_output_flatten(output: ModelOutput) -> Tuple[List[Any], "_torch_pytree.Context"]:
    return list(output.values()), list(output.keys())

def _model_output_unflatten(
    values: Iterable[Any],
    context: "_torch_pytree.Context",
    output_type=None,
) -> ModelOutput:
    return output_type(**dict(zip(context, values)))

def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    if not is_torch_available():
        return int(x)

    import torch

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def torch_float(x):
    """
    Casts an input to a torch float32 tensor if we are in a tracing context, otherwise to a Python float.
    """
    if not is_torch_available():
        return int(x)

    import torch

    return x.to(torch.float32) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)

def set_attribute_for_modules(module: "torch.nn.Module", key: str, value: Any):
    """
    Set a value to a module and all submodules.
    """
    setattr(module, key, value)
    for submodule in module.children():
        set_attribute_for_modules(submodule, key, value)


def del_attribute_from_modules(module: "torch.nn.Module", key: str):
    """
    Delete a value from a module and all submodules.
    """
    # because we might remove it previously in case it's a shared module, e.g. activation function
    if hasattr(module, key):
        delattr(module, key)

    for submodule in module.children():
        del_attribute_from_modules(submodule, key)

def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    use_return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        is_requested_to_return_tuple = kwargs.pop("return_dict", True) is False
        is_configured_to_return_tuple = self.config.use_return_dict is False if hasattr(self, "config") else False

        # The following allows to convert output to tuple ONLY on top level forward call,
        # while internal modules of the model will return Output objects
        # to be able to use name-based attribute access in modeling code.

        # We will check if we are on top level module, if so, turn off to tuple conversion for all
        # underling calls.
        is_top_level_module = getattr(self, "_is_top_level_module", True)
        if is_configured_to_return_tuple and is_top_level_module:
            set_attribute_for_modules(self, "_is_top_level_module", False)

        try:
            output = func(self, *args, **kwargs)
            if is_requested_to_return_tuple or (is_configured_to_return_tuple and is_top_level_module):
                output = output.to_tuple()
        finally:
            # Remove the flag after the model forward call is finished.
            if is_configured_to_return_tuple and is_top_level_module:
                del_attribute_from_modules(self, "_is_top_level_module")

        return output

    return wrapper
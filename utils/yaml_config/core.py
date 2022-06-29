from __future__ import annotations

import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from prestring.python import PythonModule
from termcolor import colored

NONE_CLASS_FIELD = "__none_class_field__"  # for judgement of class field.
ROOT_CLASS_NAME = "Config"


def create_logger(name: str = "") -> logging.Logger:
    """Create logger.

    Returns:
        logging.Logger: Logger.
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # color fomatter
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "cyan")
        + colored("(%(filename)s %(lineno)d)", "yellow")  # noqa
        + ": %(levelname)s | %(message)s"  # noqa
    )

    # create console handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)
    return logger


@dataclass
class TypeValuePair:
    """The pair of type and value for yaml data."""

    type_: str
    value: Any


@dataclass
class Generator:
    """Generate python script of dataclass from yaml data."""

    yaml_data: dict[str, Any]

    module: PythonModule = field(init=False)
    indent: str = field(init=False, default="    ")
    value_dict: dict[type, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.value_dict = {int: "int", float: "float", str: "str", bool: "bool"}

    def _concat_cap_from_list(self, list_: list[str]) -> str:
        """Concatnate capitalized string from list element.

        Args:
            list_ (list[str]): List of string.

        Returns:
            str: String that all element capitalized.
        """
        return "".join([x.capitalize() for x in list_])

    def _extract_value_from_dict(self, parent_key_stack: list[str]) -> Any:
        """Extract value from dictionary.

        Args:
            parent_key_stack (list[str]): Key stack of nested dictionary.

        Returns:
            Any: Extracted value.
        """
        data = deepcopy(self.yaml_data)  # Copy
        for key in parent_key_stack:
            data = data[key]  # Update
        return data

    def _get_field_statement(self, var_name: str, type_: str, value: Any) -> str:
        """Get statement for dataclass field.

        Args:
            var_name (str): Name of variable.
            type_ (str): Annotation type.
            value (Any): Value of variable.

        Returns:
            str: String of field statement.
        """
        return f"{var_name}: {type_} = {value}"

    def _get_assign_statement_for_post_init(
        self, var_name: str, value: Any, *, is_indent: bool = True
    ) -> str:
        """Get assign statement for dataclass's `__post_init__`.

        Args:
            var_name (str): Name of variable.
            value (Any): Value of variable.
            is_indent (bool, optional): Whether require indent for statement. Defaults to True.

        Returns:
            str: String of assign statement.
        """
        state_ = f"{var_name} = {value}"
        return self.indent + state_ if is_indent else state_

    def make_dataclass(
        self, cls_name: str, child_types: list[TypeValuePair], data: dict[str, Any]
    ) -> None:
        """Make dataclass structure.

        Args:
            cls_name (str): Class name.
            value (list[TypeValuePair]): Value of data.
            # data (dict[str, Any]): Original data of yaml.
        """
        # Making dataclass structure.
        self.module.stmt("@dataclass")
        with self.module.class_(cls_name):
            post_init_list: list[str] = []  # For `__post_init__`.
            for var_name, v in zip(data, child_types):
                if v.type_.find("list") > -1:
                    assign_str = "field(init=False)"
                    post_init_list.append(
                        self._get_assign_statement_for_post_init(
                            f"self.{var_name}", v.value, is_indent=True
                        )
                    )
                elif v.value == NONE_CLASS_FIELD:
                    assign_str = f"{v.type_}()"
                else:
                    assign_str = str(v.value)
                self.module.stmt(self._get_field_statement(var_name, v.type_, assign_str))

            # `__post_init__`
            if len(post_init_list) > 0:
                self.module.sep()
                self.module.stmt("def __post_init__(self) -> None:")
                for statement in post_init_list:
                    self.module.stmt(statement)

            if cls_name == ROOT_CLASS_NAME:
                self.module.sep()
                with self.module.def_("update", "self", "data: dict[str, Any]"):
                    self.module.stmt("data_dict_as_attr = AttrDict.from_nested_dict(data)")
                    self.module.stmt("self.__dict__.update(data_dict_as_attr)")

    def get_type(
        self,
        key: str,
        value: Any,
        parent_key_stack: Optional[list[str]] = None,
    ) -> TypeValuePair:
        """Get type of default yaml value.

        Args:
            key (str): Key of `value`.
            value (Any): Value of `key`.
            parent_key (Optional[list[str]], optional): Parent key of `key`. Defaults to None.

        Raises:
            NotImplementedError: Exception of non-existance type in python.

        Returns:
            tuple[str, Any]: String of type annotation, Default value.
        """
        if isinstance(value, list):
            # Get type of element of list.
            type_ = self.get_type("", value[0]).type_

            # Convert single quote('') to double quote("")
            if type_ == "str":
                value = str(value).replace("'", '"')
            return TypeValuePair(f"list[{type_}]", value)

        elif isinstance(value, dict):
            # Whether this key is root.
            if parent_key_stack is None:
                # Go to next layer-level and make dataclass structure.
                parent_key_stack = [key]  # Update parent.
                self.make_dataclass(
                    cls_name=key.capitalize(),
                    child_types=[self.get_type(k, v, parent_key_stack) for k, v in value.items()],
                    data=self._extract_value_from_dict(parent_key_stack),
                )
                return TypeValuePair(key.capitalize(), NONE_CLASS_FIELD)
            else:
                # Go to next layer-level and make dataclass structure with parent.
                parent_key_stack.append(key)
                self.make_dataclass(
                    cls_name=self._concat_cap_from_list(parent_key_stack),
                    child_types=[self.get_type(k, v, parent_key_stack) for k, v in value.items()],
                    data=self._extract_value_from_dict(parent_key_stack),
                )
                cls_name = self._concat_cap_from_list(parent_key_stack)
                parent_key_stack.pop()
                return TypeValuePair(cls_name, NONE_CLASS_FIELD)

        else:
            # Default types.
            value_type = type(value)  # Get type of value.

            if value_type in [int, float, bool]:
                return TypeValuePair(self.value_dict[value_type], value)

            elif value_type is str:
                return TypeValuePair("str", f'"{value}"')

            elif value is None:
                return TypeValuePair("None", None)

            else:
                raise NotImplementedError(f"[{value_type}] is not supported.")


def generate(dict_data: dict[str, Any], dst_path: str) -> None:
    """Generate dataclass from dict data.

    Args:
        dict_data (dict[str, Any]): Data of config file.
        dst_path (str): Path of output destination.
    """
    logger = create_logger(name=__name__)  # Get logger.
    inner_generate(logger, dict_data, dst_path)


def generate_from_yaml_file(yaml_path: str, dst_path: str):
    """Generate dataclass from yaml file.

    Args:
        yaml_path (str): Path of yaml file.
        dst_path (str): Path of output destination.
    """
    logger = create_logger(name=__name__)  # Get logger.

    # Load yaml data.
    yaml_data: dict[str, Any] = yaml.load(open(yaml_path), Loader=yaml.FullLoader)
    logger.info(f"'{Path(yaml_path).name}' was loaded.")
    inner_generate(logger, yaml_data, dst_path)


def inner_generate(
    logger: logging.Logger,
    dict_data: dict[str, Any],
    dst_path: str,
) -> None:
    """Inner function of `generate` and `generate_from_file`.

    Args:
        logger (logging.Logger): Logger.
        yaml_data (dict[str, Any]): Data of config file.
        dst_path (str): Path of output destination.
    """
    gen = Generator(yaml_data=dict_data)
    gen.module = PythonModule(width=80)

    # write header(import modules and required class)
    gen.module.stmt("""\
# ----------------------------------------------------- #
# Automatically generated from yaml configuration file. #
# ----------------------------------------------------- #
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data: Any) -> Union[dict, "AttrDict"]:
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dict(data[key]) for key in data})

""")

    # Get types of default value.
    types = [gen.get_type(k, v, parent_key_stack=None) for k, v in dict_data.items()]
    gen.make_dataclass(ROOT_CLASS_NAME, types, dict_data)
    logger.info("dataclass structure was made.")
    # logger.info(md.module)

    # Write code to file.
    dst = Path(dst_path)
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, "w") as f:
        f.write(str(gen.module) + "\n")
    logger.info(f"Code was wrote to '{dst_path}'.")

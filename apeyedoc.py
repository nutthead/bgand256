#!/usr/bin/env python3
"""
apeyedoc.py - Extract and document the public API of the colour-science module

Author: Behrang Saeedzadeh <hello@behrang.org>
"""

import argparse
import enum
import importlib
import inspect
import json
import pkgutil
import sys
import warnings
from typing import Any, NotRequired, TypedDict

import yaml


def suppress_warnings() -> None:
    """Suppress common warnings during module inspection."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def get_type_signature(obj: Any) -> str | None:
    """Extract type signature from an object if available."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return None


def get_qualified_name(module_name: str, obj_name: str) -> str:
    """Get the fully qualified name of an object."""
    return f"{module_name}.{obj_name}"


def extract_docstring(obj: Any) -> str | None:
    """Extract and clean docstring from an object."""
    docstring = inspect.getdoc(obj)
    if docstring:
        return docstring.strip()
    return None


def is_public_name(name: str) -> bool:
    """Check if a name is considered public (doesn't start with underscore)."""
    return not name.startswith('_')


class FunctionInfo(TypedDict):
    name: str  # Will be renamed to functionName/function-name in output
    docstring: NotRequired[str | None]
    signature: NotRequired[str]


class MethodInfo(TypedDict):
    name: str  # Will be renamed to methodName/method-name in output
    docstring: NotRequired[str | None]
    signature: NotRequired[str]


class ClassInfo(TypedDict):
    name: str  # Will be renamed to className/class-name in output
    docstring: NotRequired[str | None]
    methods: NotRequired[list[MethodInfo]]


class EnumInfo(TypedDict):
    name: str  # Will be renamed to enumName/enum-name in output
    docstring: NotRequired[str | None]
    enum_type: NotRequired[str]  # Enum, IntEnum, Flag, etc.
    members: NotRequired[list[dict[str, Any]]]


class ConstantInfo(TypedDict):
    name: str  # Will be renamed to constantName/constant-name in output
    value: NotRequired[str]
    type_name: NotRequired[str]
    docstring: NotRequired[str | None]


class VariableInfo(TypedDict):
    name: str  # Will be renamed to variableName/variable-name in output
    type_name: NotRequired[str]
    value: NotRequired[str]
    docstring: NotRequired[str | None]


class ModuleInfo(TypedDict):
    name: str  # Will be renamed to moduleName/module-name in output
    docstring: NotRequired[str | None]
    file_path: NotRequired[str | None]


class APICollection(TypedDict):
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    enums: list[EnumInfo]
    constants: list[ConstantInfo]
    variables: list[VariableInfo]
    modules: list[ModuleInfo]


def _load_package(package_name: str) -> Any:
    """Load and return the specified package."""
    try:
        return importlib.import_module(package_name)
    except ImportError as e:
        print(f"Error: Could not import {package_name}: {e}", file=sys.stderr)
        sys.exit(1)


def _should_skip_module(modname: str) -> bool:
    """Check if a module should be skipped during analysis."""
    return 'test' in modname or '._' in modname


def _load_module(modname: str, verbose: bool) -> Any | None:
    """Load a module and handle errors gracefully."""
    try:
        return importlib.import_module(modname)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not import {modname}: {e}")
        return None


def _get_public_names(module: Any) -> list[str]:
    """Get public names from a module."""
    if hasattr(module, '__all__'):
        return module.__all__
    return [name for name in dir(module) if is_public_name(name)]


def _create_function_info(
    qualified_name: str, obj: Any, exclusions: list[str] | None = None
) -> FunctionInfo:
    """Create function information dictionary."""
    if exclusions is None:
        exclusions = []

    function_info: FunctionInfo = {
        "name": qualified_name,
    }
    if "docstrings" not in exclusions:
        function_info["docstring"] = extract_docstring(obj)
    if "signatures" not in exclusions:
        signature = get_type_signature(obj)
        if signature:
            function_info["signature"] = signature
    return function_info


def _collect_class_methods(
    obj: Any, exclusions: list[str] | None = None
) -> list[MethodInfo]:
    """Collect public methods from a class object."""
    if exclusions is None:
        exclusions = []

    methods: list[MethodInfo] = []

    # If methods are excluded, return empty list
    if "methods" in exclusions:
        return methods

    for method_name in dir(obj):
        if not is_public_name(method_name):
            continue

        try:
            method = getattr(obj, method_name)
            if not (inspect.ismethod(method) or inspect.isfunction(method)):
                continue

            method_info: MethodInfo = {
                "name": method_name,
            }
            if "docstrings" not in exclusions:
                method_doc = extract_docstring(method)
                if method_doc:
                    method_info["docstring"] = method_doc
            if "signatures" not in exclusions:
                method_signature = get_type_signature(method)
                if method_signature:
                    method_info["signature"] = method_signature
            methods.append(method_info)
        except Exception:
            continue

    return methods


def _create_class_info(
    qualified_name: str, obj: Any, exclusions: list[str] | None = None
) -> ClassInfo:
    """Create class information dictionary."""
    if exclusions is None:
        exclusions = []

    class_info: ClassInfo = {
        "name": qualified_name,
    }
    if "docstrings" not in exclusions:
        docstring = extract_docstring(obj)
        if docstring:
            class_info["docstring"] = docstring

    methods = _collect_class_methods(obj, exclusions)
    if methods:
        class_info["methods"] = methods

    return class_info


def _is_constant(name: str, obj: Any) -> bool:
    """Check if an object appears to be a constant."""
    # Constants are typically uppercase and not callable
    return (name.isupper() and
            not callable(obj) and
            not inspect.ismodule(obj) and
            not inspect.isclass(obj))


def _is_enum_class(obj: Any) -> bool:
    """Check if an object is an enum class."""
    try:
        return inspect.isclass(obj) and issubclass(obj, enum.Enum)
    except TypeError:
        return False


def _get_enum_type_name(obj: Any) -> str:
    """Get the specific enum type name."""
    if issubclass(obj, enum.IntFlag):
        return "IntFlag"
    elif issubclass(obj, enum.Flag):
        return "Flag"
    elif issubclass(obj, enum.IntEnum):
        return "IntEnum"
    elif hasattr(enum, 'StrEnum') and issubclass(obj, enum.StrEnum):
        return "StrEnum"
    elif issubclass(obj, enum.Enum):
        return "Enum"
    return "Enum"


def _create_enum_info(
    qualified_name: str, obj: Any, exclusions: list[str] | None = None
) -> EnumInfo:
    """Create enum information dictionary."""
    if exclusions is None:
        exclusions = []

    enum_info: EnumInfo = {
        "name": qualified_name,
    }
    if "docstrings" not in exclusions:
        docstring = extract_docstring(obj)
        if docstring:
            enum_info["docstring"] = docstring

    enum_info["enum_type"] = _get_enum_type_name(obj)

    # Collect enum members
    members: list[dict[str, Any]] = []

    # If members are excluded, skip member collection
    if "members" not in exclusions:
        try:
            for member in obj:
                member_dict: dict[str, Any] = {
                    "name": member.name,
                    "value": str(member.value)
                }
                if "docstrings" not in exclusions:
                    member_doc = extract_docstring(member)
                    if member_doc:
                        member_dict["docstring"] = member_doc
                members.append(member_dict)

            if members:
                enum_info["members"] = members
        except Exception:
            pass

    return enum_info


def _get_safe_value_repr(obj: Any) -> str:
    """Get a safe string representation of an object's value."""
    try:
        repr_str = repr(obj)
        # Limit length to avoid extremely long outputs
        if len(repr_str) > 200:
            return repr_str[:200] + "..."
        return repr_str
    except Exception:
        return "<repr unavailable>"


def _create_constant_info(qualified_name: str, obj: Any) -> ConstantInfo:
    """Create constant information dictionary."""
    constant_info: ConstantInfo = {
        "name": qualified_name
    }

    # Add type information
    type_name = type(obj).__name__
    if type_name != "type":
        constant_info["type_name"] = type_name

    # Add value representation
    constant_info["value"] = _get_safe_value_repr(obj)

    return constant_info


def _create_variable_info(qualified_name: str, obj: Any) -> VariableInfo:
    """Create variable information dictionary."""
    variable_info: VariableInfo = {
        "name": qualified_name
    }

    # Add type information
    type_name = type(obj).__name__
    if type_name != "type":
        variable_info["type_name"] = type_name

    # Add value representation for simple types
    if isinstance(obj, str | int | float | bool | type(None)):
        variable_info["value"] = _get_safe_value_repr(obj)

    return variable_info


def _is_submodule_of_package(obj: Any, package_name: str) -> bool:
    """Check if a module is actually a submodule of the target package."""
    if not inspect.ismodule(obj):
        return False

    # Check if the module name starts with the package name
    if hasattr(obj, "__name__"):
        module_name = obj.__name__
        return module_name.startswith(package_name + ".")

    return False


def _create_module_info(
    qualified_name: str, obj: Any, exclusions: list[str] | None = None
) -> ModuleInfo:
    """Create module information dictionary."""
    if exclusions is None:
        exclusions = []

    module_info: ModuleInfo = {
        "name": qualified_name,
    }
    if "docstrings" not in exclusions:
        docstring = extract_docstring(obj)
        if docstring:
            module_info["docstring"] = docstring

    # Add file path if available
    if hasattr(obj, "__file__") and obj.__file__:
        module_info["file_path"] = obj.__file__

    return module_info


def _process_object(
    obj: Any,
    modname: str,
    name: str,
    package_name: str,
    verbose: bool,
    exclusions: list[str] | None = None,
) -> tuple[
    FunctionInfo | None,
    ClassInfo | None,
    EnumInfo | None,
    ConstantInfo | None,
    VariableInfo | None,
    ModuleInfo | None
]:
    """Process a single object and return appropriate info."""
    if exclusions is None:
        exclusions = []

    qualified_name = get_qualified_name(modname, name)

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return (
            _create_function_info(qualified_name, obj, exclusions),
            None,
            None,
            None,
            None,
            None,
        )
    elif _is_enum_class(obj):
        return (
            None,
            None,
            _create_enum_info(qualified_name, obj, exclusions),
            None,
            None,
            None,
        )
    elif inspect.isclass(obj):
        return (
            None,
            _create_class_info(qualified_name, obj, exclusions),
            None,
            None,
            None,
            None,
        )
    elif inspect.ismodule(obj) and _is_submodule_of_package(obj, package_name):
        return (
            None,
            None,
            None,
            None,
            None,
            _create_module_info(qualified_name, obj, exclusions),
        )
    elif _is_constant(name, obj):
        return None, None, None, _create_constant_info(qualified_name, obj), None, None
    elif inspect.ismodule(obj):
        # Skip imported modules that are not submodules of this package
        return None, None, None, None, None, None
    else:
        # Treat as variable
        return None, None, None, None, _create_variable_info(qualified_name, obj), None


def _process_module(
    modname: str, package_name: str, verbose: bool, exclusions: list[str] | None = None
) -> tuple[
    list[FunctionInfo],
    list[ClassInfo],
    list[EnumInfo],
    list[ConstantInfo],
    list[VariableInfo],
    list[ModuleInfo]
]:
    """Process a single module and extract all API elements."""
    if exclusions is None:
        exclusions = []

    module = _load_module(modname, verbose)
    if module is None:
        return [], [], [], [], [], []

    if verbose:
        print(f"Examining module: {modname}")

    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    enums: list[EnumInfo] = []
    constants: list[ConstantInfo] = []
    variables: list[VariableInfo] = []
    modules: list[ModuleInfo] = []

    public_names = _get_public_names(module)

    for name in public_names:
        try:
            obj = getattr(module, name)
        except AttributeError:
            if verbose:
                print(f"Warning: {modname} has no attribute {name}")
            continue

        (
            function_info,
            class_info,
            enum_info,
            constant_info,
            variable_info,
            module_info,
        ) = _process_object(
            obj, modname, name, package_name, verbose, exclusions
        )

        if function_info:
            functions.append(function_info)
        if class_info:
            classes.append(class_info)
        if enum_info:
            enums.append(enum_info)
        if constant_info:
            constants.append(constant_info)
        if variable_info:
            variables.append(variable_info)
        if module_info:
            modules.append(module_info)

    return functions, classes, enums, constants, variables, modules


def collect_api_items(
    package_name: str = "colour",
    verbose: bool = False,
    exclusions: list[str] | None = None,
) -> APICollection:
    """
    Collect all public API elements from the specified package.

    Args:
        package_name: Name of the package to analyze
        verbose: Whether to print verbose output during collection
        exclusions: List of content types to exclude from output

    Returns:
        APICollection containing all discovered API elements
    """
    if exclusions is None:
        exclusions = []

    package = _load_package(package_name)

    if verbose:
        print(f"Analyzing package: {package_name}")

    all_functions: list[FunctionInfo] = []
    all_classes: list[ClassInfo] = []
    all_enums: list[EnumInfo] = []
    all_constants: list[ConstantInfo] = []
    all_variables: list[VariableInfo] = []
    all_modules: list[ModuleInfo] = []

    # Walk through all modules in the package
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + '.'
    ):
        if _should_skip_module(modname):
            continue

        (
            functions, classes, enums, constants, variables, modules
        ) = _process_module(modname, package_name, verbose, exclusions)

        # Apply exclusions at the collection level
        if "functions" not in exclusions:
            all_functions.extend(functions)
        if "classes" not in exclusions:
            all_classes.extend(classes)
        if "enums" not in exclusions:
            all_enums.extend(enums)
        if "constants" not in exclusions:
            all_constants.extend(constants)
        if "variables" not in exclusions:
            all_variables.extend(variables)
        if "modules" not in exclusions:
            all_modules.extend(modules)

    return {
        "functions": all_functions,
        "classes": all_classes,
        "enums": all_enums,
        "constants": all_constants,
        "variables": all_variables,
        "modules": all_modules
    }


def _format_functions_section(functions: list[FunctionInfo]) -> list[str]:
    """Format functions section for text output."""
    if not functions:
        return []

    lines = ["=== FUNCTIONS ===\n"]
    for func in sorted(functions, key=lambda x: x["name"]):
        name = func["name"]
        signature = func.get("signature", "")
        if signature:
            lines.append(f"Function: {name}{signature}")
        else:
            lines.append(f"Function: {name}")

        docstring = func.get("docstring")
        if docstring:
            lines.append(f"DocString: {docstring}")
        else:
            lines.append("DocString: <no documentation available>")

        lines.append("")
    return lines


def _format_classes_section(classes: list[ClassInfo]) -> list[str]:
    """Format classes section for text output."""
    if not classes:
        return []

    lines = ["=== CLASSES ===\n"]
    for cls in sorted(classes, key=lambda x: x["name"]):
        name = cls["name"]
        lines.append(f"Class: {name}")

        docstring = cls.get("docstring")
        if docstring:
            lines.append(f"DocString: {docstring}")
        else:
            lines.append("DocString: <no documentation available>")

        # Include methods if available
        methods = cls.get("methods", [])
        if methods:
            lines.append("Methods:")
            for method in sorted(methods, key=lambda x: x["name"]):
                method_name = method["name"]
                method_signature = method.get("signature", "")
                if method_signature:
                    lines.append(f"  - {method_name}{method_signature}")
                else:
                    lines.append(f"  - {method_name}")

                method_doc = method.get("docstring")
                if method_doc:
                    # Truncate long method docstrings
                    if len(method_doc) > 100:
                        method_doc = method_doc[:100] + "..."
                    lines.append(f"    {method_doc}")

        lines.append("")
    return lines


def _format_enums_section(enums: list[EnumInfo]) -> list[str]:
    """Format enums section for text output."""
    if not enums:
        return []

    lines = ["=== ENUMS ===\n"]
    for enum_item in sorted(enums, key=lambda x: x["name"]):
        name = enum_item["name"]
        enum_type = enum_item.get("enum_type", "Enum")
        lines.append(f"Enum: {name} ({enum_type})")

        docstring = enum_item.get("docstring")
        if docstring:
            lines.append(f"DocString: {docstring}")
        else:
            lines.append("DocString: <no documentation available>")

        # Include enum members if available
        members = enum_item.get("members", [])
        if members:
            lines.append("Members:")
            for member in members:
                member_name = member["name"]
                member_value = member["value"]
                lines.append(f"  - {member_name} = {member_value}")

                member_doc = member.get("docstring")
                if member_doc:
                    lines.append(f"    {member_doc}")

        lines.append("")
    return lines


def _format_constants_section(constants: list[ConstantInfo]) -> list[str]:
    """Format constants section for text output."""
    if not constants:
        return []

    lines = ["=== CONSTANTS ===\n"]
    for const in sorted(constants, key=lambda x: x["name"]):
        name = const["name"]
        type_name = const.get("type_name", "")
        value = const.get("value", "")

        if type_name:
            lines.append(f"Constant: {name} ({type_name})")
        else:
            lines.append(f"Constant: {name}")

        if value:
            lines.append(f"Value: {value}")

        lines.append("")
    return lines


def _format_variables_section(variables: list[VariableInfo]) -> list[str]:
    """Format variables section for text output."""
    if not variables:
        return []

    lines = ["=== VARIABLES ===\n"]
    for var in sorted(variables, key=lambda x: x["name"]):
        name = var["name"]
        type_name = var.get("type_name", "")
        value = var.get("value", "")

        if type_name:
            lines.append(f"Variable: {name} ({type_name})")
        else:
            lines.append(f"Variable: {name}")

        if value:
            lines.append(f"Value: {value}")

        lines.append("")
    return lines


def _format_modules_section(modules: list[ModuleInfo]) -> list[str]:
    """Format modules section for text output."""
    if not modules:
        return []

    lines = ["=== MODULES ===\n"]
    for mod in sorted(modules, key=lambda x: x["name"]):
        name = mod["name"]
        lines.append(f"Module: {name}")

        docstring = mod.get("docstring")
        if docstring:
            lines.append(f"DocString: {docstring}")
        else:
            lines.append("DocString: <no documentation available>")

        file_path = mod.get("file_path")
        if file_path:
            lines.append(f"File: {file_path}")

        lines.append("")
    return lines



def _rename_dict_keys_for_json(
    item: (
        dict[str, Any]
        | FunctionInfo
        | ClassInfo
        | EnumInfo
        | ConstantInfo
        | VariableInfo
        | ModuleInfo
    ),
    item_type: str,
) -> dict[str, Any]:  # type: ignore
    """Rename 'name' key in a dictionary based on the item type."""
    new_item = {}
    for key, value in item.items():
        if key == "name":
            if item_type == "function":
                new_key = "functionName"
            elif item_type == "class":
                new_key = "className"
            elif item_type == "method":
                new_key = "methodName"
            elif item_type == "enum":
                new_key = "enumName"
            elif item_type == "constant":
                new_key = "constantName"
            elif item_type == "variable":
                new_key = "variableName"
            elif item_type == "module":
                new_key = "moduleName"
            elif item_type == "member":
                new_key = "memberName"
            else:
                new_key = key
        else:
            new_key = key

        if key == "methods" and isinstance(value, list):
            new_item[new_key] = [
                _rename_dict_keys_for_json(method, "method")  # type: ignore[arg-type]
                for method in value  # type: ignore[misc]
                if isinstance(method, dict)
            ]
        elif key == "members" and isinstance(value, list):
            new_item[new_key] = [
                _rename_dict_keys_for_json(member, "member")  # type: ignore[arg-type]
                for member in value  # type: ignore[misc]
                if isinstance(member, dict)
            ]
        else:
            new_item[new_key] = value

    return new_item  # type: ignore[return-value]


def _get_yaml_key_name(key: str, item_type: str) -> str:
    """Get the appropriate YAML key name based on key and item type."""
    if key == "name":
        name_mappings = {
            "function": "function-name",
            "class": "class-name",
            "method": "method-name",
            "enum": "enum-name",
            "constant": "constant-name",
            "variable": "variable-name",
            "module": "module-name",
            "member": "member-name",
        }
        return name_mappings.get(item_type, key)

    key_mappings = {
        "enum_type": "enum-type",
        "type_name": "type-name",
        "file_path": "file-path",
    }
    return key_mappings.get(key, key)


def _rename_dict_keys_for_yaml(
    item: (
        dict[str, Any]
        | FunctionInfo
        | ClassInfo
        | EnumInfo
        | ConstantInfo
        | VariableInfo
        | ModuleInfo
    ),
    item_type: str,
) -> dict[str, Any]:  # type: ignore
    """Rename 'name' key in a dictionary based on the item type with kebab-case."""
    new_item = {}
    for key, value in item.items():
        new_key = _get_yaml_key_name(key, item_type)

        if key == "methods" and isinstance(value, list):
            new_item[new_key] = [
                _rename_dict_keys_for_yaml(method, "method")  # type: ignore[arg-type]
                for method in value  # type: ignore[misc]
                if isinstance(method, dict)
            ]
        elif key == "members" and isinstance(value, list):
            new_item[new_key] = [
                _rename_dict_keys_for_yaml(member, "member")  # type: ignore[arg-type]
                for member in value  # type: ignore[misc]
                if isinstance(member, dict)
            ]
        else:
            new_item[new_key] = value

    return new_item  # type: ignore[return-value]


def format_text_output(api_collection: APICollection) -> str:
    """Format API information as text."""
    output_lines: list[str] = []

    # Add each section
    output_lines.extend(_format_functions_section(api_collection["functions"]))
    output_lines.extend(_format_classes_section(api_collection["classes"]))
    output_lines.extend(_format_enums_section(api_collection["enums"]))
    output_lines.extend(_format_constants_section(api_collection["constants"]))
    output_lines.extend(_format_variables_section(api_collection["variables"]))
    output_lines.extend(_format_modules_section(api_collection["modules"]))

    return "\n".join(output_lines)


def format_json_output(
    api_collection: APICollection, package_name: str = "colour"
) -> str:
    """Format API information as JSON with type-specific keys."""
    # Transform the API collection with renamed keys
    transformed_collection = {
        "functions": [
            _rename_dict_keys_for_json(func, "function")
            for func in api_collection["functions"]
        ],
        "classes": [
            _rename_dict_keys_for_json(cls, "class")
            for cls in api_collection["classes"]
        ],
        "enums": [
            _rename_dict_keys_for_json(enum, "enum")
            for enum in api_collection["enums"]
        ],
        "constants": [
            _rename_dict_keys_for_json(const, "constant")
            for const in api_collection["constants"]
        ],
        "variables": [
            _rename_dict_keys_for_json(var, "variable")
            for var in api_collection["variables"]
        ],
        "modules": [
            _rename_dict_keys_for_json(mod, "module")
            for mod in api_collection["modules"]
        ]
    }

    # Wrap with package-api root key
    root_key = f"{package_name}-api"
    wrapped_data = {root_key: transformed_collection}

    return json.dumps(wrapped_data, indent=2, ensure_ascii=False)


def _represent_str(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Custom string representer that uses literal style for multiline strings."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')  # type: ignore[misc]
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)  # type: ignore[misc]


def format_yaml_output(
    api_collection: APICollection, package_name: str = "colour"
) -> str:
    """Format API information as YAML with type-specific kebab-case keys."""
    # Transform the API collection with renamed keys
    transformed_collection = {
        "functions": [
            _rename_dict_keys_for_yaml(func, "function")
            for func in api_collection["functions"]
        ],
        "classes": [
            _rename_dict_keys_for_yaml(cls, "class")
            for cls in api_collection["classes"]
        ],
        "enums": [
            _rename_dict_keys_for_yaml(enum, "enum")
            for enum in api_collection["enums"]
        ],
        "constants": [
            _rename_dict_keys_for_yaml(const, "constant")
            for const in api_collection["constants"]
        ],
        "variables": [
            _rename_dict_keys_for_yaml(var, "variable")
            for var in api_collection["variables"]
        ],
        "modules": [
            _rename_dict_keys_for_yaml(mod, "module")
            for mod in api_collection["modules"]
        ]
    }

    # Wrap with package-api root key
    root_key = f"{package_name}-api"
    wrapped_data = {root_key: transformed_collection}

    # Add custom representer for multiline strings
    yaml.add_representer(str, _represent_str)

    try:
        return yaml.dump(
            wrapped_data,
            default_flow_style=False,
            sort_keys=True,
            indent=2,
            allow_unicode=True
        )
    finally:
        # Reset the representer to avoid affecting other uses of yaml
        yaml.add_representer(str, yaml.representer.SafeRepresenter.represent_str)


def main() -> None:
    """Main entry point for apeyedoc."""
    parser = argparse.ArgumentParser(
        description="Extract and document the public API of Python packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s colour                      # Analyze colour-science package
  %(prog)s bgand256 -v                 # Analyze bgand256 with verbose output
  %(prog)s json -f json                # Analyze json package in JSON format
  %(prog)s numpy -f yaml               # Analyze numpy package in YAML format
  %(prog)s scipy -o api.txt            # Save scipy API to text file
  %(prog)s requests -f json -o api.json # Save requests API as JSON
  %(prog)s flask -f yaml -o api.yaml   # Save flask API as YAML
  %(prog)s django -x docstrings        # Analyze django without docstrings
  %(prog)s numpy -x signatures -x docstrings # Analyze numpy without both
  %(prog)s scipy -x classes -x methods # Analyze scipy without classes and methods
        """
    )

    parser.add_argument(
        "package",
        nargs="?",
        default="colour",
        help="Package name to analyze (default: colour)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output during analysis"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Specify output file (default: stdout)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Specify output format (default: text)"
    )

    parser.add_argument(
        "-x", "--exclude",
        action="append",
        choices=[
            "docstrings",
            "signatures",
            "classes",
            "functions",
            "methods",
            "enums",
            "constants",
            "variables",
            "modules",
            "members",
        ],
        default=[],
        help="Exclude specific content types from the output (repeatable)"
    )

    args = parser.parse_args()

    # Suppress warnings unless verbose mode is enabled
    if not args.verbose:
        suppress_warnings()

    # Collect API information
    try:
        api_collection = collect_api_items(
            package_name=args.package,
            verbose=args.verbose,
            exclusions=args.exclude
        )
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        total_functions = len(api_collection["functions"])
        total_classes = len(api_collection["classes"])
        total_enums = len(api_collection["enums"])
        total_constants = len(api_collection["constants"])
        total_variables = len(api_collection["variables"])
        total_modules = len(api_collection["modules"])

        print(f"Found {total_functions} functions, {total_classes} classes, "
              f"{total_enums} enums, {total_constants} constants, "
              f"{total_variables} variables, {total_modules} modules")

    # Format output
    if args.format == "json":
        output_content = format_json_output(api_collection, args.package)
    elif args.format == "yaml":
        output_content = format_yaml_output(api_collection, args.package)
    else:
        output_content = format_text_output(api_collection)

    # Write output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            if args.verbose:
                print(f"Output written to: {args.output}")
        except OSError as e:
            print(f"Error writing to file {args.output}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_content)


if __name__ == "__main__":
    main()

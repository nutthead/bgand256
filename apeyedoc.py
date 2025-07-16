#!/usr/bin/env python3
"""
apeyedoc.py - Extract and document the public API of the colour-science module

Author: Behrang Saeedzadeh <hello@behrang.org>
"""

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
import warnings
from typing import Any, NotRequired, TypedDict


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
    name: str
    docstring: str | None
    signature: NotRequired[str]


class MethodInfo(TypedDict):
    name: str
    docstring: str | None
    signature: NotRequired[str]


class ClassInfo(TypedDict):
    name: str
    docstring: str | None
    methods: NotRequired[list[MethodInfo]]


def collect_api_items(
    package_name: str = "colour", verbose: bool = False
) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """
    Collect all public functions and classes from the colour-science package.

    Args:
        package_name: Name of the package to analyze
        verbose: Whether to print verbose output during collection

    Returns:
        Tuple of (functions, classes) lists containing API information
    """
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []

    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"Error: Could not import {package_name}: {e}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Analyzing package: {package_name}")

    # Walk through all modules in the package
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + '.'
    ):
        # Skip test modules and private modules
        if 'test' in modname or '._' in modname:
            continue

        if verbose:
            print(f"Examining module: {modname}")

        try:
            module = importlib.import_module(modname)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not import {modname}: {e}")
            continue

        # Get public names from module
        if hasattr(module, '__all__'):
            public_names = module.__all__
        else:
            public_names = [name for name in dir(module) if is_public_name(name)]

        for name in public_names:
            try:
                obj = getattr(module, name)
            except AttributeError:
                if verbose:
                    print(f"Warning: {modname} has no attribute {name}")
                continue

            qualified_name = get_qualified_name(modname, name)
            docstring = extract_docstring(obj)

            if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                # Handle functions
                signature = get_type_signature(obj)
                function_info: FunctionInfo = {
                    "name": qualified_name,
                    "docstring": docstring
                }
                if signature:
                    function_info["signature"] = signature

                functions.append(function_info)

            elif inspect.isclass(obj):
                # Handle classes
                class_info: ClassInfo = {
                    "name": qualified_name,
                    "docstring": docstring
                }

                # Collect public methods
                methods: list[MethodInfo] = []
                for method_name in dir(obj):
                    if is_public_name(method_name):
                        try:
                            method = getattr(obj, method_name)
                            if inspect.ismethod(method) or inspect.isfunction(method):
                                method_signature = get_type_signature(method)
                                method_doc = extract_docstring(method)
                                method_info: MethodInfo = {
                                    "name": method_name,
                                    "docstring": method_doc
                                }
                                if method_signature:
                                    method_info["signature"] = method_signature
                                methods.append(method_info)
                        except Exception:
                            continue

                if methods:
                    class_info["methods"] = methods

                classes.append(class_info)

    return functions, classes


def format_text_output(functions: list[FunctionInfo], classes: list[ClassInfo]) -> str:
    """Format API information as text."""
    output_lines: list[str] = []

    # Output functions
    if functions:
        output_lines.append("=== FUNCTIONS ===\n")
        for func in sorted(functions, key=lambda x: x["name"]):
            name = func["name"]
            signature = func.get("signature", "")
            if signature:
                output_lines.append(f"Function: {name}{signature}")
            else:
                output_lines.append(f"Function: {name}")

            docstring = func.get("docstring")
            if docstring:
                output_lines.append(f"DocString: {docstring}")
            else:
                output_lines.append("DocString: <no documentation available>")

            output_lines.append("")

    # Output classes
    if classes:
        output_lines.append("=== CLASSES ===\n")
        for cls in sorted(classes, key=lambda x: x["name"]):
            name = cls["name"]
            output_lines.append(f"Class: {name}")

            docstring = cls.get("docstring")
            if docstring:
                output_lines.append(f"DocString: {docstring}")
            else:
                output_lines.append("DocString: <no documentation available>")

            # Include methods if available
            methods = cls.get("methods", [])
            if methods:
                output_lines.append("Methods:")
                for method in sorted(methods, key=lambda x: x["name"]):
                    method_name = method["name"]
                    method_signature = method.get("signature", "")
                    if method_signature:
                        output_lines.append(f"  - {method_name}{method_signature}")
                    else:
                        output_lines.append(f"  - {method_name}")

                    method_doc = method.get("docstring")
                    if method_doc:
                        # Truncate long method docstrings
                        if len(method_doc) > 100:
                            method_doc = method_doc[:100] + "..."
                        output_lines.append(f"    {method_doc}")

            output_lines.append("")

    return "\n".join(output_lines)


def format_json_output(functions: list[FunctionInfo], classes: list[ClassInfo]) -> str:
    """Format API information as JSON."""
    output = {
        "functions": functions,
        "classes": classes
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def main() -> None:
    """Main entry point for apeyedoc."""
    parser = argparse.ArgumentParser(
        description="Extract and document the public API of the colour-science module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Output to stdout in text format
  %(prog)s -v                        # Verbose output
  %(prog)s -f json                   # Output in JSON format
  %(prog)s -o api.txt                # Save to file
  %(prog)s -f json -o api.json       # Save JSON to file
        """
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
        choices=["text", "json"],
        default="text",
        help="Specify output format (default: text)"
    )

    args = parser.parse_args()

    # Suppress warnings unless verbose mode is enabled
    if not args.verbose:
        suppress_warnings()

    # Collect API information
    try:
        functions, classes = collect_api_items(verbose=args.verbose)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(functions)} functions and {len(classes)} classes")

    # Format output
    if args.format == "json":
        output_content = format_json_output(functions, classes)
    else:
        output_content = format_text_output(functions, classes)

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

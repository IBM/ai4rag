#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import ast
import inspect
from typing import Any, Callable

# pylint: disable=invalid-name


def _get_default_args(func: Callable) -> dict[str, Any]:
    """Get a mapping that stores func parameters as keys and their default values as values.

    Parameters
    ----------
    func : Callable
        The function that is checked.

    Returns
    -------
    dict[str. Any]
        Dictionary representing function parameters and corresponding default values.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


class FunctionTransformer(ast.NodeTransformer):
    """Class for injecting data into function

    Parameters
    ----------
    function : ast.FunctionDef
        Function to be transformed.
    """

    def __init__(self, function: ast.FunctionDef, **kwargs: Any) -> None:
        self.args_to_replace = kwargs
        self.function_args = [arg.arg for arg in function.args.args]

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Visits node of type Call and injects values if needed

        Parameters
        ----------
        node : ast.Call
            Call node.

        Returns
        -------
        ast.AST
            The same node after injecting missing values or left as is.
        """
        new_keywords: list[ast.keyword] = []
        for keyword in node.keywords:
            new_keyword = None
            if hasattr(keyword.value, "id") and keyword.value.id in self.args_to_replace:
                v: dict = self.args_to_replace.get(keyword.value.id, {})
                replace = v.get("replace")
                value = v.get("value")
                if replace:
                    if isinstance(value, dict):
                        new_keyword = ast.keyword(
                            arg=keyword.arg,
                            value=ast.parse(str(value), filename="tmp", mode="exec")
                            .body[0]
                            .value,  # type: ignore[attr-defined]
                        )
                    else:
                        new_keyword = ast.keyword(arg=keyword.arg, value=ast.Constant(value=value))
            else:
                new_keyword = keyword

            if new_keyword and new_keyword.arg not in [el.arg for el in new_keywords]:
                new_keywords.append(new_keyword)

        node.keywords = new_keywords

        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_node":
            for arg in node.args:
                if isinstance(arg, ast.Name) and (new_id := self.args_to_replace.get(arg.id)):
                    arg.id = new_id

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visits node of type Function definition and injects values if needed

        Parameters
        ----------
        node : ast.Assign
            Object representing the function definition node.

        Returns
        -------
        ast.AST | ast.Assign
            The same node after injecting missing values or left as is.
        """
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Name):
                if stmt.value.id.startswith("REPLACE_THIS_CODE_WITH"):
                    replacement_code: str = self.args_to_replace.get(stmt.value.id, "")
                    replacement_ast = ast.parse(replacement_code).body
                    new_body.extend(replacement_ast)
                    continue
            new_body.append(stmt)

        node.body = new_body
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        """Visits node of type Import and injects values if needed

        Parameters
        ----------
        node : ast.Assign
            Object representing the import node.

        Returns
        -------
        ast.AST | ast.Assign
            The same node after injecting missing values or left as is.
        """
        for alias in node.names:
            if alias.name in self.args_to_replace:
                alias.name = self.args_to_replace.get(alias.name)
        return self.generic_visit(node)

    # pylint: disable=no-else-return
    def visit_Assign(self, node: ast.Assign) -> ast.AST | ast.Assign:
        """Visits node of type Assign and injects values if needed

        Parameters
        ----------
        node : ast.Assign
            Object representing the Assign node.

        Returns
        -------
        ast.AST | ast.Assign
            The same node after injecting missing values or left as is.
        """
        for target in node.targets:
            if hasattr(node.value, "id") and node.value.id in self.args_to_replace:
                if node.value.id.endswith("input_data_references".upper()):
                    return ast.Assign(
                        targets=[target],
                        value=ast.List(
                            [
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="DataConnection", ctx=ast.Load()),
                                        attr="from_dict",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.parse(
                                            str(el),
                                            filename="tmp",
                                            mode="exec",
                                        )
                                        .body[0]
                                        .value
                                    ],  # type: ignore[attr-defined]
                                    keywords=[],
                                )
                                for el in self.args_to_replace[node.value.id]
                            ],
                            ctx=ast.Load(),
                        ),
                        lineno=target.lineno,
                    )

                if node.value.id.endswith("vector_stores_initialization".upper()):
                    return ast.Assign(
                        targets=[target],
                        value=ast.List(
                            [
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id=el.get("class"), ctx=ast.Load()),
                                        attr="from_dict",
                                        ctx=ast.Load(),
                                    ),
                                    args=[],
                                    keywords=[
                                        ast.keyword(
                                            arg=el.get("client_kw"), value=ast.Name(id="client", ctx=ast.Load())
                                        ),
                                        ast.keyword(
                                            arg="data",
                                            value=ast.Subscript(
                                                value=ast.Name(id="vector_store_init_data", ctx=ast.Load()),
                                                slice=ast.Constant(value=el.get("index")),
                                                ctx=ast.Load(),
                                            ),
                                        ),
                                    ],
                                )
                                for el in self.args_to_replace[node.value.id]
                            ],
                            ctx=ast.Load(),
                        ),
                        lineno=target.lineno,
                    )

                else:
                    new_value = self.args_to_replace[node.value.id]
                    if isinstance(new_value, (dict, list)):
                        value = (
                            ast.parse(str(new_value), filename="tmp", mode="exec")
                            .body[0]
                            .value  # type: ignore[attr-defined]
                        )
                    else:
                        value = ast.Constant(value=new_value)
                    return ast.Assign(
                        targets=[target],
                        value=value,
                        lineno=target.lineno,
                    )
        return self.generic_visit(node)


class FunctionVisitor(ast.NodeVisitor):
    """Class for mapping function definition into nodes"""

    def __init__(self) -> None:
        self.function: ast.FunctionDef | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Helper fo visits node of type functionDef.

        Parameters
        ----------
        node : ast.FunctionDef
            Function def node.
        """
        self.function = node
        self.generic_visit(node)


def _get_components_replace_data(init_params: dict, func: Callable, suffix: str) -> dict:
    """Helper for selecting minimum set of parameters from `init_params` object
    that need to be passed to func to assure proper and unique call.

    Parameters
    ----------
    init_params : dict
        The set of parameters to be passed to the func object.

    func : Callable
        function object whose parameters are analyzed.

    suffix : str
        Suffix added after `REPLACE_THIS_CODE_WITH_` to distinguish
                   between parameter groups to be replaced in template

    Returns
    -------
    dict
        A set of func parameters with information whether they are to be passed
        and under what name they can be located in the template. Sample record
            {
                f'REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}': {
                    "value": init_params.get(arg),
                    "replace": False,
                }
            }
    """
    new_model_init_params = {}
    init_params_defaults = _get_default_args(func)
    for arg, value in init_params_defaults.copy().items():
        if arg in init_params and value == init_params[arg]:
            new_model_init_params.update(
                {
                    f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                        "value": init_params.get(arg),
                        "replace": False,
                    }
                }
            )
        elif arg in init_params:
            if arg.upper() == "URL" and suffix.upper() == "CREDENTIALS":
                new_model_init_params.update(
                    {f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": init_params.get(arg)}
                )
            else:
                new_model_init_params.update(
                    {
                        f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                            "value": init_params.get(arg),
                            "replace": True,
                        }
                    }
                )
        else:
            new_model_init_params.update(
                {
                    f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                        "value": init_params.get(arg),
                        "replace": False,
                    }
                }
            )

    return new_model_init_params

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


class StrategyValidationError(Exception):
    pass


@dataclass(frozen=True)
class ValidationReport:
    path: str
    class_name: str
    passed: bool


class _StrategyAstValidator(ast.NodeVisitor):
    ALLOWED_IMPORTS = {"math", "statistics", "typing", "ashare_backtest.protocol"}
    BANNED_CALLS = {"eval", "exec", "open", "compile", "__import__", "input", "globals", "locals"}
    BANNED_NODES = (
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Lambda,
        ast.While,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )

    def __init__(self) -> None:
        self.strategy_classes: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self.BANNED_NODES):
            raise StrategyValidationError(f"banned syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in self.ALLOWED_IMPORTS:
                raise StrategyValidationError(f"import not allowed: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if module not in self.ALLOWED_IMPORTS:
            raise StrategyValidationError(f"from import not allowed: {module}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.BANNED_CALLS:
            raise StrategyValidationError(f"call not allowed: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise StrategyValidationError("dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        base_names = {self._resolve_base_name(base) for base in node.bases}
        if "BaseStrategy" in base_names:
            self.strategy_classes.append(node.name)
        self.generic_visit(node)

    @staticmethod
    def _resolve_base_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""


class StrategyValidator:
    REQUIRED_METHODS = {"rebalance", "select", "allocate"}

    def validate_file(self, path: str | Path) -> ValidationReport:
        target = Path(path)
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
        ast_validator = _StrategyAstValidator()
        ast_validator.visit(tree)

        if len(ast_validator.strategy_classes) != 1:
            raise StrategyValidationError("strategy script must define exactly one BaseStrategy subclass")

        class_name = ast_validator.strategy_classes[0]
        class_node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
        missing = self.REQUIRED_METHODS - method_names
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise StrategyValidationError(f"missing required methods: {missing_text}")

        return ValidationReport(path=str(target), class_name=class_name, passed=True)

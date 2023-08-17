"""Primary interactive API for manipulating artifacts."""

from typing import Any, Dict, Union
from pathlib import Path
import re
import time

from iree.compiler.api import (
    Session,
    Invocation,
    Source,
    Output,
)

from iree.compiler.ir import (
    Block,
    Context,
    Location,
    Module,
    Operation,
    StringAttr,
    SymbolTable,
)

from . import merge_utils

__all__ = [
    "InputModule",
    "Workspace",
]


class Workspace:
    """Workspace for artifacts."""

    def __init__(self):
        self.session = Session()
        self.context = self.session.context
        self.inputs: Dict[str, "InputModule"] = AttrDict()
        self.outputs: Dict[str, "OutputModule"] = AttrDict()

    def open_input(
        self, path: Union[str, Path], ident: str = "input0"
    ) -> "InputModule":
        inv = self.session.invocation()
        ident = self.inputs._reserve(ident)
        t = report_start(f"Opening file {path} as {ident}...")
        try:
            source = Source.open_file(self.session, str(path))
            if not inv.parse_source(source):
                raise RuntimeError(f"see diagnostics")
            module = inv.export_module()
            input = InputModule(self, ident, inv, module)
            self.inputs[ident] = input
            return input
        except Exception as e:
            report_end(f"ERROR: {e}")
            raise e
        finally:
            report_end(f" complete in {t.elapsed}")

    def create_empty(self, ident: str = "output0") -> "OutputModule":
        inv = self.session.invocation()
        ident = self.outputs._reserve(ident)
        with Location.unknown(self.session.context):
            module = Module.create().operation
        inv.import_module(module)
        output = OutputModule(self, ident, inv, module)
        self.outputs[ident] = output
        return output

    def _resolve_output(self, output: Union[str, "OutputModule"]) -> "OutputModule":
        if isinstance(output, OutputModule):
            return output
        try:
            return self.outputs[output]
        except KeyError:
            raise ValueError(f"Output '{output}' is unknown")


class WorkspaceModule:
    """Base class for input and output modules."""

    def __init__(
        self, workspace: Workspace, ident: str, inv: Invocation, module: Operation
    ):
        self.workspace = workspace
        self.ident = ident
        self.inv = inv
        self.module = module

    @property
    def body(self) -> Block:
        return self.module.regions[0].blocks[0]

    @property
    def public_functions(self) -> Dict[str, "FunctionInfo"]:
        results = {}
        for op_view in self.body:
            op = op_view.operation
            op_name = op.name
            if op_name not in ["func.func"]:
                continue
            try:
                vis = str(SymbolTable.get_visibility(op))
                print(vis)
            except ValueError:
                vis = "public"
            if vis != "public":
                continue
            func_name = StringAttr(SymbolTable.get_symbol_name(op)).value
            results[func_name] = FunctionInfo(op)
        return results

    def merge_to(self, output: Union[str, "OutputModule"], symbol_map: Dict[str, str]):
        """Destructively merges this module into the given OutputModule."""
        output = self.workspace._resolve_output(output)
        merger = merge_utils.Merger(
            self.module, output.module, symbol_map, logger=report
        )
        merger.merge()
        output.module.verify()


class InputModule(WorkspaceModule):
    """An input module that we are operating from."""

    def __init__(
        self, workspace: Workspace, ident: str, inv: Invocation, module: Operation
    ):
        super().__init__(workspace, ident, inv, module)
        self.transforms = InputModuleTransforms(self)

    def __repr__(self):
        return f"InputModule({self.ident})"


class OutputModule(WorkspaceModule):
    """A module under construction."""

    def __init__(
        self, workspace: Workspace, ident: str, inv: Invocation, module: Operation
    ):
        super().__init__(workspace, ident, inv, module)

    def __repr__(self):
        return f"OutputModule({self.ident})"


class InputModuleTransforms:
    def __init__(self, input: InputModule):
        self.input = input

    def normalize_constants(self):
        """Normalizes any eligible constants in the program to globals.

        In general, most optimizations for constants are keyed off of them
        being defined as module-level globals vs inline constants. Applying
        this transform will create globals out of any eligible inline
        constants.
        """
        # We first make sure to legalize any foreign dialect globals.
        self.input.inv.execute_text_pass_pipeline(
            "iree-import-public, iree-import-ml-program, iree-util-outline-constants, symbol-dce"
        )


class FunctionInfo:
    """Wraps a function operation and provides ergonomics."""

    def __init__(self, op: Operation):
        self.op = op

    def __repr__(self):
        return f"Function(@{SymbolTable.get_symbol_name(self.op)})"


class AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"Key not found '{key}'", name=key, obj=self)

    def _reserve(self, requested: str) -> str:
        if requested not in self:
            return requested
        stem = requested
        index = 1
        m = re.match(r"^(.+)([0-9]+)$", requested)
        if m:
            stem = m.group(1)
            index = int(m.group(2))
        while True:
            requested = f"{stem}{index}"
            if requested not in self:
                return requested
            index += 1


class Timer:
    def __init__(self):
        self.start_time = time.time()

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    @property
    def elapsed(self) -> str:
        t = self.elapsed_s
        if t > 1.0:
            t = int(t * 1000.0) / 1000.0
            return f"{t}s"
        if t >= 0.001:
            return f"{int(t * 1000)}ms"
        if t >= 0.000001:
            return f"{int(t * 1000000)}us"
        return f"{t}s"


def report(message: str):
    print(":", message)


def report_start(message: str) -> Timer:
    print(":", message, flush=True, end="")
    return Timer()


def report_end(message: str):
    print(message)

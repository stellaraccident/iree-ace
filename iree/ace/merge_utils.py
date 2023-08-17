from typing import Any, Dict, List, Sequence, Union

from iree.compiler.ir import (
    Attribute,
    Block,
    Context,
    Location,
    Module,
    Operation,
    StringAttr,
    SymbolTable,
)


def null_logger(msg):
    pass


def get_top_level_ops(module_op: Operation, *op_names: str) -> Sequence[Operation]:
    results = []
    for op_view in module_op.regions[0].blocks[0]:
        op = op_view.operation
        if op.name in op_names:
            results.append(op)
    return results


def is_global_immutable_initialized(global_op: Operation):
    return (
        "is_mutable" not in global_op.attributes
        and "initial_value" in global_op.attributes
    )


def uniqueify_name(local_name: str, st: SymbolTable) -> str:
    index = -1
    while True:
        index += 1
        full_name = local_name
        if index > 0:
            full_name += f"${index}"
        if full_name not in st:
            return full_name


class Merger:
    def __init__(
        self,
        source_module: Operation,
        target_module: Operation,
        user_rename_map: Dict[str, str],
        *,
        logger=None,
    ):
        self.context = source_module.context
        self.source_module = source_module
        self.target_module = target_module
        self.user_rename_map = user_rename_map
        self.logger = logger if logger else null_logger
        self.source_symbol_table = SymbolTable(self.source_module)
        self.target_symbol_table = SymbolTable(self.target_module)
        self.rename_map: Dict[StringAttr, StringAttr] = {}

        self.nested_symbol_ops: List[Operation] = []
        self.nested_symbol_table_ops: List[Operation] = []

        # Map of value attributes to global operation.
        self.initialized_globals: Dict[Attribute, Operation] = {}
        target_globals = get_top_level_ops(self.target_module, "util.global")
        for global_op in target_globals:
            if not is_global_immutable_initialized(global_op):
                continue
            self.initialized_globals[global_op.attributes["initial_value"]] = global_op

    @property
    def target_body(self) -> Block:
        return self.target_module.regions[0].blocks[0]

    def merge(self):
        # Merge globals.
        source_globals = get_top_level_ops(self.source_module, "util.global")
        for global_op in source_globals:
            if not is_global_immutable_initialized(global_op):
                self.import_symbol_op(global_op)
                continue
            global_value = global_op.attributes["initial_value"]
            alias_global_op = self.initialized_globals.get(global_value)
            if alias_global_op:
                # Don't import the global, just note the rename.
                alias_from = SymbolTable.get_symbol_name(global_op)
                alias_to = SymbolTable.get_symbol_name(alias_global_op)
                self.logger(
                    f"Aliasing imported global {StringAttr(alias_from).value} -> {StringAttr(alias_to).value}"
                )
                self._rename(alias_from, alias_to)
            else:
                # Import the global.
                self.import_symbol_op(global_op)

        # Merge initializers.
        initializers = get_top_level_ops(self.source_module, "util.initializer")
        for init_op in initializers:
            init_op.detach_from_parent()
            self.nested_symbol_table_ops.append(init_op)
            self.target_body.append(init_op)

        # Merge functions.
        funcs = get_top_level_ops(self.source_module, "func.func")
        for func_op in funcs:
            self.import_symbol_op(func_op)
            self.nested_symbol_table_ops.append(func_op)

        self.logger(f"The following symbol renames will be made: {self.rename_map}")

        # Go back through to nested symbol table ops and RAUW.
        for sym_operation in self.nested_symbol_table_ops:
            for from_symbol, to_symbol in self.rename_map.items():
                from_name = StringAttr(from_symbol).value
                to_name = StringAttr(to_symbol).value
                SymbolTable.replace_all_symbol_uses(from_name, to_name, sym_operation)

    def import_symbol_op(self, symbol_op):
        symbol_op = symbol_op.detach_from_parent()
        orig_symbol = SymbolTable.get_symbol_name(symbol_op)
        orig_symbol_name = StringAttr(orig_symbol).value
        requested_symbol = self.user_rename_map.get(orig_symbol_name)
        if requested_symbol:
            # Has a user mapping.
            if requested_symbol in self.target_symbol_table:
                raise ValueError(
                    f"Requested symbol rename {requested_symbol} exists in the target"
                )
            self.logger(f"Requested rename {orig_symbol_name} -> {requested_symbol}")
            SymbolTable.set_symbol_name(symbol_op, requested_symbol)
            self._rename(orig_symbol, requested_symbol)
        else:
            # No user mapping - make sure it is unique.
            new_symbol_name = uniqueify_name(orig_symbol_name, self.target_symbol_table)
            if new_symbol_name != orig_symbol_name:
                self.logger(
                    f"Implicit rename of conflicting symbol: {orig_symbol_name} -> {new_symbol_name}"
                )
                SymbolTable.set_symbol_name(symbol_op, new_symbol_name)
                self._rename(orig_symbol, new_symbol_name)

        self.target_body.append(symbol_op)
        self.nested_symbol_ops.append(symbol_op)
        self.target_symbol_table.insert(symbol_op)

    def _rename(self, from_symbol, to_symbol):
        from_symbol = self._make_string_attr(from_symbol)
        to_symbol = self._make_string_attr(to_symbol)
        if from_symbol != to_symbol:
            self.rename_map[from_symbol] = to_symbol

    def _make_string_attr(self, string_attr_or_str):
        if isinstance(string_attr_or_str, str):
            with self.context:
                return StringAttr.get(string_attr_or_str)
        else:
            return StringAttr(string_attr_or_str)

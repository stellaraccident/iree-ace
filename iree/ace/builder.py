from typing import Any, Dict, List, Sequence, Union

from iree.compiler.ir import (
    Block,
    Context,
    FlatSymbolRefAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    Operation,
    OpView,
    StringAttr,
    SymbolTable,
    Type,
    TypeAttr,
    UnitAttr,
    Value,
)


class TensorSliceOp(OpView):
    OPERATION_NAME = "flow.tensor.slice"
    _ODS_OPERAND_SEGMENTS = [1, -1, -1, -1, -1]


class TensorUpdateOp(OpView):
    OPERATION_NAME = "flow.tensor.update"
    _ODS_OPERAND_SEGMENTS = [1, -1, -1, 1, -1]


class Builder:
    def __init__(self, module_op: Operation):
        self.module_op = module_op
        self.st = SymbolTable(self.module_op)
        self.loc = Location.unknown(self.module_op.context)
        self.body = self.module_op.regions[0].blocks[0]

    def integer_type(self, bitwidth: int) -> IntegerType:
        with self.loc:
            return IntegerType.get_signless(bitwidth)

    def define_global(self, name: str, type: Type, *, mutable: bool) -> Operation:
        with self.loc, InsertionPoint.at_block_begin(self.body):
            attrs = {
                "sym_name": StringAttr.get(name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(type),
            }
            if mutable:
                attrs["is_mutable"] = UnitAttr.get()
            global_op = Operation.create("util.global", attributes=attrs)
            self.st.insert(global_op)
        return global_op

    def define_function(
        self,
        name: str,
        input_types: Sequence[Type],
        result_types=Sequence[Type],
        *,
        public: bool = False
    ) -> "FunctionBuilder":
        with self.loc, InsertionPoint(self.body):
            ftype = FunctionType.get(input_types, result_types)
            attrs = {
                "sym_name": StringAttr.get(name),
                "function_type": TypeAttr.get(ftype),
            }
            if not public:
                attrs["sym_visibility"] = StringAttr.get("private")
            f_op = Operation.create("func.func", attributes=attrs, regions=1)
            self.st.insert(f_op)
        return FunctionBuilder(f_op, input_types, result_types)


class FunctionBuilder:
    def __init__(
        self, f_op: Operation, input_types: Sequence[Type], result_types: Sequence[Type]
    ):
        self.f_op = f_op
        self.loc = Location.unknown(self.f_op.context)
        with self.loc:
            self.body = f_op.regions[0].blocks.append(*input_types)
        self.ip = InsertionPoint(self.body)

    @property
    def arguments(self):
        return self.body.arguments

    def addi_imm(self, input: Value, imm: int) -> Value:
        with self.ip, self.loc:
            t = input.type
            imm_value = Operation.create(
                "arith.constant",
                results=[t],
                attributes={"value": IntegerAttr.get(t, imm)},
            ).result
            return Operation.create(
                "arith.addi",
                results=[input.type],
                operands=[input, imm_value],
            ).result

    def cast_to_index(self, input: Value) -> Value:
        with self.ip, self.loc:
            return Operation.create(
                "arith.index_cast", results=[IndexType.get()], operands=[input]
            ).result

    def constant_index(self, value: int) -> Value:
        with self.ip, self.loc:
            index_type = IndexType.get()
            return Operation.create(
                "arith.constant",
                results=[index_type],
                attributes={"value": IntegerAttr.get(index_type, value)},
            ).result

    def constant_int(self, value: int, bitwidth: int) -> Value:
        with self.ip, self.loc:
            int_type = IntegerType.get_signless(bitwidth)
            return Operation.create(
                "arith.constant",
                results=[int_type],
                attributes={"value": IntegerAttr.get(int_type, value)},
            ).result

    def load_global(self, global_op: Operation) -> Value:
        sym_name = global_op.attributes["sym_name"]
        t = TypeAttr(global_op.attributes["type"]).value
        with self.ip, self.loc:
            attrs = {
                "global": FlatSymbolRefAttr.get(StringAttr(sym_name).value),
            }
            return Operation.create(
                "util.global.load", results=[t], attributes=attrs
            ).result

    def store_global(self, global_op: Operation, update: Value):
        sym_name = global_op.attributes["sym_name"]
        with self.ip, self.loc:
            attrs = {
                "global": FlatSymbolRefAttr.get(StringAttr(sym_name).value),
            }
            Operation.create(
                "util.global.store", results=[], operands=[update], attributes=attrs
            )

    def call(self, callee_op: Operation, *operands: Value) -> Sequence[Value]:
        sym_name = callee_op.attributes["sym_name"]
        ftype = FunctionType(TypeAttr(callee_op.attributes["function_type"]).value)
        with self.ip, self.loc:
            attrs = {
                "callee": FlatSymbolRefAttr.get(StringAttr(sym_name).value),
            }
            return Operation.create(
                "func.call",
                results=ftype.results,
                operands=operands,
                attributes=attrs,
            ).results

    def tensor_dim(self, input: Value, dim: Value) -> Value:
        with self.ip, self.loc:
            return Operation.create(
                "tensor.dim", results=[IndexType.get()], operands=[input, dim]
            ).result

    def ret(self, *values: Value):
        with self.ip, self.loc:
            Operation.create("func.return", operands=values)

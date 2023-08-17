from iree.ace import *
from iree.ace.builder import TensorSliceOp, TensorUpdateOp
from iree.compiler.ir import (
    IndexType,
    Location,
    Operation,
    RankedTensorType,
    Type,
    Value,
)

second_file = (
    "/home/stella/tmp/vicuna/vicuna_unsharded_mlir_second_vicuna_int4_stripped.mlir"
)

ws = Workspace()
first = ws.open_input(second_file, "first")
second = ws.open_input(second_file, "second")
out = ws.create_empty()
# ws.open_input(second_file, "second")

print("Public functions:", second.public_functions)

print(second)
second.transforms.normalize_constants()

print("MERGE 1:")
first.merge_to(
    "output0",
    {
        "forward": "step_impl",
    },
)

step_f = out.functions["step_impl"]
step_f.set_private()

step_input_types = step_f.input_types
step_result_types = step_f.result_types


# print("INPUTS:", step_input_types)
# print("RESULTS:", step_input_types)
# The state types have shapes like tensor<1x32x?x128xf32>
# We keep them in globals as fixed size allocations based on the model's
# context size.
def fixate_state_size(t, fixed_size: int):
    tt = RankedTensorType(t)
    shape = [-1 if tt.is_dynamic_dim(i) else tt.get_dim_size(i) for i in range(tt.rank)]
    assert shape[2] == -1, "expected dynamic state like 1x32x?x128"
    shape[2] = fixed_size
    with Location.unknown(tt.context):
        return RankedTensorType.get(shape, tt.element_type)


step_input_type = step_input_types[0]
step_result_type = step_result_types[0]
step_state_types = step_input_types[1:]
state_context_size = 4096  # Magic number for model.
global_state_types = [
    fixate_state_size(t, state_context_size) for t in step_state_types
]
print("GLOBAL TYPES:", global_state_types)

builder = out.builder
state_globals = [
    builder.define_global(f"_context_{i}", global_state_types[i], mutable=True)
    for i in range(len(global_state_types))
]
step_count_global = builder.define_global(
    f"_step_count", builder.integer_type(32), mutable=True
)


def define_step_function():
    def slice_state(result_type: Type, input: Value) -> Value:
        with fb.loc, fb.ip:
            return TensorSliceOp.build_generic(
                results=[result_type],
                operands=[
                    input,
                    [],  # source_dims
                    [zero, zero, zero, zero],  # start_indices
                    [
                        # TODO: Derive these from the type.
                        fb.constant_index(1),
                        fb.constant_index(32),
                        step_count,
                        fb.constant_index(128),
                    ],  # lengths
                    [step_count],  # result_dims
                ],
            ).result

    def update_slice(target: Value, update: Value) -> Value:
        with fb.loc, fb.ip:
            dim = fb.tensor_dim(update, fb.constant_index(2))
            return TensorUpdateOp.build_generic(
                results=[target.type],
                operands=[
                    target,
                    [],  # target_dims
                    [zero, zero, zero, zero],  # start_indices
                    update,
                    [dim],  # update_dims
                ],
            ).result

    fb = builder.define_function(
        "step",
        input_types=[step_input_type],
        result_types=[step_result_type],
        public=True,
    )
    zero = fb.constant_index(0)
    step_count_i32 = fb.load_global(step_count_global)
    step_count = fb.cast_to_index(step_count_i32)
    loaded_globals = [fb.load_global(g) for g in state_globals]
    sliced_globals = [
        slice_state(t, s) for s, t in zip(loaded_globals, step_state_types)
    ]
    result, *updates = fb.call(step_f.op, fb.arguments[0], *sliced_globals)
    updates = [
        update_slice(target, update) for target, update in zip(loaded_globals, updates)
    ]
    for update, global_op in zip(updates, state_globals):
        fb.store_global(global_op, update)

    # Increment step.
    next_step = fb.addi_imm(step_count_i32, 1)
    fb.store_global(step_count_global, next_step)
    fb.ret(result)


define_step_function()

# print("MERGE 2:")
# second.merge_to(
#     "output0",
#     {
#         "forward": "step",
#     },
# )

out.transforms.cse()

with open("/home/stella/tmp/fudge/vicuna_step_raw.mlir", "wt") as f:
    print(out.module, file=f)

out.transforms.inline()
out.transforms.cse()

with open("/home/stella/tmp/fudge/vicuna_step_inline.mlir", "wt") as f:
    print(out.module, file=f)

# print(second.module)

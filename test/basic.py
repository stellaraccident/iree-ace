from iree.ace import *

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
        "forward": "initialize",
    },
)

print("MERGE 2:")
second.merge_to(
    "output0",
    {
        "forward": "step",
    },
)

print(out.module)
# print(second.module)

from typing import Tuple, List

from e3nn import o3


# Based on mir-group/nequip
def tp_out_irreps_with_instructions(irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps) \
        -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_mid)  # instruction index
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, permut, _ = irreps_mid.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [(i_in1, i_in2, permut[i_out], mode, train) for i_in1, i_in2, i_out, mode, train in instructions]

    return irreps_mid, instructions


def linear_out_irreps(irreps: o3.Irreps, target_irreps: o3.Irreps) -> o3.Irreps:
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f'{ir_in} not in {target_irreps}')

    return o3.Irreps(irreps_mid)

from e3nn import o3


def node_edge_combined_irreps(node_irreps: o3.Irreps, edge_irreps: o3.Irreps, out_irreps: o3.Irreps) -> o3.Irreps:
    combined = []
    for i, (mul, ir_in) in enumerate(node_irreps):
        for j, (_, ir_edge) in enumerate(edge_irreps):
            for ir_out in ir_in * ir_edge:
                if ir_out in out_irreps:
                    combined.append((mul, ir_out))

    combined_irreps = o3.Irreps(combined)
    sorted_irreps, p, _ = combined_irreps.sort()

    return sorted_irreps

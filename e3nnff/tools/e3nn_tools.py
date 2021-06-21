from e3nn import o3


def tp_combine_irreps(node_irreps: o3.Irreps, edge_irreps: o3.Irreps, max_ell: int, num_channels: int) -> o3.Irreps:
    out_irreps = o3.FullTensorProduct(node_irreps, edge_irreps).irreps_out.simplify()
    new_irreps = [(num_channels, item.ir) for item in out_irreps if item.ir.l <= max_ell]  # noqa: E741
    combined_irreps = o3.Irreps(new_irreps)
    sorted_irreps, _, _ = combined_irreps.sort()
    return sorted_irreps.simplify()


def get_num_e0_channels(irreps: o3.Irreps) -> int:
    for channels, (ell, p) in irreps:
        if ell == 0 and p == 1:
            return channels

    raise RuntimeError(f'Could not find e0 irrep in {irreps}')

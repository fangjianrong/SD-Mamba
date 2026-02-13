import torch
import numpy as np

# original scans
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs

# partial scan 1
class CrossScan_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.flatten(2, 3).flip(dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # in: (B, 2, C, l)
        # out: (b, c, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1])
        return y.view(B, -1, H, W)

class CrossMerge_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1])
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.flip(dims=[-1])
        xs = xs.view(B, 2, C, H, W)
        return xs


# partial scan 2
def spiral_order(rows, columns):
    matrix = np.arange(rows * columns).reshape(rows, columns)
    result = []

    while matrix.size > 0:
        result.append(matrix[0, :])
        matrix = matrix[1:]
        matrix = np.rot90(matrix)

    result = np.hstack(result)
    index_x, index_y = np.unravel_index(result, (rows, columns))
    return index_x, index_y

# 根据索引恢复模块
def reconstruct_array(idx_x, idx_y, values, shape):
    device = values.device
    reconstructed = torch.zeros(shape, dtype=values.dtype, device=device)
    reconstructed[:,:,idx_x, idx_y] = values
    return reconstructed

class CrossScan_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        index_x, index_y = spiral_order(H, W)
        ctx.list = [(B, C, H, W),index_x, index_y]

        xs = x.new_empty((B, 2, C, H * W))

        xs[:, 0] = x[:,:,index_x, index_y]
        xs[:, 1] = x[:,:,index_y, index_x]
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        (B, C, H, W),index_x, index_y = ctx.list

        y = reconstruct_array(index_x, index_y, ys[:, 0], (B, C, H, W))+\
            reconstruct_array(index_y, index_x, ys[:, 1], (B, C, H, W))
        return y

class CrossMerge_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        index_x, index_y = spiral_order(H, W)
        ctx.list = [(H, W), index_x, index_y]

        ys = ys.view(B, K, D, -1)

        y = reconstruct_array(index_x, index_y, ys[:, 0], (B, D, H, W))+\
            reconstruct_array(index_y, index_x, ys[:, 1], (B, D, H, W))
        return y.view(B, D, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        (H, W), index_x, index_y = ctx.list
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x.view(B, C, H, W)[:,:,index_x, index_y]
        xs[:, 1] = x.view(B, C, H, W)[:,:,index_y, index_x]
        xs = xs.view(B, 2, C, H, W)
        return xs


# partial scan 3
class CrossScan_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor): # B*H*W, self.channel_num, group_channel_num, 1
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.reshape(B,-1).flip(dims=[-1]).reshape(B,C,-1)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor): # ys实则xs
        # in: (B, 2, C, l)
        # out: (b, c, d, l)
        B, C, H, W = ctx.shape
        y = ys[:, 0] + ys[:, 1].reshape(B,-1).flip(dims=[-1]).reshape(B,C,-1)
        return y.view(B, -1, H, W)


class CrossMerge_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].reshape(B,-1).flip(dims=[-1]).reshape(B,D,-1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.reshape(B,-1).flip(dims=[-1]).reshape(B,C,-1)
        xs = xs.view(B, 2, C, H, W)
        return xs

# partial scan 4
def antidiagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)

        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)

        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb; pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops




# ==============================================================
# Dynamic Semantic-Aware Scanning (Innovation for TGRS)
# ==============================================================

class DynamicCenterScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, sort_idx: torch.Tensor):
        """
        x: (B, C, H, W)
        sort_idx: (B, L) - Indices sorted by similarity to center pixel
        """
        B, C, H, W = x.shape
        L = H * W
        ctx.shape = (B, C, H, W)
        ctx.save_for_backward(sort_idx)

        # 1. Flatten spatial dimensions
        x_flat = x.view(B, C, L)

        # 2. Expand indices for gathering: (B, L) -> (B, C, L)
        sort_idx_expand = sort_idx.unsqueeze(1).expand(-1, C, -1)

        # 3. Gather: Reorder pixels based on similarity (High -> Low)
        # xs_0 represents the sequence from "Most Similar" to "Least Similar"
        xs_0 = torch.gather(x_flat, 2, sort_idx_expand)

        # 4. Create reverse direction (Low -> High) for bidirectional modeling
        xs_1 = torch.flip(xs_0, dims=[-1])

        # 5. Stack to match Mamba input format: (B, K=2, C, L)
        xs = torch.stack([xs_0, xs_1], dim=1)
        
        return xs

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, 2, C, L)
        sort_idx, = ctx.saved_tensors
        B, C, H, W = ctx.shape
        L = H * W

        # 1. Decompose gradients from both directions
        grad_0 = grad_output[:, 0]
        grad_1 = torch.flip(grad_output[:, 1], dims=[-1])
        
        # 2. Sum gradients (similar to CrossScan_2 logic)
        grad_sorted = grad_0 + grad_1 # (B, C, L) in sorted order

        # 3. Scatter gradients back to original spatial positions
        sort_idx_expand = sort_idx.unsqueeze(1).expand(-1, C, -1)
        
        # Create empty tensor for original shape
        grad_x = torch.zeros(B, C, L, device=grad_output.device, dtype=grad_output.dtype)
        
        # Scatter puts values from `grad_sorted` back into `grad_x` at `sort_idx` positions
        grad_x.scatter_(2, sort_idx_expand, grad_sorted)

        return grad_x.view(B, C, H, W), None


class DynamicCenterMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, sort_idx: torch.Tensor):
        # ys: (B, K, D, H, W) -> standard input shape from some Mamba implementations
        # sort_idx: (B, L)
        B, K, D, H, W = ys.shape
        L = H * W
        ctx.hw = (H, W)
        ctx.save_for_backward(sort_idx)

        # 1. Flatten: (B, K, D, L)
        ys = ys.view(B, K, D, -1)

        # 2. Merge branches
        y_0 = ys[:, 0] # High -> Low
        y_1 = torch.flip(ys[:, 1], dims=[-1]) # Low -> High reversed back
        
        y_combined = y_0 + y_1 # (B, D, L) in sorted order

        # 3. Scatter back to original spatial structure
        y_out = torch.zeros(B, D, L, device=ys.device, dtype=ys.dtype)
        sort_idx_expand = sort_idx.unsqueeze(1).expand(-1, D, -1)
        
        y_out.scatter_(2, sort_idx_expand, y_combined)

        return y_out.view(B, D, -1) # Returns (B, D, L)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, D, L)
        H, W = ctx.hw
        sort_idx, = ctx.saved_tensors
        B, D, L = grad_output.shape

        # 1. Gather gradients to sorted order
        sort_idx_expand = sort_idx.unsqueeze(1).expand(-1, D, -1)
        grad_sorted = torch.gather(grad_output, 2, sort_idx_expand)

        # 2. Duplicate for two directions
        grad_ys_0 = grad_sorted
        grad_ys_1 = torch.flip(grad_sorted, dims=[-1])

        # 3. Stack
        grad_ys = torch.stack([grad_ys_0, grad_ys_1], dim=1) # (B, 2, D, L)

        # 4. View back to (B, K, D, H, W) to match input requirement
        return grad_ys.view(B, 2, D, H, W), None
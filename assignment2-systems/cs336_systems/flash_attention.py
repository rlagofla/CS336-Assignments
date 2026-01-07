import torch
import torch.nn as nn
import triton
import triton.language as tl

class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):  
        B, T, C = Q.shape
        
        BQ = 16  # Row tile size
        BK = 16  # Col tile size
        
        # 模拟在 HBM 中的，全体
        O = torch.zeros_like(Q)
        L = torch.full((B, T), float('-inf'), device=Q.device, dtype=Q.dtype)

        def kernel(b, i):
            # 部分最大值，过程迭代
            m_i = torch.full((BQ, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
            # 部分 sum(e^{})，过程迭代
            l_i = torch.zeros((BQ, 1), device=Q.device, dtype=Q.dtype)
            # 部分 O，过程迭代
            o_i = torch.zeros((BQ, C), device=Q.device, dtype=Q.dtype)

            i_end = i + BQ
            # load
            qi = Q[b, i:i_end, :] # (BQ, D)
            
            for j in range(0, T, BK):
                j_end = j + BK
                # load
                kj = K[b, j:j_end, :] # (BK, D)
                vj = V[b, j:j_end, :] # (BK, D)
                
                s_ij = torch.matmul(qi, kj.transpose(-2, -1)) * C ** -0.5 # (BQ, BK)
                
                # 回忆 top-p，max 返回两个最大和索引
                m_ij, _ = torch.max(s_ij, dim=-1, keepdim=True) # (BQ, 1)
                # 按照伪代码这里应该要更新 m_i，但是伪代码后面还用了 m_i^j-1，所以最后更新
                m_next = torch.maximum(m_i, m_ij)
                
                # (BQ, BK) - (BQ, 1) dim=1 的位置广播，正是按行减去最大值
                p_ij = torch.exp(s_ij - m_next)
                
                l_i = l_i * torch.exp(m_i - m_next) + torch.sum(p_ij, dim=-1, keepdim=True)
                o_i = o_i * torch.exp(m_i - m_next) + torch.matmul(p_ij, vj)
                
                m_i = m_next
                
            # store
            # (BQ, D) / (BQ, 1) dim=1 广播，相当于左乘对角
            O[b, i:i_end, :] = o_i / l_i
            # 前面一直 keepDim，现在把最后一个维度拿掉
            L[b, i:i_end] = (m_i + torch.log(l_i)).squeeze(-1)
        
        # 模拟并行
        # 双层，所以 grid 是 2D 的
        for b in range(B):
            for i in range(0, T, BQ):
                kernel(b, i)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        
        return O

    @staticmethod
    def backward(ctx, dO):
        # 虽然这里都是 dX，但是不是微分，而默认是 dL/dX
        Q, K, V, O, L = ctx.saved_tensors
        B, T, C = Q.shape
        D = torch.sum(O * dO, dim=-1, keepdim=True)
        scale = C ** -0.5

        S = Q @ K.transpose(-2, -1) * scale
        P = torch.exp(S - L.unsqueeze(-1))
        dV = P.transpose(-2, -1) @ dO
        dP = dO @ V.transpose(-2, -1)
        dS = P * (dP - D)
        dQ = dS @ K * scale
        dK = dS.transpose(-2, -1) @ Q * scale

        return dQ, dK, dV, None
    
@triton.jit 
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd, 
    stride_kb, stride_kk, stride_kd, 
    stride_vb, stride_vk, stride_vd, 
    stride_ob, stride_oq, stride_od, 
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, 
    scale, 
    is_causal: tl.constexpr,
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr, 
): 
    # Program indices 
    # 很底层了，没有广播用了，所以 batch 并行要自己做了
    # 每个 kernel 最好是处理 2D 矩阵
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1) 
    # Offset each pointer with the corresponding batch index 
    # multiplied with the batch stride for each tensor 
    Q_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    K_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        K_ptr + batch_index * stride_kb, 
        shape=(N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        # 还真写错了，去看了原理，每个 query tile，key 都要在内部的循环从头遍历的
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    V_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        V_ptr + batch_index * stride_vb, 
        shape=(N_KEYS, D), 
        strides=(stride_vk, stride_vd), 
        # value 和 key 是一样的
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    O_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        O_ptr + batch_index * stride_ob, 
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od), 
        # o 和 q 是一样的
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    L_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        L_ptr + batch_index * stride_lb, 
        shape=(N_QUERIES,), 
        strides=(stride_lq,), 
        # 比 q 少一维 D
        offsets=(query_tile_index * Q_TILE_SIZE,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,), 
    )
    # 行由 query 确定
    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    # m_i = tl.arange(0, Q_TILE_SIZE)
    # tl.device_print("mi value: ", m_i)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    
    loop_end = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        # 算 key 超过 query 的那部分，直接不循环了，对应逻辑里的 break
        tmp = tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE)
        loop_end = tl.minimum(loop_end, tmp)
    
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
    for j in range(loop_end):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        
        s_ij = tl.dot(q_i, k_j.trans()) * scale
        
        # 算完部分分，找最大值之前，做 mask
        if is_causal and (j == loop_end - 1):
            # 列由 key 决定
            offs_n = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            # 注意 where 和 mask_fill 不一样
            # where 是 True 的地方 s_ij；False 的地方 -inf
            s_ij = tl.where(causal_mask, s_ij, float("-inf"))
        
        m_ij = tl.max(s_ij, 1)
        # tl.device_print("mij value: ", m_ij)
        m_next = tl.maximum(m_i, m_ij)
        
        # 类似 unsqueeze
        p_ij = tl.exp(s_ij - m_next[:,None])
        
        update = tl.exp(m_i - m_next)
        l_i = l_i * update + tl.sum(p_ij, 1)
        o_i = o_i * update[:,None] + tl.dot(p_ij, v_j)
        
        m_i = m_next
        
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(O_block_ptr, o_i / l_i[:,None], boundary_check=(0, 1))
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))
    
    
@triton.jit 
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, D_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd, 
    stride_kb, stride_kk, stride_kd, 
    stride_vb, stride_vk, stride_vd, 
    stride_ob, stride_oq, stride_od, 
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod, 
    stride_dqb, stride_dqq, stride_dqd, 
    stride_dkb, stride_dkk, stride_dkd, 
    stride_dvb, stride_dvk, stride_dvd, 
    N_QUERIES, N_KEYS, 
    scale, 
    is_causal: tl.constexpr,
    D: tl.constexpr, 
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr, 
): 
    # key 方面并行了
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1) 
    # Offset each pointer with the corresponding batch index 
    # multiplied with the batch stride for each tensor 
    Q_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    K_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        K_ptr + batch_index * stride_kb, 
        shape=(N_KEYS, D), 
        strides=(stride_kk, stride_kd), 
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    V_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        V_ptr + batch_index * stride_vb, 
        shape=(N_KEYS, D), 
        strides=(stride_vk, stride_vd), 
        # value 和 key 是一样的
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    O_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        O_ptr + batch_index * stride_ob, 
        shape=(N_QUERIES, D), 
        strides=(stride_oq, stride_od), 
        # o 和 q 是一样的
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    L_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        L_ptr + batch_index * stride_lb, 
        shape=(N_QUERIES,), 
        strides=(stride_lq,), 
        # 比 q 少一维 D
        offsets=(0,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,), 
    )
    
    D_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        D_ptr + batch_index * stride_db, 
        shape=(N_QUERIES,), 
        strides=(stride_dq,), 
        # 比 q 少一维 D
        offsets=(0,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,), 
    )

    dO_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        dO_ptr + batch_index * stride_dob, 
        shape=(N_QUERIES, D), 
        strides=(stride_doq, stride_dod), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0), 
    )

    dQ_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        dQ_ptr + batch_index * stride_dqb, 
        shape=(N_QUERIES, D), 
        strides=(stride_dqq, stride_dqd), 
        offsets=(0, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    dK_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        dK_ptr + batch_index * stride_dkb, 
        shape=(N_KEYS, D), 
        strides=(stride_dkk, stride_dkd), 
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0), 
    )
    
    dV_block_ptr = tl.make_block_ptr(
        # 每次只看一个 batch
        dV_ptr + batch_index * stride_dvb, 
        shape=(N_KEYS, D), 
        strides=(stride_dvk, stride_dvd), 
        # value 和 key 是一样的
        offsets=(key_tile_index * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, D), 
        order=(1, 0), 
    )
    # 列由 key 确定，类似 fwd 的风格
    offs_n = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    k_j = tl.load(K_block_ptr, boundary_check=(0, 1))
    v_j = tl.load(V_block_ptr, boundary_check=(0, 1))

    dk_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dv_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    loop_start = 0
    if is_causal:
        # 从对角线循环
        # 直接赋值，等效 maximum
        loop_start = key_tile_index * K_TILE_SIZE // Q_TILE_SIZE
        # 用于这回是中途开始，所以需要提前 advance 好指针
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE * loop_start, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE * loop_start, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE * loop_start, 0))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE * loop_start, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE * loop_start,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE * loop_start,))

    for i in range(loop_start, tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
        o_i = tl.load(O_block_ptr, boundary_check=(0, 1))
        do_i = tl.load(dO_block_ptr, boundary_check=(0, 1))
        dq_i = tl.load(dQ_block_ptr, boundary_check=(0, 1))

        l_i = tl.load(L_block_ptr, boundary_check=(0,))
        d_i = tl.load(D_block_ptr, boundary_check=(0,))

        s_ij = tl.dot(q_i, k_j.trans()) * scale
        if is_causal and (i == loop_start):
            offs_m = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(mask, s_ij, float('-inf'))
        p_ij = tl.exp(s_ij - l_i[:,None])

        dv_j += tl.dot(p_ij.trans(), do_i)
        dp_ij = tl.dot(do_i, v_j.trans())
        ds_ij = p_ij * (dp_ij - d_i[:,None]) * scale

        # 原子加，要手动计算偏移
        offs_dq_m = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        offs_dq_n = tl.arange(0, D)
        dq_step = tl.dot(ds_ij, k_j)
        dq_ptrs = dQ_ptr + batch_index * stride_dqb + (offs_dq_m[:, None] * stride_dqq + offs_dq_n[None, :] * stride_dqd)
        tl.atomic_add(dq_ptrs, dq_step)

        dk_j += tl.dot(ds_ij.trans(), q_i)
        
        # 别忘了 advance
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        dQ_block_ptr = dQ_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dk_j, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_j, boundary_check=(0, 1))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, T, D = Q.shape
        O = torch.zeros_like(Q)
        L = torch.full((B, T), float('-inf'), device=Q.device, dtype=torch.float32)
        
        BQ = 16
        BK = 16
        grid = (triton.cdiv(T, BQ), B)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2), 
            V.stride(0), V.stride(1), V.stride(2), 
            O.stride(0), O.stride(1), O.stride(2), 
            L.stride(0), L.stride(1), 
            T, T,
            D ** -0.5,
            is_causal,
            D=D, Q_TILE_SIZE=BQ, K_TILE_SIZE=BK
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        # 虽然这里都是 dX，但是不是微分，而默认是 dL/dX
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        B, T, C = Q.shape
        D = torch.sum(O * dO, dim=-1)
        
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        BQ = 16
        BK = 16
        # key 方面并行，所以 grid 要改
        grid = (triton.cdiv(T, BK), B)
        flash_bwd_kernel[grid](
            Q, K, V,
            O, L, D, dO,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2), 
            V.stride(0), V.stride(1), V.stride(2), 
            O.stride(0), O.stride(1), O.stride(2), 
            L.stride(0), L.stride(1), 
            D.stride(0), D.stride(1), 
            dO.stride(0), dO.stride(1), dO.stride(2), 
            dQ.stride(0), dQ.stride(1), dQ.stride(2), 
            dK.stride(0), dK.stride(1), dK.stride(2), 
            dV.stride(0), dV.stride(1), dV.stride(2), 
            T, T,
            C ** -0.5,
            is_causal,
            D=C, Q_TILE_SIZE=BQ, K_TILE_SIZE=BK
        )

        return dQ, dK, dV, None

if __name__ == '__main__':
    Q = torch.rand((1, 16, 16), device='cuda')
    K = torch.rand((1, 16, 16), device='cuda')
    V = torch.rand((1, 16, 16), device='cuda')
    out = FlashAttention2Triton.apply(Q, K, V)
    ref_out = FlashAttention2PyTorch.apply(Q, K, V)
    print(torch.allclose(out, ref_out, atol=1e-3))
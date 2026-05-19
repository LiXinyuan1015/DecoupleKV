import torch
import torch.linalg as linalg

import functools

def add_description(description):
    def decorator(func):
        func.__doc__ = description
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@add_description("Omage = argmin[||A * Omega - B||^2_F] subject to C^TC = I")
def solve_Procrustes(A, B):
    device = A.device
    U, _, V_T = linalg.svd(B.T @ A)
    if U.size(0) == V_T.size(0):
        Omega = V_T.T @ U.T
    else:
        I_uv = torch.eye(V_T.size(0), U.size(0), device = device)
        Omega = V_T.T @ I_uv @ U.T
    return Omega

@add_description("Omage = argmin[tr(Omega @ B @ A + Omega @ D @ C)] subject to C^TC = I")
def solve_Procrustes_multi(A, B, C, D):
        device = A.device
        U, _, V_T = torch.linalg.svd(B.T @ A + D.T @ C)
        if U.size(0) == V_T.size(0):
            Omega = V_T.T @ U.T
        else:
            I_uv = torch.eye(V_T.size(0), U.size(0), device = device)
            Omega = V_T.T @ I_uv @ U.T
        return Omega

@add_description("Gram Schmidt Orthogonalization")
def gram_schmidt_complete(Q_k):
    n, k = Q_k.size()
    if n <= k:
        return Q_k
    Q = torch.zeros((n, n-k), device = Q_k.device)
    Q = torch.cat([Q_k, Q], dim = 1)

    for i in range(k, n):
        v = torch.rand(n, device=Q.device)
        v -= Q[:, :i] @ torch.matmul(Q[:, :i].T, v)
        v /= torch.norm(v)
        Q[:, i] = v

    return Q

@add_description("Complete Orthogonalization")
def complete_orthogonal_matrix(C):
    p, n = C.shape
    assert p >= n, "C 的行数必须 ≥ 列数"
    
    Q_rand = torch.randn(p, p - n, device=C.device)
    Q_rand -= C @ (C.T @ Q_rand)
    
    Q_orth, _ = torch.linalg.qr(Q_rand)
    
    Q = torch.cat([C, Q_orth], dim=1)
    return Q
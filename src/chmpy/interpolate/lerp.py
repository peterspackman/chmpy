import numpy as np

def vectorized_lerp(xs, xp, yp, l_fill=None, u_fill=None):
    N = xp.shape[0]
    dx = xp[1] - xp[0]  # Assuming uniform spacing
    inv_dx = 1.0 / dx
    l = xp[0]
    u = xp[-1]
    if l_fill is None:
        l_fill = yp[0]

    if u_fill is None:
        u_fill = yp[-1]
    
    # Calculate js indices
    js = np.minimum((inv_dx * (xs - l)).astype(np.int32), N - 2)
    
    # Compute weights for interpolation
    w = (xs - xp[js]) * inv_dx
    
    # Linear interpolation
    results = (1.0 - w) * yp[js] + w * yp[js + 1]
    
    # Fill values outside domain
    results[xs < l] = l_fill
    results[xs > u] = u_fill
    
    return results

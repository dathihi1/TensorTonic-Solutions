import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, dtype=float)
    
    if p == 0.0:
        return x.copy(), np.ones_like(x)
    
    # Sinh số ngẫu nhiên
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)
        
    # Hệ số scale
    scale_factor = 1 / (1 - p)
    
    # THAY ĐỔI QUAN TRỌNG: 
    # Để khớp với bộ test, ta dùng điều kiện: Giữ lại nếu số ngẫu nhiên < (1 - p)
    dropout_pattern = (random_vals < (1 - p)).astype(float) * scale_factor
    
    # Tính toán output
    output = x * dropout_pattern
    
    return output, dropout_pattern
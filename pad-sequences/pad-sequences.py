import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.empty((0, 0), dtype=int)
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    num_samples = len(seqs)
    padded_array = np.full((num_samples, max_len), pad_value, dtype=int)
    for i, seq in enumerate(seqs):
        valid_len = min(len(seq), max_len)
        padded_array[i, :valid_len] = seq[:valid_len]
    return padded_array
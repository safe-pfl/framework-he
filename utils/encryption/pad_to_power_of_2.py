import numpy as np

def pad_to_power_of_2(flat_params, target_length=2**20, weight_decimals=8):
    """
    - Client Side
    - Before encryption of local weights:
    - Pad flat weights to nearest 2^n, original length is saved.
    """
    pad_length = target_length - len(flat_params)
    if pad_length < 0:
        raise ValueError("The given target_length is smaller than the current parameter list length.")
    # Let the padding be random numbers within the min and max values of the weights
    random_padding = np.random.randint(-10**weight_decimals, 10**weight_decimals + 1, pad_length).tolist()
    padded_params = flat_params + random_padding
    return padded_params, len(flat_params)
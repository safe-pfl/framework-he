import torch

def pad_to_power_of_2(flat_params, target_length=2**20, weight_decimals=8):
    current_length = len(flat_params)
    pad_length = target_length - current_length

    # Use torch.randint for integer random padding on the same device and dtype
    random_padding = torch.randint(
        low=-10**weight_decimals,
        high=10**weight_decimals + 1,
        size=(pad_length,),
        device=flat_params.device,
        dtype=torch.int64  # randint returns integers, so int64 is typical
    ).to(flat_params.dtype)  # convert to same dtype as flat_params (e.g., float32)

    padded_params = torch.cat([flat_params, random_padding])
    return padded_params, current_length

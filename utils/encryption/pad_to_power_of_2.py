import torch


def pad_to_power_of_2(flat_params, target_length=2**20, weight_decimals=8):
    current_length = len(flat_params)
    pad_length = target_length - current_length

    # Calculate the mean and std of the actual parameters to generate similar padding
    mean_val = torch.mean(flat_params).item()
    std_val = max(torch.std(flat_params).item(), 1e-5)  # Avoid zero std

    # Use much smaller random values for padding (1/100 of the original range)
    # This reduces the impact of padding on encryption noise
    random_padding = torch.normal(
        mean=0,  # Center at zero
        std=std_val * 0.01,  # Use very small std to minimize impact
        size=(pad_length,),
        device=flat_params.device,
        dtype=flat_params.dtype,
    )

    padded_params = torch.cat([flat_params, random_padding])
    return padded_params, current_length

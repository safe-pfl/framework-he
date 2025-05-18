from xmkckks import RLWE

from constants.framework import GAUSSIAN_DISTRIBUTION
from utils.count_parameters import count_parameters
from utils.encryption.next_prime import next_prime
from utils.log import Log
import math

from validators.config_validator import ConfigValidator


def rlwe_generator(model, config: ConfigValidator, log: Log) -> RLWE:
    log.info(
        "----------    Homomorphic xMK-CKKS (RLEW) initialization --------------------------------------------------"
    )

    # find closest 2^x larger than number of weights
    num_weights = count_parameters(model, config.MODEL_TYPE, log)
    n = 2 ** math.ceil(math.log2(num_weights))
    log.info(f"the vlue for RLWE `n` is: {n}")

    # decide value range t of plaintext - use a more conservative value
    max_weight_value = 10**config.XMKCKKS_WEIGHT_DECIMALS
    num_clients = config.NUMBER_OF_CLIENTS
    # Use a more conservative multiplier to avoid overflow
    t = next_prime(num_clients * max_weight_value * 1.5)
    log.info(f"the vlue for RLWE `t` is: {t}")

    # decide value range q of encrypted plaintext - use a more moderate multiplier
    q = next_prime(t * 20)  # Reduced from 50 to 20 for better stability
    log.info(f"the vlue for RLWE `q` is: {q}")

    config.RUNTIME_CONFIG.q = q

    # standard deviation of Gaussian distribution - use a smaller value
    std = GAUSSIAN_DISTRIBUTION - 1  # Reduced from 3 to 2 for less noise

    return RLWE(n, q, t, std)

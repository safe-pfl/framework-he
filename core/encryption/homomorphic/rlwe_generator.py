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

    # Find closest 2^x larger than number of weights (like in reference code)
    num_weights = count_parameters(model, config.MODEL_TYPE, log)
    n = 2 ** math.ceil(math.log2(num_weights))
    log.info(f"RLWE parameter n: {n} (power of 2 >= num_weights {num_weights})")

    # Calculate t based on weight decimals and number of clients (like in reference code)
    max_weight_value = 10**config.XMKCKKS_WEIGHT_DECIMALS
    num_clients = config.NUMBER_OF_CLIENTS
    t = next_prime(num_clients * max_weight_value * 2)
    log.info(
        f"RLWE parameter t: {t} (prime > num_clients({num_clients}) * max_weight_value({max_weight_value}) * 2)"
    )

    # Calculate q as t * 50 (like in reference code)
    q = next_prime(t * 50)
    log.info(f"RLWE parameter q: {q} (prime > t * 50)")

    # Store q in runtime config
    config.RUNTIME_CONFIG.q = q

    # Set standard deviation to 3 (like in reference code)
    std_rlwe = GAUSSIAN_DISTRIBUTION
    log.info(f"RLWE instance std for encryption noise: {std_rlwe}")

    return RLWE(n, q, t, std_rlwe)

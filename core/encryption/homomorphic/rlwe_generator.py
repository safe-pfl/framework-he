from xmkckks import RLWE

from constants.framework import GAUSSIAN_DISTRIBUTION
from utils.count_parameters import count_parameters
from utils.encryption.next_prime import next_prime
from utils.log import Log
import math

from validators.config_validator import ConfigValidator


def rlwe_generator(model, config: ConfigValidator, log: Log) -> RLWE:
    log.info("----------    Homomorphic xMK-CKKS (RLEW) initialization --------------------------------------------------")

    # find closest 2^x larger than number of weights
    num_weights = count_parameters(model, config.MODEL_TYPE, log)
    n = 2 ** math.ceil(math.log2(num_weights))
    log.info(f'the vlue for RLWE `n` is: {n}')

    # decide value range t of plaintext
    max_weight_value = 10 ** config.XMKCKKS_WEIGHT_DECIMALS  # 100_000_000 if full weights
    num_clients = 2
    t = next_prime(num_clients * max_weight_value * 2)  # 2_000_000_011
    log.info(f'the vlue for RLWE `t` is: {t}')

    # decide value range q of encrypted plaintext
    q = next_prime(t * 50)  # *50 = 100_000_000_567
    log.info(f'the vlue for RLWE `q` is: {q}')

    config.RUNTIME_COMFIG.q = q


    # standard deviation of Gaussian distribution
    std = GAUSSIAN_DISTRIBUTION

    return RLWE(n, q, t, std)



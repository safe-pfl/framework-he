from utils.encryption.is_prime import is_prime


def next_prime(n):
    """Find the smallest prime number larger than n. """
    if n <= 1:
        return 2
    prime = n
    found = False
    while not found:
        prime += 1
        if is_prime(prime):
            found = True
    return prime
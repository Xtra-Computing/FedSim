import numpy as np
from scipy.special import erf
from scipy.optimize import bisect
from phe import paillier


class SimNoiseScale:
    def __init__(self, bf_std, noise_scale=None, sim_leak_p=None):
        self.bf_std = bf_std

        if np.isclose(sim_leak_p, 0.0):
            assert False, "The required noise is np.inf to reach 0 leaking probability"

        noise_to_p = lambda x: erf(np.sqrt(x ** 2 + 1) / (2 * np.sqrt(2) * x * bf_std))

        if noise_scale is None and sim_leak_p is not None:
            # plus 1e-12 to ensure the noise is sufficient for probability p
            noise_scale = bisect(lambda x: noise_to_p(x) - sim_leak_p, 1e-8, 1e8, xtol=1e-12) + 1e-12
        elif sim_leak_p is None and noise_scale is not None:
            sim_leak_p = noise_to_p(noise_scale)
        else:
            assert False, "noise_scale={}, sim_leak_p={}".format(noise_scale, sim_leak_p)

        self.noise_scale = noise_scale
        self.sim_leak_p = sim_leak_p


def l2_distance_with_he(a, encrypted_b, encrypted_b_square,
                        private_key: paillier.PaillierPrivateKey):
    """
    Calculate l2 distance with partial homomorphic encryption
    :param encrypted_b_square:
    :param encrypted_b:
    :param a: array 1
    :param private_key: private key to decrypt result
    :return: real distance
    """
    encrypted_dists = a * a - 2 * a * encrypted_b + encrypted_b_square
    encrypted_dist = sum(encrypted_dists)
    dist = private_key.decrypt(encrypted_dist)
    return dist


def jaccard_sim_with_he(a, b, public_key: paillier.PaillierPublicKey,
                        private_key: paillier.PaillierPrivateKey):
    """
    Calculate jaccard similarity with partial homomorphic encryption
    :param a: bit vector 1
    :param b: bit vector 2 (to be encrypted)
    :param public_key: public key to encrypt value
    :param private_key: private key to decrypt result
    :return: real jaccard similarity
    """
    encrypted_b = [public_key.encrypt(b_i) for b_i in b]
    raise NotImplementedError

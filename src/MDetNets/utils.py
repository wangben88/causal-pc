import random
import jax.numpy as jnp

def random_scope_partition(scope, n):
    """Randomly partitions a given scope into n nearly equal parts.

    :param scope: iterable representing scope
    :param n: Number of parts
    :return: Random nearly-equal n-partition of scope
    """
    scope_list = list(scope)
    random.shuffle(scope_list)
    partition = [frozenset(scope_list[i::n]) for i in range(n)]
    return partition

def tensordot_logdomain(arr_1, arr_2):
    return jnp.reshape(arr_1, newshape=arr_1.shape + (1,) * arr_2.ndim) + jnp.reshape(arr_2, newshape=(1,)*arr_1.ndim +  arr_2.shape)

def interleave(tup_1, tup_2):
    tup_output = list(tup_1 + tup_2)
    tup_output[0::2] = tup_1
    tup_output[1::2] = tup_2
    return tuple(tup_output)

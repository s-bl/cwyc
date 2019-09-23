import os
import importlib
import struct

import numpy as np
from collections.abc import Mapping
import logging
import hashlib
from six import integer_types

def make_nested_dict(dicts):
    out = dict()

    for key, value in dicts.items():
        keys = key.split('/')[1:]
        tmp = out
        for key_ in keys[:-1]:
            if key_ not in tmp:
                tmp[key_] = dict()
            tmp = tmp[key_]
        tmp[keys[-1]] = value

    return out

def make_shallow_dict(dicts):
    out = dict()

    def unpack_dict(prefix, item):
        if isinstance(item[1], dict):
            for item_ in item[1].items():
                unpack_dict(prefix + '/' + str(item[0]), item_)
        else:
            out[prefix + '/' + str(item[0])] = item[1]

    for item in dicts.items():
        unpack_dict('', item)

    return out

def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn

def to_transitions(episode):
    """ flatten rollouts to one big list"""
    out = {key: np.concatenate(value, axis=0) for key, value in episode.items()}
    return out

def normalize(x, zero_out=False):
    if zero_out:
        p = np.zeros_like(x) if sum(x) == 0 else x / np.sum(x)
    else:
        p = np.ones_like(x) / x.size if sum(x) == 0 else x / np.sum(x)

    return p

def get_parameter(*args, params, default=None):
    if args[0] not in params:
        return default
    if len(args) == 1:
        if default is not None and params[args[0]] is None:
            return default
        else:
            return params[args[0]]

    return get_parameter(*args[1:], params=params[args[0]], default=default)

def update_default_params(d, u):
    if isinstance(u, list):
        for i, item in enumerate(u):
            if i >= len(d):
                d.append(item)
            else:
                if isinstance(d[i], Mapping):
                    d[i] = update_default_params(d[i], item)
                else:
                    d[i] = item
    else:
        for k, v in u.items():
            if isinstance(v, Mapping):
                d[k] = update_default_params(d.get(k, {}) or {}, v)
            elif isinstance(v, list):
                d[k] = update_default_params(d.get(k, []) or [], v)
            else:
                d[k] = v
    return d

def get_jobdir(logger):
    logger_handlers = logger.root.handlers
    file_handlers = [handler for handler in logger_handlers if isinstance(handler, logging.FileHandler)]
    assert len(file_handlers) == 1
    file_handler = file_handlers[0]
    jobdir = os.path.dirname(file_handler.baseFilename)

    return jobdir

def create_seed(a=None, max_bytes=8):
    """Create a strong random seed. Otherwise, Python 2 would seed using
    the system time, which might be non-robust especially in the
    presence of concurrency.

    Args:
        a (Optional[int, str]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the seed.
    """
    # Adapted from https://svn.python.org/projects/python/tags/r32/Lib/random.py
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode('utf8')
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, integer_types):
        a = a % 2**(8 * max_bytes)
    else:
        raise Exception('Invalid type for seed: {} ({})'.format(type(a), a))

    return a

def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

if __name__ == "__main__":
    from pprint import pprint

    kwargs = dict(
        gnet_params = dict(
            pos_buffer_size=int(1e3),
            neg_buffer_size=int(1e5),
            batch_size=64,
            learning_rate=1e-4,
            train_steps=100,
            only_fst_surprising_singal=True,
            only_pos_rollouts=False,
        )
    )

    params = dict(
        gnet_params=dict(
            network_params=dict(
                net_type="gnet.models:Simple"
            )
        )
    )

    update(kwargs, params)

    pprint(kwargs)

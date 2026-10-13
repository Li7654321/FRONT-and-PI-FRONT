#!/usr/bin/env python
"""
Scaling and inverse scaling of source parameters.
"""

# Physical scales and magnitude bounds.
MAX_SLIP = 20.0   # m
MAX_RISE = 35.0   # s
MAX_RUPT = 260.0  # s
MIN_MAG = 6.0
MAX_MAG = 9.0


def d_scale(y):
    return y / MAX_SLIP

def d_scaleback(y):
    return y * MAX_SLIP

def r_scale(x, inverse=False):
    if inverse:
        return x * MAX_RISE
    return x / MAX_RISE

def t_scale(x, max_val=MAX_RUPT, inverse=False):
    if inverse:
        return x * max_val
    return x / max_val

def m_scale(mw):
    return (mw - MIN_MAG) / (MAX_MAG - MIN_MAG)

def m_descale(mw_norm):
    return mw_norm * (MAX_MAG - MIN_MAG) + MIN_MAG
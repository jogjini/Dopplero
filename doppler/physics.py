"""Emission-time Doppler model shared by the generator, the fitter and the ADS-B labeller.

Conventions: metres, seconds; x east, y north, z up, ground at z = 0.
t_e ("emission time") is when the aircraft made a sound, t_r = t_e + r/c ("reception time")
is when the mic hears it. Received frequency = source frequency * dt_e/dt_r, which for a
still medium equals c / (c + v_r) with v_r the range rate at emission time.
"""
import numpy as np


def speed_of_sound(temp_c):
    return 331.3 * np.sqrt(1.0 + np.asarray(temp_c, dtype=float) / 273.15)


def arrival_times(t_e, pos, mic, c):
    """Arrival time and range for positions pos (N, 3) sampled at emission times t_e (N,)."""
    r = np.linalg.norm(pos - mic, axis=-1)
    return t_e + r / c, r


def emission_times(t_r, t_e, arrival):
    """Invert arrival(t_e) at reception times t_r; NaN outside the covered range.

    Valid because arrival(t_e) is strictly increasing for any subsonic source.
    """
    return np.interp(t_r, arrival, t_e, left=np.nan, right=np.nan)


def cpa(t_e, pos, mic, c):
    """Closest point of approach of a sampled path: slant distance, speed, and the
    reception time of the sound emitted there (the "CPA time" we report everywhere)."""
    r = np.linalg.norm(pos - mic, axis=-1)
    i = int(np.argmin(r))
    vel = np.gradient(pos, t_e, axis=0)
    return {"t_cpa": float(t_e[i] + r[i] / c), "d_cpa": float(r[i]),
            "speed": float(np.linalg.norm(vel[i])), "t_cpa_emit": float(t_e[i])}


def straight_pass(t, t_cpa, v, d, c):
    """(f_received / f_source, range at emission) for a straight, constant-speed pass
    (closed form, DESIGN.md §2).

    t_cpa is the arrival time of the sound emitted at closest approach; at t = t_cpa the
    ratio is exactly 1. Broadcasts over all arguments.
    """
    T = t - (t_cpa - d / c)
    c2, v2 = c * c, v * v
    s = (c2 * T - np.sqrt(c2 * v2 * T * T + d * d * (c2 - v2))) / (c2 - v2)
    r = np.sqrt(d * d + v2 * s * s)
    return c / (c + v2 * s / r), r


def straight_pass_ratio(t, t_cpa, v, d, c):
    return straight_pass(t, t_cpa, v, d, c)[0]


def air_absorption_db_per_m(f, temp_c, rh_percent, pressure_kpa=101.325):
    """Atmospheric absorption (ISO 9613-1) in dB per metre at frequency f (Hz)."""
    f = np.asarray(f, dtype=float)
    T, T0, T01 = temp_c + 273.15, 293.15, 273.16
    pr = pressure_kpa / 101.325
    h = rh_percent * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151) / pr
    fr_o = pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pr * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    f2 = f * f
    return 8.686 * f2 * (1.84e-11 / pr * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (fr_o + f2 / fr_o)
        + 0.1068 * np.exp(-3352.0 / T) / (fr_n + f2 / fr_n)))

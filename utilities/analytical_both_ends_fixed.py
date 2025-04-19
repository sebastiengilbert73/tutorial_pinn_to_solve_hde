import os
import logging
import pandas as pd
import argparse
import numpy as np
import math
import sys
sys.path.append("..")
import fourier.series

def main(
    alpha,
    duration,
    numberOfTimesteps,
    initialTemperatureProfile
):
    logging.info("analytical_both_ends_fixed.main()")

    # Load the initial temperature profile
    ux0_df = pd.read_csv(initialTemperatureProfile)
    ux0 = ux0_df.values  # (N_points, 2)
    length = ux0[-1, 0]
    number_of_points = ux0.shape[0]
    xs = ux0[:, 0]
    u0 = ux0[:, 1]

    # Compute the Fourier series of u0.
    # Since both ends are fixed, we'll use an odd half-range expansion
    # u0_hat(x) = u0(x) - Cx - D
    C = (u0[-1] - u0[0]) / length
    D = u0[0]
    u0_hat = u0 - C * xs - D

    expander = fourier.series.Expander(length, 'odd')
    a, b = expander.coefficients(u0_hat)

    # Compute the lambda_n
    lambda_n = []
    for n in range(len(a)):
        lambda_n.append(-alpha * n ** 2 * math.pi ** 2 / length ** 2)

    u = np.zeros((numberOfTimesteps, number_of_points))
    delta_t = duration / (numberOfTimesteps - 1)
    ts = np.arange(0, duration + delta_t / 2, delta_t)
    for t_ndx in range(len(ts)):
        t = ts[t_ndx]
        for x_ndx in range(len(xs)):
            x = xs[x_ndx]
            sum = C * x + D
            for n in range(1, len(lambda_n)):
                sum += np.exp(lambda_n[n] * t) * b[n] * math.sin(n * math.pi * x / length)
            u[t_ndx, x_ndx] = sum

    return u
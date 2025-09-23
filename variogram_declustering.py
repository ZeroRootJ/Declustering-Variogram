"""
Python Implementation of
Rezvandehy, M., Deutsch, C.V.
Declustering experimental variograms by global estimation with fourth order moments.
Stoch Environ Res Risk Assess 32, 261–277 (2018).
https://doi.org/10.1007/s00477-017-1388-x

9/23/2025 Youngkeun Jung
"""


import numpy as np
import pandas as pd

def spherical_variogram_model(h, sill, range_a):
    """
    Calculates the Spherical Variogram Model.
    This function is used as the initial variogram model.

    Args:
        h (float or np.array): The distance(s) or lag(s).
        sill (float): The sill of the model.
        range_a (float): The range of the model.

    Returns:
        float or np.array: The variogram value(s) for the given distance(s).
    """
    h = np.asarray(h)
    gamma = np.zeros_like(h, dtype=float)

    # Case for distances within the range
    mask = h < range_a
    h_masked = h[mask]
    gamma[mask] = sill * (1.5 * (h_masked / range_a) - 0.5 * (h_masked / range_a) ** 3)

    # Case for distances beyond the range
    gamma[h >= range_a] = sill

    return gamma


def covariance_from_variogram(h, sill, range_a, model_func):
    """
    Calculates the covariance from a variogram model.
    C(h) = C(0) - γ(h), where C(0) is the sill.

    Params:
        h (float or np.array): The distance(s) or lag(s).
        sill (float): The sill of the variogram model.
        range_a (float): The range of the variogram model.
        model_func (function): The variogram model function to use (e.g., spherical_variogram_model).

    Returns:
        float or np.array: The covariance value(s) for the given distance(s).
    """
    return sill - model_func(h, sill, range_a)


def calculate_fourth_order_covariance(pair1, pair2, sill, range_a, cov_func):
    """
    Calculates the fourth-order covariance between two data pairs (pair1, pair2)

    Params:
        pair1 (tuple): The first data pair ((coords, value), (coords, value)).
        pair2 (tuple): The second data pair ((coords, value), (coords, value)).
        sill (float): The sill of the variogram model for covariance calculation.
        range_a (float): The range of the variogram model for covariance calculation.
        cov_func (function): The function to calculate covariance.

    Returns:
        float: The fourth-order covariance value between the two pairs.
    """
    coords1_head, coords1_tail = pair1[0][0], pair1[1][0]
    coords2_head, coords2_tail = pair2[0][0], pair2[1][0]

    dist_ac = np.linalg.norm(coords1_head - coords2_head)
    dist_ad = np.linalg.norm(coords1_head - coords2_tail)
    dist_bc = np.linalg.norm(coords1_tail - coords2_head)
    dist_bd = np.linalg.norm(coords1_tail - coords2_tail)

    cov_ac = cov_func(dist_ac, sill, range_a)
    cov_ad = cov_func(dist_ad, sill, range_a)
    cov_bc = cov_func(dist_bc, sill, range_a)
    cov_bd = cov_func(dist_bd, sill, range_a)

    term_in_bracket = cov_ac - cov_ad - cov_bc + cov_bd

    fourth_order_cov = 2 * (term_in_bracket ** 2)

    return fourth_order_cov


def solve_ogk_weights(pairs, sill, range_a, data_coords, grid_resolution):
    """
    Constructs and solves the OGK system to calculate variogram weights.

    Params:
        pairs (list): A list of data pairs.
        sill (float): The sill of the initial variogram model.
        range_a (float): The range of the initial variogram model.
        data_coords (np.array): All data coordinates
        grid_resolution (int): The number of grid cells along each axis

    Returns:
        np.array: An array of weights for each pair.
    """
    n_pairs = len(pairs)
    print(f"Constructing OGK system for {n_pairs} pairs...")

    # K: The fourth-order covariance matrix between pairs (Left-hand side)
    K = np.zeros((n_pairs, n_pairs))
    cov_func = lambda h, s, r: covariance_from_variogram(h, s, r, spherical_variogram_model)
    for i in range(n_pairs):
        for j in range(i, n_pairs):
            cov = calculate_fourth_order_covariance(pairs[i], pairs[j], sill, range_a, cov_func)
            K[i, j] = cov
            K[j, i] = cov

    # d: The right-hand side of the kriging system.
    if n_pairs == 0:
        d = np.array([])
    else:
        min_x, min_y = np.min(data_coords, axis=0)
        max_x, max_y = np.max(data_coords, axis=0)
        x_grid = np.linspace(min_x, max_x, grid_resolution)
        y_grid = np.linspace(min_y, max_y, grid_resolution)
        h_avg = np.mean([p[1][0] - p[0][0] for p in pairs], axis=0)

        d = np.zeros(n_pairs)
        for i in range(n_pairs):
            total_cov, count = 0, 0
            pair_i = ((pairs[i][0][0], 0), (pairs[i][1][0], 0))
            for xg in x_grid:
                for yg in y_grid:
                    grid_head = np.array([xg, yg])
                    grid_tail = grid_head + h_avg
                    if (min_x <= grid_tail[0] <= max_x) and (min_y <= grid_tail[1] <= max_y):
                        grid_pair = ((grid_head, 0), (grid_tail, 0))
                        total_cov += calculate_fourth_order_covariance(pair_i, grid_pair, sill, range_a, cov_func)
                        count += 1
            d[i] = total_cov / count if count > 0 else 0


    # Augment and solve the kriging system
    if n_pairs > 0:
        diag_mean = np.mean(np.diag(K))
        nugget = 1e-8 * diag_mean if diag_mean > 0 else 1e-8
        K += np.eye(n_pairs) * nugget

    K_aug = np.ones((n_pairs + 1, n_pairs + 1))
    K_aug[:n_pairs, :n_pairs] = K
    K_aug[n_pairs, n_pairs] = 0
    d_aug = np.ones(n_pairs + 1)
    d_aug[:n_pairs] = d

    try:
        weights_aug = np.linalg.solve(K_aug, d_aug)
        weights = weights_aug[:n_pairs]
        print("OGK system solved successfully.")
        return weights
    except np.linalg.LinAlgError:
        print("Warning: Kriging system is singular. Falling back to pseudo-inverse.")
        inv_K_aug = np.linalg.pinv(K_aug)
        weights_aug = np.dot(inv_K_aug, d_aug)
        return weights_aug[:n_pairs]


def calculate_declustered_variogram_value(pairs, weights):
    """
    Calculates the declustered variogram value using the computed weights.

    Params:
        pairs (list): The list of data pairs.
        weights (np.array): The weights for each pair.

    Returns:
        float: The declustered variogram value for the given lag.
    """
    squared_diffs = []
    for pair in pairs:
        val_head = pair[0][1]
        val_tail = pair[1][1]
        squared_diffs.append((val_head - val_tail) ** 2)

    squared_diffs = np.array(squared_diffs)

    # Calculate the weighted average
    weighted_mean_sq_diff = np.sum(weights * squared_diffs)

    return 0.5 * weighted_mean_sq_diff


def generate_variogram_pairs(df, xcol, ycol, vcol, lag, tol, azm=0, atol=10):
    pairs = []
    coords = df[[xcol, ycol]].values
    values = df[vcol].values
    n = len(df)

    azm_rad = np.deg2rad(azm)
    # Define direction vector based on azimuth (angle from North)
    direction = np.array([np.sin(azm_rad), np.cos(azm_rad)])

    for i in range(n):
        for j in range(i + 1, n):
            h_vec = coords[j] - coords[i]
            h_dist = np.linalg.norm(h_vec)

            # Check if the distance is within the lag tolerance
            if abs(h_dist - lag) <= tol:
                # Calculate angle with the direction vector (avoid division by zero)
                norm_h_vec = np.linalg.norm(h_vec)
                if norm_h_vec < 1e-10: continue

                dot_product = np.dot(h_vec / norm_h_vec, direction)
                # Clip dot_product to be within [-1.0, 1.0] to avoid math errors
                dot_product = np.clip(dot_product, -1.0, 1.0)
                h_angle = np.rad2deg(np.arccos(dot_product))

                # Consider both forward and backward directions
                if min(h_angle, 180 - h_angle) <= atol:
                    pairs.append(((coords[i], values[i]), (coords[j], values[j])))
    return pairs


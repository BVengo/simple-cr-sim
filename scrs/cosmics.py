"""
Cosmic Ray Simulation
This method of simulating cosmic rays has been adapted from the paper "Nancy Grace Roman Space
Telescope (Roman) Technical Report" by J Wu et al. (2023). You can find the paper at:

https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000502-SimulatingCosmicRays.pdf
"""

import numpy as np
import scipy.interpolate as interpolate
from numpy import ndarray, dtype
from typing import Any


def get_rng(rng: np.random.Generator, seed: int) -> np.random.Generator:
    """
    Get a random number generator from a given seed or use the default generator.

    Args:
        rng (np.random.Generator): Random number generator to use.
        seed (int): Seed to use for random number generator.

    Returns:
        np.random.Generator: Random number generator to use.
    """
    return rng if rng is not None else np.random.default_rng(seed)


def create_sampler(pdf: callable, x: np.ndarray) -> callable:
    """
    A function for performing inverse transform sampling.

    Args:
        pdf (callable): A function or empirical set of tabulated values which can be used to
            call or evaluate 'x'.
        x (np.ndarray): 1-d array of values where the pdf should be evaluated.

    Returns:
        callable: A function which can be used to sample from the inverse CDF.
    """
    y = pdf(x)
    cdf_y = np.cumsum(y) - y[0]
    cdf_y /= cdf_y[-1]

    return interpolate.interp1d(cdf_y, x, bounds_error=False, fill_value=(x[0], x[-1]))


def moyal_distribution(x: np.ndarray, location: float = 120, scale: float = 50) -> np.ndarray:
    """
    Return an unnormalized Moyal distribution, which approximates a Landau distribution and is
    used to describe the energy loss probability distribution of a charged particle through a
    detector.

    Args:
        x (np.ndarray): 1-d array of values where the Moyal distribution should be evaluated.
        location (float): The peak location of the distribution, in units of eV / micron.
        scale (float): A width parameter for the distribution, in units of eV / micron.

    Returns:
        np.ndarray: Moyal distribution (pdf) evaluated on 'x' grid of points.
    """
    normalised_x = (x - location) / scale
    return np.exp(-(normalised_x + np.exp(-normalised_x)) / 2)


def power_law_distribution(x: np.ndarray, slope: float = -4.33) -> np.ndarray:
    """
    Return unnormalized power - law distribution parameterized by a log-log slope, used to describe
    the cosmic ray path lengths .

    Args:
        x (np.ndarray): 1-d array of floats. An array of cosmic ray path lengths (units: micron).
        slope (float): The log-log slope of the distribution, default based on
            Miles et al. (2021).

    Returns:
        np.ndarray: Power-law distribution (pdf) evaluated on 'x' grid of points.
    """
    return np.power(x, slope)


def sample_cr_params(
    N_samples: int,
    N_i: int = 4096,
    N_j: int = 4096,
    min_dEdx: float = 10,
    max_dEdx: float = 10000,
    min_cr_len: float = 10,
    max_cr_len: float = 2000,
    grid_size: int = 10000,
    rng: np.random.Generator = None,
    seed: int = 48,
) -> tuple[Any, Any, ndarray[Any, dtype[Any]], Any, Any]:
    """
    Generates cosmic ray parameters randomly sampled from distribution. One might re-implement this
    by reading in parameters from a reference file, or something similar .

    Args:
        N_samples (int): Number of CRs to generate.
        N_i (int): Number of pixels along i-axis of detector.
        N_j (int): Number of pixels along j-axis of detector.
        min_dEdx (float): Minimum value of CR energy loss (dE/dx), units of eV/micron.
        max_dEdx (float): Maximum value of CR energy loss (dE/dx), units of eV/micron.
        min_cr_len (float): Minimum length of cosmic ray trail, units of micron.
        max_cr_len (float): Maximum length of cosmic ray trail, units of micron.
        grid_size (int): Number of points on the cosmic ray length and energy loss grids.
            Increasing this parameter increases the level of sampling for the distributions.
        rng (np.random.Generator): Random number generator to use.
        seed (int): Seed to use for random number generator.

    Returns:
        cr_x (float, between 0 and N_x-1): x pixel coordinate of cosmic ray, units of pixels.
        cr_y (float between 0 and N_y-1): y pixel coordinate of cosmic ray, units of pixels.
        cr_phi (float between 0 and 2*pi): Direction of cosmic ray, units of radians.
        cr_length (float): Cosmic ray length, units of micron.
        cr_d_edx (float): Cosmic ray energy loss, units of eV/micron.
    """
    rng = get_rng(rng, seed)

    # Sample cosmic ray positions in the detector
    cr_x, cr_y = rng.random((2, N_samples)) * np.array([[N_i], [N_j]])

    # Sample angles in radians
    cr_angle = rng.random(N_samples) * 2 * np.pi

    # Generate grid and inverse CDF samplers for cosmic ray lengths and energy losses
    len_grid = np.linspace(min_cr_len, max_cr_len, grid_size)
    inv_cdf_length = create_sampler(power_law_distribution, len_grid)
    cr_length = inv_cdf_length(rng.random(N_samples))

    d_edx_grid = np.linspace(min_dEdx, max_dEdx, grid_size)
    inv_cdf_d_edx = create_sampler(moyal_distribution, d_edx_grid)
    cr_d_edx = inv_cdf_d_edx(rng.random(N_samples))

    return cr_x, cr_y, cr_angle, cr_length, cr_d_edx


def traverse(
    trail_start: tuple[float, float],
    trail_end: tuple[float, float],
    N_i: int = 4096,
    N_j: int = 4096,
    eps: float = 1e-10,
) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    """
    Given a starting and ending pixel , returns a list of pixel coordinates ( ii, jj ) and their
    traversed path lengths. Note that the centers of pixels are treated as integers , while the
    borders are treated as half-integers .

    Args:
        trail_start (float, float): The starting coordinates in (i, j) of the cosmic ray trail,
            in units of pix.
        trail_end (float, float): The ending coordinates in (i, j) of the cosmic ray trail, in
            units of pix.
        N_i (int): Number of pixels along i-axis of detector.
        N_j (int): Number of pixels along j-axis of detector.
        eps (float): Tiny value used for stable numerical rounding.

    Returns:
        ii (np.ndarray[int]): i-axis positions of traversed trail, in units of pix.
        jj (np.ndarray[int]): j-axis positions of traversed trail, in units of pix.
        lengths (np.ndarray[float]): Chord lengths for each traversed pixel, in units of pix.
    """
    i0, j0 = trail_start
    i1, j1 = trail_end

    # Handle the case of a 0-length trail
    if np.allclose(trail_start, trail_end):
        return (
            np.array([np.round(i0).astype(int)]),
            np.array([np.round(j0).astype(int)]),
            np.array([0]),
        )

    # Ensure that the trail is always moving from left to right
    if i0 > i1:
        i0, j0, i1, j1 = i1, j1, i0, j0

    # Ensure that the trail is always moving from bottom to top
    if j0 > j1:
        i0, j0, i1, j1 = i1, j1, i0, j0

    di = i1 - i0
    dj = j1 - j0

    # Horizontal ticks every half-pixel. i_horiz will not necessarily stick to
    # these half values.
    if dj != 0:
        j_horiz = np.arange(np.round(j0), np.round(j1), np.sign(dj) * 0.5)
        i_horiz = i0 + (di / dj) * (j_horiz - j0)

        cross_horiz = np.transpose([i_horiz, j_horiz])
    else:
        cross_horiz = np.array([])

    # Vertical ticks every half-pixel. j_horiz will not necessarily stick to
    # these half values.
    if di != 0:
        i_vert = np.arange(np.round(i0), np.round(i1), np.sign(di) * 0.5)
        j_vert = j0 + (dj / di) * (i_vert - i0)

        cross_vert = np.transpose([i_vert, j_vert])
    else:
        cross_vert = np.array([])

    # Compute centers of traversed pixels. Uses 'eps' to cover rounding issues when the corner is
    # intersected.
    ii_horiz, jj_horiz = np.round(cross_horiz - np.array([eps, np.sign(dj) * 0.5])).astype(int).T
    ii_vert, jj_vert = np.round(cross_vert - np.array([0.5, np.sign(dj) * eps])).astype(int).T

    # Combine crossings and pixel centers
    crossings = np.vstack((cross_horiz, cross_vert))
    ii = np.concatenate((ii_horiz, ii_vert))
    jj = np.concatenate((jj_horiz, jj_vert))

    # Sort by i axis
    sorted_by_i = np.argsort(crossings[:, 0])
    crossings = crossings[sorted_by_i]
    ii = ii[sorted_by_i]
    jj = jj[sorted_by_i]

    # Add the final pixel center (previous steps ignored the last pixel)
    ii = np.concatenate((ii, [np.round(i1).astype(int)]))
    jj = np.concatenate((jj, [np.round(j1).astype(int)]))

    if len(crossings) == 0:
        # if no crossings , then it’s just the total Euclidean distance
        lengths = np.linalg.norm(np.array([di, dj]), keepdims=1)
    else:
        # otherwise, compute starting, crossing, and ending distances
        first_length = np.linalg.norm(crossings[0] - np.array([i0, j0]), keepdims=1)
        middle_lengths = np.linalg.norm(np.diff(crossings, axis=0), axis=1)
        last_length = np.linalg.norm(np.array([i1, j1]) - crossings[-1], keepdims=1)

        lengths = np.concatenate([first_length, middle_lengths, last_length])

        # Remove 0 - length trails
        positive_lengths = lengths > 0
        lengths = lengths[positive_lengths]
        ii = ii[positive_lengths]
        jj = jj[positive_lengths]

    # remove any pixels that go off the detector
    inside_detector = (ii > -0.5) & (ii < (N_i - 0.5)) & (jj > -0.5) & (jj < (N_j - 0.5))
    ii = ii[inside_detector]
    jj = jj[inside_detector]
    lengths = lengths[inside_detector]

    return ii, jj, lengths


def simulate_crs(
    data: np.ndarray,
    time: float = 1000,
    flux: float = 8,
    area: float = 16.8,
    conversion_factor: float = 0.5,
    pixel_size: float = 10,
    pixel_depth: float = 5,
    rng: np.random.Generator = None,
    seed: int = 47,
) -> np.ndarray:
    """
    Adds CRs to an existing dataset.

    Args:
        data (np.ndarray): The detector image with values in units of electrons.
        time (float): The exposure time, units of s.
        flux (float): Cosmic ray flux, units of cm^-2 s^-1. Default value of 8 is equal to the value
            assumed by the JWST ETC.
        area (float): The area of the WFI detector, units of cm^2.
        conversion_factor (float): The convert from eV to electrons, assumed to be the bandgap
            energy, in units of eV / electrons.
        pixel_size (float): The size of an individual pixel in the detector, units of micron.
        pixel_depth (float): The depth of an individual pixel in the detector, units of micron.
        rng (np.random.Generator): Random number generator to use.
        seed (int): Seed to use for random number generator.

    Returns:
        image (np.ndarray): The detector image, in units of electrons, updated to include all of the
            generated cosmic ray hits.
    """
    rng = get_rng(rng, seed)

    N_i, N_j = data.shape
    N_samples = rng.poisson(flux * area * time)

    cr_i0, cr_j0, cr_angle, cr_length, cr_dEdx = sample_cr_params(
        N_samples, N_i=N_i, N_j=N_j, rng=rng
    )

    cr_length = cr_length / pixel_size
    cr_i1 = (cr_i0 + cr_length * np.cos(cr_angle)).clip(-0.5, N_i + 0.5)
    cr_j1 = (cr_j0 + cr_length * np.sin(cr_angle)).clip(-0.5, N_j + 0.5)

    # go from eV / micron -> electrons / pixel
    cr_counts_per_pix = cr_dEdx * pixel_size / conversion_factor
    for i0, j0, i1, j1, counts_per_pix in zip(cr_i0, cr_j0, cr_i1, cr_j1, cr_counts_per_pix):
        ii, jj, length_2d = traverse((i0, j0), (i1, j1), N_i=N_i, N_j=N_j)
        length_3d = ((pixel_depth / pixel_size) ** 2 + length_2d**2) ** 0.5
        data[ii, jj] += rng.poisson(counts_per_pix * length_3d).astype(np.uint16)

    return data

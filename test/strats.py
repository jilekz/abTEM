import os
import tempfile
import uuid

import hypothesis.strategies as st
import numpy as np
from ase import Atoms

from abtem import FrozenPhonons, Probe
from abtem.measure.detect import AnnularDetector, FlexibleAnnularDetector, SegmentedDetector, PixelatedDetector, \
    WavesDetector
from hypothesis.extra.numpy import arrays, array_shapes


def round_to_multiple(x, base=5):
    return base * round(x / base)


@st.composite
def gpts(draw, min_value=32, max_value=128, allow_none=False, base=None):
    gpts = st.integers(min_value=min_value, max_value=max_value)
    gpts = gpts | st.tuples(gpts, gpts)
    if allow_none:
        gpts = gpts | st.none()
    gpts = st.one_of(gpts)
    gpts = draw(gpts)

    if base is not None:
        if isinstance(gpts, int):
            return round_to_multiple(round_to_multiple(gpts, base))
        else:
            return tuple(round_to_multiple(n, base) for n in gpts)

    return gpts


@st.composite
def sampling(draw, min_value=0.01, max_value=0.1, allow_none=False):
    sampling = st.floats(min_value=min_value, max_value=max_value)
    sampling = sampling | st.tuples(sampling, sampling)
    if allow_none:
        sampling = sampling | st.none()
    sampling = st.one_of(sampling)
    sampling = draw(sampling)
    return sampling


@st.composite
def extent(draw, min_value=1., max_value=10., allow_none=False):
    extent = st.floats(min_value=min_value, max_value=max_value)
    extent = extent | st.tuples(extent, extent)
    if allow_none:
        extent = extent | st.none()
    extent = st.one_of(extent)
    extent = draw(extent)
    return extent


@st.composite
def energy(draw, min_value=80e3, max_value=300e3, allow_none=False):
    energy = st.floats(min_value=min_value, max_value=max_value)
    if allow_none:
        energy = energy | st.none()
    energy = draw(energy)
    return energy


# @st.composite
# def tilt(draw, min_value=0., max_value=10.):
#     energy = st.floats(min_value=min_value, max_value=max_value)
#     energy = energy | st.none()
#     energy = draw(energy)
#     return energy


@st.composite
def empty_atoms_data(draw,
                     min_side_length=1.,
                     max_side_length=5.,
                     min_thickness=.5,
                     max_thickness=5.):
    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))
    return {
        'numbers': [],
        'positions': [],
        'cell': cell
    }


@st.composite
def random_atoms_data(draw,
                      min_side_length=1.,
                      max_side_length=5.,
                      min_thickness=.5,
                      max_thickness=5.,
                      max_atoms=10):
    n = draw(st.integers(1, max_atoms))

    numbers = draw(st.lists(elements=st.integers(min_value=1, max_value=80), min_size=n, max_size=n))

    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))

    position = st.tuples(st.floats(min_value=0, max_value=cell[0]),
                         st.floats(min_value=0, max_value=cell[1]),
                         st.floats(min_value=0, max_value=cell[2]))

    positions = draw(st.lists(elements=position, min_size=n, max_size=n))

    return {
        'numbers': numbers,
        'positions': positions,
        'cell': cell
    }


@st.composite
def random_atoms(draw,
                 min_side_length=1.,
                 max_side_length=5.,
                 min_thickness=.5,
                 max_thickness=5.,
                 max_atoms=10):
    n = draw(st.integers(1, max_atoms))
    numbers = draw(st.lists(elements=st.integers(min_value=1, max_value=80), min_size=n, max_size=n))
    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))
    position = st.tuples(st.floats(min_value=0, max_value=cell[0]),
                         st.floats(min_value=0, max_value=cell[1]),
                         st.floats(min_value=0, max_value=cell[2]))
    positions = draw(st.lists(elements=position, min_size=n, max_size=n))
    return Atoms(numbers=numbers, positions=positions, cell=cell)


@st.composite
def random_frozen_phonons(draw,
                          min_side_length=1.,
                          max_side_length=5.,
                          min_thickness=.5,
                          max_thickness=5.,
                          max_atoms=10,
                          max_configs=2):
    atoms = draw(random_atoms(min_side_length=min_side_length,
                              max_side_length=max_side_length,
                              min_thickness=min_thickness,
                              max_thickness=max_thickness,
                              max_atoms=max_atoms))
    num_configs = draw(st.integers(min_value=1, max_value=max_configs))
    sigmas = draw(st.floats(min_value=0., max_value=.2))
    return FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas, seed=13)





@st.composite
def probe(draw,
          min_gpts=64,
          max_gpts=128,
          max_semiangle_cutoff=30,
          ):
    gpts = draw(gpts(min_value=64, max_value=128))
    semiangle_cutoff = draw(st.floats(min_value=5, max_value=max_semiangle_cutoff))
    return Probe(gpts=gpts, semiangle_cutoff=semiangle_cutoff)


@st.composite
def images(draw):
    sampling = draw(sampling)
    return sampling
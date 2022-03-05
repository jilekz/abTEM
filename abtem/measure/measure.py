import copy
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Union, Tuple, TypeVar, Dict, List, Sequence

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import zarr
from ase import Atom
from matplotlib.axes import Axes

from abtem.core.axes import HasAxes, RealSpaceAxis, AxisMetadata, FourierSpaceAxis, LinearAxis, axis_to_dict, \
    axis_from_dict, OrdinalAxis
from abtem.core.backend import cp, asnumpy, get_array_module, get_ndimage_module, copy_to_device, \
    device_name_from_array_module
from abtem.core.complex import abs2
from abtem.core.dask import HasDaskArray
from abtem.core.energy import energy2wavelength
from abtem.core.fft import fft2, fft2_interpolate
from abtem.core.interpolate import interpolate_bilinear
from abtem.measure.utils import polar_detector_bins, sum_run_length_encoded
from abtem.visualize.utils import domain_coloring, add_domain_coloring_cbar

if cp is not None:
    from abtem.core.cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
    from abtem.core.cuda import interpolate_bilinear as interpolate_bilinear_cuda
else:
    sum_run_length_encoded_cuda = None
    interpolate_bilinear_cuda = None

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from hyperspy._signals.signal2d import Signal2D
        from hyperspy._signals.signal2d import Signal1D
except ImportError:
    Signal2D = None
    Signal1D = None

T = TypeVar('T', bound='AbstractMeasurement')

missing_hyperspy_message = 'This functionality of abTEM requires hyperspy, see https://hyperspy.org/.'


def _to_hyperspy_axes_metadata(axes_metadata, shape):
    hyperspy_axes = []

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    for metadata, n in zip(axes_metadata, shape):
        hyperspy_axes.append({'size': n})

        axes_mapping = {'sampling': 'scale',
                        'units': 'units',
                        'label': 'name',
                        'offset': 'offset'
                        }
        for attr, mapped_attr in axes_mapping.items():
            if hasattr(metadata, attr):
                hyperspy_axes[-1][mapped_attr] = getattr(metadata, attr)

    return hyperspy_axes


def from_zarr(url):
    with zarr.open(url, mode='r') as f:
        d = {}

        for key, value in f.attrs.items():
            if key == 'extra_axes_metadata':
                extra_axes_metadata = [axis_from_dict(d) for d in value]
            elif key == 'type':
                cls = globals()[value]
            else:
                d[key] = value

    array = da.from_zarr(url, component='array', chunks=None)
    return cls(array, extra_axes_metadata=extra_axes_metadata, **d)


def stack_measurements(measurements, axes_metadata):
    xp = get_array_module(measurements[0].array)

    if measurements[0].is_lazy:
        array = da.stack([measurement.array for measurement in measurements])
    else:
        array = xp.stack([measurement.array for measurement in measurements])

    cls = measurements[0].__class__
    d = measurements[0]._copy_as_dict(copy_array=False)
    d['array'] = array
    d['extra_axes_metadata'] = [axes_metadata] + d['extra_axes_metadata']
    return cls(**d)


def _poisson_noise(measurement, pixel_area, dose: float, samples: int = 1, seed: int = None):
    d = measurement._copy_as_dict(copy_array=False)
    xp = get_array_module(measurement.array)

    def add_poisson_noise(array, _):
        return xp.random.poisson(array * dose * pixel_area).astype(xp.float32)

    arrays = []
    for i in range(samples):
        if measurement.is_lazy:
            arrays.append(measurement.array.map_blocks(add_poisson_noise, _=i, dtype=np.float32))
        else:
            arrays.append(add_poisson_noise(measurement.array, i))

    if measurement.is_lazy:
        arrays = da.stack(arrays)
    else:
        arrays = xp.stack(arrays)

    d['array'] = arrays
    d['extra_axes_metadata'] = [OrdinalAxis()] + d['extra_axes_metadata']
    return measurement.__class__(**d)


class AbstractMeasurement(HasDaskArray, HasAxes, metaclass=ABCMeta):

    def __init__(self, array, extra_axes_metadata, metadata, allow_complex=False, allow_base_axis_chunks=False):
        self._array = array

        if extra_axes_metadata is None:
            extra_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._extra_axes_metadata = extra_axes_metadata
        self._metadata = metadata

        super().__init__(array)
        self._check_axes_metadata()

        if len(array.shape) < len(self.base_shape):
            raise RuntimeError(f'array dim smaller than base dim of measurement type {self.__class__}')

        if not allow_complex:
            if np.iscomplexobj(array):
                raise RuntimeError(f'complex dtype not implemented for {self.__class__}')

        if not allow_base_axis_chunks:
            if self.is_lazy and (not all(len(chunks) == 1 for chunks in array.chunks[-2:])):
                raise RuntimeError(f'chunks not allowed in base axes of {self.__class__}')

    @property
    def device(self):
        return device_name_from_array_module(get_array_module(self.array))

    def rechunk(self, **kwargs):
        if not self.is_lazy:
            return self
        self._array = self._array.rechunk(**kwargs)
        return self

    @property
    def energy(self):
        if not 'energy' in self.metadata.keys():
            raise RuntimeError('energy not in measurement metadata')
        return self.metadata['energy']

    @property
    def wavelength(self):
        return energy2wavelength(self.metadata['energy'])

    def scan_positions(self):
        positions = ()
        for n, metadata in zip(self.scan_shape, self.scan_axes_metadata):
            positions += (
                np.linspace(metadata.offset, metadata.offset + metadata.sampling * n, n, endpoint=metadata.endpoint),)
        return positions

    def scan_extent(self):
        extent = ()
        for n, metadata in zip(self.scan_shape, self.scan_axes_metadata):
            extent += (metadata.sampling * n,)
        return extent

    def squeeze(self):
        d = self._copy_as_dict(copy_array=False)
        d['extra_axes_metadata'] = [m for s, m in zip(self.extra_axes_shape, self.extra_axes_metadata) if s != 1]
        d['array'] = self.array[tuple(0 if s == 1 else slice(None) for s in self.extra_axes_shape)]
        return self.__class__(**d)

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list:
        pass

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def base_shape(self) -> Tuple[int, ...]:
        return self.shape[-self.num_base_axes:]

    @property
    def dimensions(self) -> int:
        return len(self._array.shape)

    def _check_is_base_axes(self, axes) -> bool:
        if isinstance(axes, Number):
            axes = (axes,)
        return len(set(axes).intersection(self.base_axes)) > 0

    def __eq__(self, other: 'AbstractMeasurement') -> bool:
        if not isinstance(other, self.__class__):
            return False

        if not self.shape == other.shape:
            return False

        for (key, value), (other_key, other_value) in zip(self._copy_as_dict(copy_array=False).items(),
                                                          other._copy_as_dict(copy_array=False).items()):
            if np.any(value != other_value):
                return False

        if not np.allclose(self.array, other.array):
            return False

        return True

    def check_is_compatible(self, other: 'AbstractMeasurement'):
        if not isinstance(other, self.__class__):
            raise RuntimeError(f'Incompatible measurement types ({self.__class__} is not {other.__class__})')

        if self.shape != other.shape:
            raise RuntimeError()

        for (key, value), (other_key, other_value) in zip(self._copy_as_dict(copy_array=False).items(),
                                                          other._copy_as_dict(copy_array=False).items()):
            if np.any(value != other_value):
                raise RuntimeError(f'{key}, {other_key} {value} {other_value}')

    def relative_difference(self, other, min_relative_tol=0.):
        difference = self.subtract(other)

        xp = get_array_module(self.array)

        # if min_relative_tol > 0.:
        valid = xp.abs(self.array) >= min_relative_tol * self.array.max()
        difference._array[valid] /= self.array[valid]
        difference._array[valid == 0] = 0.
        # else:
        #    difference._array[:] /= self.array

        return difference

    def __mul__(self, other) -> 'T':
        return self._arithmetic(other, '__mul__')

    def __imul__(self, other) -> 'T':
        return self._in_place_arithmetic(other, '__imul__')

    def __truediv__(self, other) -> 'T':
        return self._arithmetic(other, '__truediv__')

    def __itruediv__(self, other) -> 'T':
        return self._arithmetic(other, '__itruediv__')

    def __sub__(self, other) -> 'T':
        return self._arithmetic(other, '__sub__')

    def __isub__(self, other) -> 'T':
        return self._in_place_arithmetic(other, '__isub__')

    def __add__(self, other) -> 'T':
        return self._arithmetic(other, '__add__')

    def __iadd__(self, other) -> 'T':
        return self._in_place_arithmetic(other, '__iadd__')

    __rmul__ = __mul__
    __rtruediv__ = __truediv__

    def _arithmetic(self, other, func) -> 'T':
        self.check_is_compatible(other)
        d = self._copy_as_dict(copy_array=False)
        d['array'] = getattr(self.array, func)(other.array)
        return self.__class__(**d)

    def _in_place_arithmetic(self, other, func) -> 'T':
        if self.is_lazy or other.is_lazy:
            raise RuntimeError('inplace arithmetic operation not implemented for lazy measurement')
        return self._arithmetic(other, func)

    def mean(self, axes, **kwargs) -> 'T':
        return self._reduction('mean', axes=axes, **kwargs)

    def sum(self, axes, **kwargs) -> 'T':
        return self._reduction('sum', axes=axes, **kwargs)

    def std(self, axes, **kwargs) -> 'T':
        return self._reduction('std', axes=axes, **kwargs)

    def _reduction(self, reduction_func, axes, split_every: int = 2) -> 'T':
        if isinstance(axes, Number):
            axes = (axes,)

        axes = tuple(axis if axis >= 0 else len(self) + axis for axis in axes)

        if self._check_is_base_axes(axes):
            raise RuntimeError('base axes cannot be reduced')

        extra_axes_metadata = copy.deepcopy(self._extra_axes_metadata)
        extra_axes_metadata = [axis_metadata for axis_metadata, axis in zip(extra_axes_metadata, self.extra_axes)
                               if axis not in axes]

        d = self._copy_as_dict(copy_array=False)
        if self.is_lazy:
            d['array'] = getattr(da, reduction_func)(self.array, axes, split_every=split_every)
        else:
            xp = get_array_module(self.array)
            d['array'] = getattr(xp, reduction_func)(self.array, axes)

        d['extra_axes_metadata'] = extra_axes_metadata
        return self.__class__(**d)

    def __getitem__(self, items):
        if isinstance(items, (Number, slice)):
            items = (items,)
        # elif isinstance(items, ):
        #    items = (items,)

        removed_axes = []
        for i, item in enumerate(items):
            if isinstance(item, Number):
                removed_axes.append(i)
            elif isinstance(item, (type(...), type(None))):
                raise NotImplementedError

        if self._check_is_base_axes(removed_axes):
            raise RuntimeError('base axes cannot be indexed')

        axes = [element for i, element in enumerate(self.extra_axes_metadata) if not i in removed_axes]

        d = self._copy_as_dict(copy_array=False)
        d['array'] = self._array[items]
        d['extra_axes_metadata'] = axes
        return self.__class__(**d)

    def _apply_element_wise_func(self, func: callable) -> 'T':
        d = self._copy_as_dict(copy_array=False)
        d['array'] = func(self.array)
        return self.__class__(**d)

    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False):
        with zarr.open(url, mode='w') as root:
            if self.device == 'gpu':
                measurement = self.to_cpu()
            else:
                measurement = self

            if not measurement.is_lazy:
                measurement._array = da.from_array(measurement.array)

            # if measurement.is_lazy:
            array = measurement.array.to_zarr(url, compute=compute, component='array', overwrite=overwrite)
            # else:
            #    array = zarr.save(url, array=measurement.array, chunks=None)

            d = measurement._copy_as_dict(copy_array=False)
            for key, value in d.items():
                if key == 'extra_axes_metadata':
                    root.attrs[key] = [axis_to_dict(axis) for axis in value]
                else:
                    root.attrs[key] = value

            root.attrs['type'] = measurement.__class__.__name__
        return array

    @staticmethod
    def from_zarr(url) -> 'T':
        return from_zarr(url)

    @abstractmethod
    def to_hyperspy(self):
        pass

    def to_cpu(self) -> 'T':
        d = self._copy_as_dict(copy_array=False)
        if self.is_lazy:
            d['array'] = self.array.map_blocks(asnumpy)
        else:
            d['array'] = asnumpy(self.array)
        return self.__class__(**d)

    @abstractmethod
    def _copy_as_dict(self, copy_array: bool = True) -> dict:
        pass

    def copy(self) -> 'T':
        return self.__class__(**self._copy_as_dict())


class Images(AbstractMeasurement):

    def __init__(self,
                 array: Union[da.core.Array, np.array],
                 sampling: Union[float, Tuple[float, float]],
                 extra_axes_metadata: List[AxisMetadata] = None,
                 metadata: Dict = None):

        if np.isscalar(sampling):
            sampling = (float(sampling),) * 2
        else:
            sampling = tuple(sampling)

        self._sampling = float(sampling[0]), float(sampling[1])
        super().__init__(array=array, extra_axes_metadata=extra_axes_metadata, metadata=metadata, allow_complex=True,
                         allow_base_axis_chunks=True)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [RealSpaceAxis(label='x', sampling=self.sampling[0], units='Å'),
                RealSpaceAxis(label='y', sampling=self.sampling[0], units='Å')]

    def _check_is_complex(self):
        if not np.iscomplexobj(self.array):
            raise RuntimeError('function not implemented for non-complex image')

    def angle(self):
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).angle)

    def abs(self):
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).abs)

    def intensity(self):
        self._check_is_complex()
        return self._apply_element_wise_func(abs2)

    def integrate_gradient(self):
        self._check_is_complex()
        if self.is_lazy:
            array = self.array.rechunk(self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],)))
            array = array.map_blocks(integrate_gradient_2d, sampling=self.sampling, dtype=np.float32)
        else:
            array = integrate_gradient_2d(self.array, sampling=self.sampling)

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        return self.__class__(**d)

    def to_hyperspy(self):
        if Signal2D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_axes_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.extra_axes_metadata,
            self.extra_axes_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.array, self.extra_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal2D(array, axes=axes_extra[::-1] + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s

    def crop(self, extent: Tuple[float, float], offset: Tuple[float, float] = (0., 0.)):
        offset = (np.round(self.base_shape[0] * offset[0] / self.extent[0]),
                  np.round(self.base_shape[1] * offset[1] / self.extent[1]))
        new_shape = (np.round(self.base_shape[0] * extent[0] / self.extent[0]),
                     np.round(self.base_shape[1] * extent[1] / self.extent[1]))
        new_array = self.array[..., offset[0]:offset[0] + new_shape[0], offset[1]:offset[1] + new_shape[1]]

        d = self._copy_as_dict(copy_array=False)
        d['array'] = new_array
        return self.__class__(**d)

    def _copy_as_dict(self, copy_array: bool = True):
        d = {'sampling': self.sampling,
             'extra_axes_metadata': copy.deepcopy(self.extra_axes_metadata),
             'metadata': copy.deepcopy(self.metadata)}

        if copy_array:
            d['array'] = self.array.copy()
        return d

    @property
    def sampling(self) -> Tuple[float, float]:
        return self._sampling

    @property
    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0., self.shape[-2] * self.sampling[0], self.shape[-2])
        y = np.linspace(0., self.shape[-1] * self.sampling[1], self.shape[-1])
        return x, y

    @property
    def extent(self) -> Tuple[float, float]:
        return self.sampling[0] * self.base_shape[0], self.sampling[1] * self.base_shape[1]

    def interpolate(self,
                    sampling: Union[float, Tuple[float, float]] = None,
                    gpts: Union[int, Tuple[int, int]] = None,
                    method: str = 'fft',
                    boundary: str = 'periodic',
                    normalization: str = 'values') -> 'Images':

        if method != 'fft':
            # TODO: implement quintic interpolation
            raise NotImplementedError

        if method == 'fft' and boundary != 'periodic':
            raise ValueError()

        if sampling is None and gpts is None:
            raise ValueError()

        if gpts is None and sampling is not None:
            if np.isscalar(sampling):
                sampling = (sampling,) * 2
            gpts = tuple(int(np.ceil(l / d)) for d, l in zip(sampling, self.extent))

        elif gpts is not None:
            if np.isscalar(gpts):
                gpts = (gpts,) * 2
        else:
            raise ValueError()

        xp = get_array_module(self.array)

        sampling = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])

        if self.is_lazy:
            array = dask.delayed(fft2_interpolate)(self.array, gpts, normalization=normalization)
            array = da.from_delayed(array, shape=self.shape[:-2] + gpts, meta=xp.array((), dtype=self.array.dtype))
        else:
            array = fft2_interpolate(self.array, gpts, normalization=normalization)

        d = self._copy_as_dict(copy_array=False)
        d['sampling'] = sampling
        d['array'] = array
        return self.__class__(**d)

    def poisson_noise(self, dose: float, samples: int = 1, seed: int = None):
        pixel_area = np.prod(self.sampling)
        return _poisson_noise(self, pixel_area, dose, samples, seed)

    def interpolate_line_at_position(self,
                                     center: Union[Tuple[float, float], Atom],
                                     angle: float,
                                     extent: float,
                                     gpts: int = None,
                                     sampling: float = None,
                                     width: float = 0.,
                                     order: int = 3,
                                     endpoint: bool = True):

        from abtem.waves.scan import LineScan

        scan = LineScan.at_position(position=center, extent=extent, angle=angle)

        return self.interpolate_line(scan.start, scan.end, gpts=gpts, sampling=sampling, width=width, order=order,
                                     endpoint=endpoint)

    def interpolate_line(self,
                         start: Union[Tuple[float, float], Atom] = None,
                         end: Union[Tuple[float, float], Atom] = None,
                         gpts: int = None,
                         sampling: float = None,
                         width: float = 0.,
                         order: int = 3,
                         endpoint: bool = False) -> 'LineProfiles':
        """
        Interpolate image along a line.

        Parameters
        ----------
        start : two float, Atom, optional
            Start point of line [Å]. Also provide end, do not provide center, angle and extent.
        end : two float, Atom, optional
            End point of line [Å].
        gpts : int
            Number of grid points along line.
        sampling : float
            Sampling rate of grid points along line [1 / Å].
        width : float, optional
            The interpolation will be averaged across line of this width.
        order : str, optional
            The interpolation method.

        Returns
        -------
        LineProfiles
        """

        from abtem.waves.scan import LineScan
        map_coordinates = get_ndimage_module(self.array).map_coordinates

        if (sampling is None) and (gpts is None):
            sampling = min(self.sampling)

        xp = get_array_module(self.array)

        if start is None:
            start = (0., 0.)

        if end is None:
            end = (0., self.extent[0])

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        positions = xp.asarray(scan.get_positions(lazy=False) / self.sampling)

        if width:
            direction = xp.array(scan.end) - xp.array(scan.start)
            direction = direction / xp.linalg.norm(direction)
            perpendicular_direction = xp.array([-direction[1], direction[0]])
            n = xp.floor(width / min(self.sampling) / 2) * 2 + 1
            perpendicular_positions = xp.linspace(-n / 2, n / 2, int(n))[:, None] * perpendicular_direction[None]
            positions = perpendicular_positions[None, :] + positions[:, None]
        else:
            positions = positions[:, None]

        def interpolate_1d_from_2d(array, positions):
            positions_shape = positions.shape
            positions = positions.reshape((-1, 2))

            old_shape = array.shape
            array = array.reshape((-1,) + array.shape[-2:])
            array = xp.pad(array, ((0, 0), (3, 3), (3, 3)), mode='wrap')

            positions = (positions + 3.) % xp.array(old_shape[-2:])

            output = xp.zeros((array.shape[0], positions.shape[0]), dtype=np.float32)

            for i in range(array.shape[0]):
                map_coordinates(array[i], positions.T, output=output[i], order=order, mode='wrap')

            output = output.reshape((array.shape[0],) + positions_shape[:-1]).mean(-1)
            output = output.reshape(old_shape[:-2] + (output.shape[-1],))
            return output

        if self.is_lazy:
            array = self.array.map_blocks(interpolate_1d_from_2d,
                                          positions=positions,
                                          drop_axis=self.base_axes,
                                          new_axis=self.base_axes[0],
                                          chunks=self.array.chunks[:-2] + (positions.shape[0],),
                                          meta=xp.array((), dtype=np.float32))
        else:
            array = interpolate_1d_from_2d(self.array, positions)

        return LineProfiles(array=array, start=scan.start, end=scan.end, endpoint=endpoint,
                            extra_axes_metadata=self.extra_axes_metadata, metadata=self.metadata)

    def tile(self, reps: Tuple[int, int]) -> 'Images':
        if len(reps) != 2:
            raise RuntimeError()
        d = self._copy_as_dict(copy_array=False)
        d['array'] = np.tile(self.array, (1,) * (len(self.array.shape) - 2) + reps)
        return self.__class__(**d)

    def gaussian_filter(self, sigma: Union[float, Tuple[float, float]], boundary: str = 'periodic'):
        xp = get_array_module(self.array)
        gaussian_filter = get_ndimage_module(self.array).gaussian_filter

        if boundary == 'periodic':
            mode = 'wrap'
        else:
            raise NotImplementedError()

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        sigma = (0,) * (len(self.shape) - 2) + tuple(s / d for s, d in zip(sigma, self.sampling))

        if self.is_lazy:
            depth = tuple(min(int(np.ceil(4.0 * s)), n) for s, n in zip(sigma, self.base_shape))
            array = self.array.map_overlap(gaussian_filter,
                                           sigma=sigma,
                                           boundary=boundary,
                                           mode=mode,
                                           depth=(0,) * (len(self.shape) - 2) + depth,
                                           meta=xp.array((), dtype=xp.float32))
        else:
            array = gaussian_filter(self.array, sigma=sigma, mode=mode)

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        return self.__class__(**d)

    def diffractograms(self) -> 'DiffractionPatterns':
        xp = get_array_module(self.array)

        def diffractograms(array):
            array = fft2(array)
            return xp.fft.fftshift(xp.abs(array), axes=(-2, -1))

        if self.is_lazy:
            array = self.array.rechunk(self.array.chunks[:-2] + ((self.array.shape[-2],), (self.array.shape[-1],)))
            array = array.map_blocks(diffractograms, meta=xp.array((), dtype=xp.float32))
        else:
            array = diffractograms(self.array)

        sampling = 1 / self.extent[0], 1 / self.extent[1]
        return DiffractionPatterns(array=array,
                                   sampling=sampling,
                                   extra_axes_metadata=self.extra_axes_metadata,
                                   metadata=self.metadata)

    def show(self,
             ax: Axes = None,
             cbar: bool = False,
             figsize: Tuple[int, int] = None,
             title: str = None,
             power: float = 1.,
             vmin: float = None,
             vmax: float = None,
             **kwargs):

        self.compute()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if title is not None:
            ax.set_title(title)

        ax.set_title(title)

        array = asnumpy(self.array)[(0,) * self.num_extra_axes].T ** power

        if np.iscomplexobj(array):
            colored_array = domain_coloring(array, vmin=vmin, vmax=vmax)
        else:
            colored_array = array

        im = ax.imshow(colored_array, extent=[0, self.extent[0], 0, self.extent[1]], origin='lower', vmin=vmin,
                       vmax=vmax, **kwargs)

        ax.set_xlabel('x [Å]')
        ax.set_ylabel('y [Å]')

        if cbar:
            if np.iscomplexobj(array):
                vmin = np.abs(array).min() if vmin is None else vmin
                vmax = np.abs(array).max() if vmax is None else vmax
                add_domain_coloring_cbar(ax, vmin, vmax)
            else:
                plt.colorbar(im, ax=ax)

        return ax, im


class LineProfiles(AbstractMeasurement):

    def __init__(self,
                 array: np.ndarray,
                 start: Tuple[float, float] = None,
                 end: Tuple[float, float] = None,
                 sampling: float = None,
                 endpoint: bool = True,
                 extra_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        from abtem.waves.scan import LineScan

        if start is None:
            start = (0., 0.)

        if end is None:
            end = (start[0] + len(array) * sampling, start[1])

        self._linescan = LineScan(start=start, end=end, gpts=array.shape[-1], endpoint=endpoint)

        super().__init__(array=array, extra_axes_metadata=extra_axes_metadata, metadata=metadata, allow_complex=True,
                         allow_base_axis_chunks=True)

    @property
    def start(self) -> Tuple[float, float]:
        return self._linescan.start

    @property
    def end(self) -> Tuple[float, float]:
        return self._linescan.end

    @property
    def extent(self) -> float:
        return self._linescan.extent

    @property
    def endpoint(self) -> bool:
        return self._linescan.endpoint

    @property
    def sampling(self) -> float:
        return self._linescan.sampling

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [RealSpaceAxis(label='x', sampling=self.sampling, units='Å')]

    def interpolate(self,
                    sampling: float = None,
                    gpts: int = None,
                    order: int = 3,
                    endpoint: bool = False) -> 'LineProfiles':

        map_coordinates = get_ndimage_module(self.array).map_coordinates
        xp = get_array_module(self.array)

        if (gpts is not None) and (sampling is not None):
            raise RuntimeError()

        if sampling is None and gpts is None:
            sampling = self.sampling

        if gpts is None:
            gpts = int(np.ceil(self.extent / sampling))

        if sampling is None:
            sampling = self.extent / gpts

        def interpolate(array, gpts, endpoint, order):
            old_shape = array.shape
            array = array.reshape((-1, array.shape[-1]))

            array = xp.pad(array, ((0,) * 2, (3,) * 2), mode='wrap')
            new_points = xp.linspace(3., array.shape[-1] - 3., gpts, endpoint=endpoint)[None]

            new_array = xp.zeros(array.shape[:-1] + (gpts,), dtype=xp.float32)
            for i in range(len(array)):
                map_coordinates(array[i], new_points, new_array[i], order=order)

            return new_array.reshape(old_shape[:-1] + (gpts,))

        if self.is_lazy:
            array = self.array.rechunk(self.array.chunks[:-1] + ((self.shape[-1],),))
            array = array.map_blocks(interpolate, gpts=gpts, endpoint=endpoint, order=order,
                                     chunks=self.array.chunks[:-1] + ((gpts,)), meta=xp.array((), dtype=xp.float32))
        else:
            array = interpolate(self.array, gpts, endpoint, order)

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        d['sampling'] = sampling
        return self.__class__(**d)

    def tile(self, reps: int):
        d = self._copy_as_dict(copy_array=False)
        xp = get_array_module(self.array)
        d['end'] = np.array(self.start) + (np.array(self.end) - np.array(self.start)) * reps
        reps = (0,) * (len(self.array.shape) - 1) + (reps,)
        if self.is_lazy:
            d['array'] = da.tile(self.array, reps)
        else:
            d['array'] = xp.tile(self.array, reps)
        return self.__class__(**d)

    def to_hyperspy(self):
        if Signal1D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_axes_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.extra_axes_metadata,
            self.extra_axes_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.array, self.extra_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal1D(array, axes=axes_extra[::-1] + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s

    def _copy_as_dict(self, copy_array: bool = True) -> dict:

        d = {'start': self.start,
             'end': self.end,
             'sampling': self.sampling,
             'endpoint': self.endpoint,
             'extra_axes_metadata': copy.deepcopy(self.extra_axes_metadata),
             'metadata': copy.deepcopy(self.metadata)}

        if copy_array:
            d['array'] = self.array.copy()

        return d

    def show(self, ax: Axes = None, title: str = None, label: str = None, **kwargs):
        if ax is None:
            ax = plt.subplot()

        if title is not None:
            ax.set_title(title)

        slic = (0,) * self.num_extra_axes

        array = copy_to_device(self.array[slic], np)

        x = np.linspace(0, self.extent, len(array), endpoint=self._linescan.endpoint)

        if np.iscomplexobj(self.array):
            if label is None:
                label = ''

            ax.plot(x, array.real, label=f'Real {label}', **kwargs)
            ax.plot(x, array.imag, label=f'Imag. {label}', **kwargs)
        else:
            ax.plot(x, array, label=label, **kwargs)

        ax.set_xlabel('x [Å]')
        return ax


def integrate_gradient_2d(gradient, sampling):
    xp = get_array_module(gradient)
    gx, gy = gradient.real, gradient.imag
    (nx, ny) = gx.shape[-2:]
    ikx = xp.fft.fftfreq(nx, d=sampling[0])
    iky = xp.fft.fftfreq(ny, d=sampling[1])
    grid_ikx, grid_iky = xp.meshgrid(ikx, iky, indexing='ij')
    k = grid_ikx ** 2 + grid_iky ** 2
    k[k == 0] = 1e-12
    That = (xp.fft.fft2(gx) * grid_ikx + xp.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
    T = xp.real(xp.fft.ifft2(That))
    T -= xp.min(T)
    return T


def bilinear_nodes_and_weight(old_shape, new_shape, old_angular_sampling, new_angular_sampling, xp):
    nodes = []
    weights = []

    old_sampling = 1 / old_angular_sampling[0] / old_shape[0], 1 / old_angular_sampling[1] / old_shape[1]
    new_sampling = 1 / new_angular_sampling[0] / new_shape[0], 1 / new_angular_sampling[1] / new_shape[1]

    for n, m, r, d in zip(old_shape, new_shape, old_sampling, new_sampling):
        k = xp.fft.fftshift(xp.fft.fftfreq(n, r).astype(xp.float32))
        k_new = xp.fft.fftshift(xp.fft.fftfreq(m, d).astype(xp.float32))
        distances = k_new[None] - k[:, None]
        distances[distances < 0.] = np.inf
        w = distances.min(0) / (k[1] - k[0])
        w[w == np.inf] = 0.
        nodes.append(distances.argmin(0))
        weights.append(w)

    v, u = nodes
    vw, uw = weights
    v, u, vw, uw = xp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])
    return v, u, vw, uw


def _reduced_scanned_images_or_line_profiles(new_array, old_measurement):
    if old_measurement.num_scan_axes not in (1, 2):
        raise RuntimeError(f'no measurement type for {old_measurement.__class__} with {old_measurement.num_scan_axes} '
                           f'scan axes')

    extra_axes_metadata = [element for i, element in enumerate(old_measurement.extra_axes_metadata)
                           if not i in old_measurement.scan_axes]

    if len(old_measurement.scan_axes) == 1:
        start = old_measurement.scan_axes_metadata[0].start
        end = old_measurement.scan_axes_metadata[0].end

        return LineProfiles(new_array,
                            sampling=old_measurement.axes_metadata[old_measurement.scan_axes[0]].sampling,
                            start=start,
                            end=end,
                            extra_axes_metadata=extra_axes_metadata,
                            metadata=old_measurement.metadata)
    else:
        sampling = old_measurement.axes_metadata[old_measurement.scan_axes[0]].sampling, \
                   old_measurement.axes_metadata[old_measurement.scan_axes[1]].sampling
        return Images(new_array,
                      sampling=sampling,
                      extra_axes_metadata=extra_axes_metadata,
                      metadata=old_measurement.metadata)


class DiffractionPatterns(AbstractMeasurement):

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 sampling: Union[float, Tuple[float, float]],
                 fftshift: bool = False,
                 extra_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        if np.isscalar(sampling):
            sampling = (float(sampling),) * 2
        else:
            sampling = tuple(sampling)

        self._fftshift = fftshift
        self._sampling = float(sampling[0]), float(sampling[1])
        super().__init__(array=array, extra_axes_metadata=extra_axes_metadata, metadata=metadata)

    def poisson_noise(self, dose: float, samples: int = 1, seed: int = None, pixel_area: float = None):

        if pixel_area is None and len(self.scan_sampling) == 2:
            pixel_area = np.prod(self.scan_sampling)
        elif pixel_area is None:
            raise RuntimeError()

        return _poisson_noise(self, pixel_area, dose, samples, seed)

    @property
    def base_axes_metadata(self):
        return [FourierSpaceAxis(sampling=self.sampling[0], label='x', units='mrad'),
                FourierSpaceAxis(sampling=self.sampling[1], label='y', units='mrad')]

    def to_hyperspy(self):
        if Signal2D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_axes_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.extra_axes_metadata,
            self.extra_axes_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.array, self.extra_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal2D(array, axes=axes_extra[::-1] + axes_base[::-1])

        s.set_signal_type('electron_diffraction')
        for axis in s.axes_manager.signal_axes:
            axis.offset = -int(axis.size / 2) * axis.scale
        if self.is_lazy:
            s = s.as_lazy()

        return s

    def _copy_as_dict(self, copy_array: bool = True) -> dict:
        d = {'sampling': self.sampling,
             'extra_axes_metadata': copy.deepcopy(self.extra_axes_metadata),
             'metadata': copy.deepcopy(self.metadata),
             'fftshift': self.fftshift}

        if copy_array:
            d['array'] = self.array.copy()
        return d

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def sampling(self) -> Tuple[float, float]:
        return self._sampling

    @property
    def angular_sampling(self) -> Tuple[float, float]:
        return self.sampling[0] * self.wavelength * 1e3, self.sampling[1] * self.wavelength * 1e3

    @property
    def max_angles(self):
        return self.shape[-2] // 2 * self.angular_sampling[0], self.shape[-1] // 2 * self.angular_sampling[1]

    @property
    def equivalent_real_space_extent(self):
        return 1 / self.sampling[0], 1 / self.sampling[1]

    @property
    def equivalent_real_space_sampling(self):
        return 1 / self.sampling[0] / self.base_shape[0], 1 / self.sampling[1] / self.base_shape[1]

    @property
    def scan_extent(self):
        extent = ()
        for d, n in zip(self.scan_sampling, self.scan_shape):
            extent += (d * n,)
        return extent

    @property
    def limits(self) -> List[Tuple[float, float]]:
        limits = []
        for i in (-2, -1):
            if self.shape[i] % 2:
                limits += [(-(self.shape[i] - 1) // 2 * self.sampling[i], (self.shape[i] - 1) // 2 * self.sampling[i])]
            else:
                limits += [(-self.shape[i] // 2 * self.sampling[i], (self.shape[i] // 2 - 1) * self.sampling[i])]
        return limits

    @property
    def angular_limits(self) -> List[Tuple[float, float]]:
        limits = self.limits
        limits[0] = limits[0][0] * self.wavelength * 1e3, limits[0][1] * self.wavelength * 1e3
        limits[1] = limits[1][0] * self.wavelength * 1e3, limits[1][1] * self.wavelength * 1e3
        return limits

    def interpolate(self, new_sampling: Union[str, float, Tuple[float, float]]):

        if new_sampling == 'uniform':
            scale_factor = self.sampling[0] / max(self.sampling), self.sampling[1] / max(self.sampling)

            new_gpts = (int(np.ceil(self.base_shape[0] * scale_factor[0])),
                        int(np.ceil(self.base_shape[1] * scale_factor[1])))

            if np.abs(new_gpts[0] - new_gpts[1]) <= 2:
                new_gpts = (min(new_gpts),) * 2

            new_sampling = self.sampling[0] / scale_factor[0], self.sampling[1] / scale_factor[1]

        else:
            raise RuntimeError('')

        def interpolate(array, sampling, new_sampling, new_gpts):
            xp = get_array_module(array)
            v, u, vw, uw = bilinear_nodes_and_weight(array.shape[-2:],
                                                     new_gpts,
                                                     sampling,
                                                     new_sampling,
                                                     xp)

            old_shape = array.shape
            array = array.reshape((-1,) + array.shape[-2:])

            old_sums = array.sum((-2, -1))

            if xp is cp:
                array = interpolate_bilinear_cuda(array, v, u, vw, uw)
            else:
                array = interpolate_bilinear(array, v, u, vw, uw)

            array = array / array.sum((-2, -1)) * old_sums

            array = array.reshape(old_shape[:-2] + array.shape[-2:])
            return array

        if self.is_lazy:
            array = self.array.map_blocks(interpolate,
                                          sampling=self.sampling,
                                          new_sampling=new_sampling,
                                          new_gpts=new_gpts,
                                          chunks=self.array.chunks[:-2] + ((new_gpts[0],), (new_gpts[1],)),
                                          dtype=np.float32)
        else:
            array = interpolate(self.array, sampling=self.sampling, new_sampling=new_sampling, new_gpts=new_gpts)

        d = self._copy_as_dict(copy_array=False)
        d['sampling'] = new_sampling
        d['array'] = array
        return self.__class__(**d)

    def _check_integration_limits(self, inner, outer):
        if inner >= outer:
            raise RuntimeError(f'inner detection ({inner} mrad) angle exceeds outer detection angle'
                               f'({outer} mrad)')

        if (outer > self.max_angles[0]) or (outer > self.max_angles[1]):
            raise RuntimeError(
                f'outer integration limit exceeds the maximum simulated angle ({outer} mrad > '
                f'{min(self.max_angles)} mrad)')

        integration_range = outer - inner
        if integration_range < min(self.angular_sampling):
            raise RuntimeError(
                f'integration range ({integration_range} mrad) smaller than angular sampling of simulation'
                f' ({min(self.angular_sampling)} mrad)')

    def gaussian_source_size(self, sigma: Union[float, Tuple[float, float]]):
        if self.num_scan_axes < 2:
            raise RuntimeError(
                'gaussian_source_size not implemented for DiffractionPatterns with less than 2 scan axes')

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        xp = get_array_module(self.array)
        gaussian_filter = get_ndimage_module(self._array).gaussian_filter

        scan_axes = self.scan_axes

        padded_sigma = ()
        depth = ()
        i = 0
        for axis, n in zip(self.extra_axes, self.extra_axes_shape):
            if axis in scan_axes:
                scan_sampling = self.scan_sampling[i]
                padded_sigma += (sigma[i] / scan_sampling,)
                depth += (min(int(np.ceil(4.0 * sigma[i] / scan_sampling)), n),)
                i += 1
            else:
                padded_sigma += (0.,)
                depth += (0,)

        padded_sigma += (0.,) * 2
        depth += (0,) * 2

        if self.is_lazy:
            array = self.array.map_overlap(gaussian_filter,
                                           sigma=padded_sigma,
                                           boundary=0.,
                                           depth=depth,
                                           meta=xp.array((), dtype=xp.float32))
        else:
            array = gaussian_filter(self.array, sigma=padded_sigma)

        return self.__class__(array,
                              sampling=self.sampling,
                              extra_axes_metadata=self.extra_axes_metadata,
                              metadata=self.metadata,
                              fftshift=self.fftshift)

    def polar_binning(self,
                      nbins_radial: int,
                      nbins_azimuthal: int,
                      inner: float = 0.,
                      outer: float = None,
                      rotation: float = 0.):

        if outer is None:
            outer = min(self.max_angles)

        self._check_integration_limits(inner, outer)
        xp = get_array_module(self.array)

        def radial_binning(array, nbins_radial, nbins_azimuthal):
            xp = get_array_module(array)

            indices = polar_detector_bins(gpts=array.shape[-2:],
                                          sampling=self.angular_sampling,
                                          inner=inner,
                                          outer=outer,
                                          nbins_radial=nbins_radial,
                                          nbins_azimuthal=nbins_azimuthal,
                                          fftshift=self.fftshift,
                                          rotation=rotation,
                                          return_indices=True)

            separators = xp.concatenate((xp.array([0]), xp.cumsum(xp.array([len(i) for i in indices]))))

            new_shape = array.shape[:-2] + (nbins_radial, nbins_azimuthal)

            array = array.reshape((-1, array.shape[-2] * array.shape[-1],))[..., np.concatenate(indices)]

            result = xp.zeros((array.shape[0], len(indices),), dtype=xp.float32)

            if xp is cp:
                sum_run_length_encoded_cuda(array, result, separators)

            else:
                sum_run_length_encoded(array, result, separators)

            return result.reshape(new_shape)

        if self.is_lazy:
            array = self.array.map_blocks(radial_binning, nbins_radial=nbins_radial,
                                          nbins_azimuthal=nbins_azimuthal,
                                          drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                          chunks=self.array.chunks[:-2] + ((nbins_radial,), (nbins_azimuthal,),),
                                          new_axis=(len(self.shape) - 2, len(self.shape) - 1,),
                                          meta=xp.array((), dtype=xp.float32))
        else:
            array = radial_binning(self.array, nbins_radial=nbins_radial, nbins_azimuthal=nbins_azimuthal)

        radial_sampling = (outer - inner) / nbins_radial
        azimuthal_sampling = 2 * np.pi / nbins_azimuthal

        return PolarMeasurements(array,
                                 radial_sampling=radial_sampling,
                                 azimuthal_sampling=azimuthal_sampling,
                                 radial_offset=inner,
                                 azimuthal_offset=rotation,
                                 extra_axes_metadata=self.extra_axes_metadata,
                                 metadata=self.metadata)

    def radial_binning(self, step_size: float = 1., inner: float = 0., outer: float = None):
        if outer is None:
            outer = min(self.max_angles)

        nbins_radial = int((outer - inner) / step_size)
        return self.polar_binning(nbins_radial, 1, inner, outer)

    def integrate_radial(self, inner: float, outer: float):
        self._check_integration_limits(inner, outer)

        xp = get_array_module(self.array)

        def integrate_fourier_space(array, sampling):

            bins = polar_detector_bins(gpts=array.shape[-2:],
                                       sampling=sampling,
                                       inner=inner,
                                       outer=outer,
                                       nbins_radial=1,
                                       nbins_azimuthal=1,
                                       fftshift=self.fftshift)

            xp = get_array_module(array)
            bins = xp.asarray(bins, dtype=xp.float32)

            return xp.sum(array * (bins == 0), axis=(-2, -1))

        if self.is_lazy:
            integrated_intensity = self.array.map_blocks(integrate_fourier_space,
                                                         sampling=self.angular_sampling,
                                                         drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                                         meta=xp.array((), dtype=xp.float32))
        else:
            integrated_intensity = integrate_fourier_space(self.array, sampling=self.angular_sampling)

        return _reduced_scanned_images_or_line_profiles(integrated_intensity, self)

    def integrated_center_of_mass(self) -> Images:
        if len(self.scan_sampling) != 2:
            raise RuntimeError(f'integrated center of mass not implemented for DiffractionPatterns with '
                               f'{self.num_scan_axes} scan axes')

        return self.center_of_mass().integrate_gradient()

    def center_of_mass(self) -> Images:

        def com(array):
            x, y = self.angular_coordinates()
            com_x = (array * x[:, None]).sum(axis=(-2, -1))
            com_y = (array * y[None]).sum(axis=(-2, -1))
            com = com_x + 1.j * com_y
            return com

        if self.is_lazy:
            array = self.array.map_blocks(com, drop_axis=self.base_axes, dtype=np.complex64)
        else:
            array = com(self.array)

        return _reduced_scanned_images_or_line_profiles(array, self)

    def angular_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        xp = get_array_module(self.array)
        limits = self.angular_limits
        alpha_x = xp.linspace(limits[0][0], limits[0][1], self.shape[-2], dtype=xp.float32)
        alpha_y = xp.linspace(limits[1][0], limits[1][1], self.shape[-1], dtype=xp.float32)
        return alpha_x, alpha_y

    def bandlimit(self, inner: float, outer: float) -> 'DiffractionPatterns':

        def bandlimit(array, inner, outer):
            alpha_x, alpha_y = self.angular_coordinates()
            alpha = alpha_x[:, None] ** 2 + alpha_y[None] ** 2
            block = (alpha >= inner ** 2) * (alpha < outer ** 2)
            return array * block

        xp = get_array_module(self.array)

        if self.is_lazy:
            print(self.array.chunks)
            array = self.array.map_blocks(bandlimit, inner=inner, outer=outer, meta=xp.array((), dtype=xp.float32))
        else:
            array = bandlimit(self.array, inner, outer)

        d = self._copy_as_dict(copy_array=False)
        d['array'] = array
        return self.__class__(**d)

    def block_direct(self, radius: float = None) -> 'DiffractionPatterns':
        if radius is None:
            if 'semiangle_cutoff' in self.metadata.keys():
                radius = self.metadata['semiangle_cutoff']
            else:
                radius = max(self.angular_sampling) * 1.0001

        return self.bandlimit(radius, outer=np.inf)

    def show(self,
             ax: Axes = None,
             cbar: bool = False,
             power: float = 1.,
             title: str = None,
             figsize: Tuple[float, float] = None,
             angular_units: bool = False,
             max_angle: float = None,
             **kwargs):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if title is not None:
            ax.set_title(title)

        if angular_units:
            ax.set_xlabel('Scattering angle x [mrad]')
            ax.set_ylabel('Scattering angle y [mrad]')
            extent = self.angular_limits[0] + self.angular_limits[1]
        else:
            ax.set_xlabel('Spatial frequency x [1 / Å]')
            ax.set_ylabel('Spatial frequency y [1 / Å]')
            extent = self.limits[0] + self.limits[1]

        slic = (0,) * self.num_extra_axes

        array = asnumpy(self.array)[slic].T ** power

        im = ax.imshow(array, extent=extent, origin='lower', **kwargs)

        if max_angle:
            ax.set_xlim([-max_angle, max_angle])
            ax.set_ylim([-max_angle, max_angle])

        if cbar:
            plt.colorbar(im, ax=ax)

        return ax, im


class PolarMeasurements(AbstractMeasurement):

    def __init__(self,
                 array: np.ndarray,
                 radial_sampling: float,
                 azimuthal_sampling: float,
                 radial_offset: float = 0.,
                 azimuthal_offset: float = 0.,
                 extra_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        self._radial_sampling = radial_sampling
        self._azimuthal_sampling = azimuthal_sampling
        self._radial_offset = radial_offset
        self._azimuthal_offset = azimuthal_offset

        super().__init__(array=array, extra_axes_metadata=extra_axes_metadata, metadata=metadata)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [LinearAxis(label='Radial scattering angle', sampling=self.radial_sampling, units='mrad'),
                LinearAxis(label='Azimuthal scattering angle', sampling=self.azimuthal_sampling, units='rad')]

    def to_hyperspy(self):
        if Signal2D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_axes_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.extra_axes_metadata,
            self.extra_axes_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.array, self.extra_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal2D(array, axes=axes_extra[::-1] + axes_base[::-1]).squeeze()

        if self.is_lazy:
            s = s.as_lazy()

        return s

    @property
    def radial_offset(self) -> float:
        return self._radial_offset

    @property
    def outer_angle(self) -> float:
        return self._radial_offset + self.radial_sampling * self.shape[-2]

    @property
    def radial_sampling(self) -> float:
        return self._radial_sampling

    @property
    def azimuthal_sampling(self) -> float:
        return self._azimuthal_sampling

    @property
    def azimuthal_offset(self) -> float:
        return self._azimuthal_offset

    def integrate_radial(self, inner: float, outer: float) -> Union[Images, LineProfiles]:
        return self.integrate(radial_limits=(inner, outer))

    def integrate(self,
                  radial_limits: Tuple[float, float] = None,
                  azimuthal_limits: Tuple[float, float] = None,
                  detector_regions: Sequence[int] = None) -> Union[Images, LineProfiles]:

        if detector_regions is not None:
            if (radial_limits is not None) or (azimuthal_limits is not None):
                raise RuntimeError()

            array = self.array.reshape(self.shape[:-2] + (-1,))[..., list(detector_regions)].sum(axis=-1)
        else:
            if radial_limits is None:
                radial_slice = slice(None)
            else:
                inner_index = int((radial_limits[0] - self.radial_offset) / self.radial_sampling)
                outer_index = int((radial_limits[1] - self.radial_offset) / self.radial_sampling)
                radial_slice = slice(inner_index, outer_index)

            if azimuthal_limits is None:
                azimuthal_slice = slice(None)
            else:
                left_index = int(azimuthal_limits[0] / self.radial_sampling)
                right_index = int(azimuthal_limits[1] / self.radial_sampling)
                azimuthal_slice = slice(left_index, right_index)

            array = self.array[..., radial_slice, azimuthal_slice].sum(axis=(-2, -1))

        return _reduced_scanned_images_or_line_profiles(array, self)

    def _copy_as_dict(self, copy_array: bool = True) -> dict:
        d = {'radial_offset': self.radial_offset,
             'radial_sampling': self.radial_sampling,
             'azimuthal_offset': self.azimuthal_offset,
             'azimuthal_sampling': self.azimuthal_sampling,
             'extra_axes_metadata': copy.deepcopy(self.extra_axes_metadata),
             'metadata': copy.deepcopy(self.metadata)}

        if copy_array:
            d['array'] = self.array.copy()
        return d

    def show(self, ax: Axes = None, title: str = None, min_azimuthal_division: float = np.pi / 20, grid: bool = False,
             **kwargs):

        if ax is None:
            ax = plt.subplot(projection="polar")

        if title is not None:
            ax.set_title(title)

        array = self.array[(0,) * (len(self.shape) - 2)]

        repeat = int(self.azimuthal_sampling / min_azimuthal_division)
        r = np.pi / (4 * repeat) + self.azimuthal_offset
        azimuthal_grid = np.linspace(r, 2 * np.pi + r, self.shape[-1] * repeat, endpoint=False)

        d = (self.outer_angle - self.radial_offset) / 2 / self.shape[-2]
        radial_grid = np.linspace(self.radial_offset + d, self.outer_angle - d, self.shape[-2])

        z = np.repeat(array, repeat, axis=-1)
        r, th = np.meshgrid(radial_grid, azimuthal_grid)

        im = ax.pcolormesh(th, r, z.T, shading='auto', **kwargs)
        ax.set_rlim([0, self.outer_angle * 1.1])

        if grid:
            ax.grid()

        return ax, im
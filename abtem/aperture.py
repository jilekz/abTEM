import numpy as np

from abtem.base_classes import Accelerator, HasAcceleratorMixin
from abtem.device import get_array_module


class BullseyeAperture(HasAcceleratorMixin):

    def __init__(self, outer_angle, energy=None, inner_angle=0., num_radials=0, cross=0., rotation=0.):
        self._outer_angle = outer_angle
        self._inner_angle = inner_angle
        self._num_radials = num_radials
        self._rotation = rotation
        self._cross = cross
        self._accelerator = Accelerator(energy=energy)

    def evaluate(self, alpha, phi):
        xp = get_array_module(alpha)

        aperture = xp.ones_like(alpha)

        alpha = alpha * 1000

        aperture[alpha < self._inner_angle] = 0.
        aperture[alpha > self._outer_angle] = 0.

        if self._num_radials > 0:
            edges = np.linspace(self._inner_angle, self._outer_angle, (self._num_radials + 1) * 2)

            start_edges = [edge for i, edge in enumerate(edges[:-1]) if i % 2]
            end_edges = [edge for i, edge in enumerate(edges[1:-1]) if i % 2]

            for start_edge, end_edge in zip(start_edges, end_edges):
                aperture[(alpha > start_edge) * (alpha < end_edge)] = 0.

        if self._cross > 0.:
            d = np.abs(np.sin(phi - self._rotation) * alpha)
            aperture[(d < self._cross / 2) * (alpha < self._outer_angle)] = 1.

            d = np.abs(np.sin(phi - self._rotation - np.pi / 2) * alpha)
            aperture[(d < self._cross / 2) * (alpha < self._outer_angle)] = 1.

        return aperture


class DeviatedAperture(HasAcceleratorMixin):
    def __init__(self, aperture_angle, energy=None, x_0=0, y_0=0):
        #x_0 - aperture deviation from center in x direction [mrad]
        #y_0 - aperture deviation from center in y direction [mrad]
        self._aperture_angle = aperture_angle*1e-3 #[rad]
        self._accelerator = Accelerator(energy=energy)
        self._x_0 = x_0*1e-3 #[rad]
        self._y_0 = y_0*1e-3 #[rad]

    def evaluate(self, alpha, phi): # alpha [rad] phi [rad]
        xp = get_array_module(alpha)

        if type(phi) == type(None):
                phi=xp.array([0])

        aperture = xp.ones_like(alpha)
        
        aperture[ (alpha**2+self._x_0**2+self._y_0**2-2*alpha*(xp.cos(phi)*self._x_0+xp.sin(phi)*self._y_0) ) >self._aperture_angle**2 ] = 0.
        
        return aperture

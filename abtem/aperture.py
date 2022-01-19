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

class DeviatedAnularAperture(HasAcceleratorMixin):
    def __init__(self, aperture_angle, energy=None, x_0=0, y_0=0,inner_aperture_angle=0, spokes = 3, spoke_thickness=1):
        #x_0 - aperture deviation from center in x direction [mrad]
        #y_0 - aperture deviation from center in y direction [mrad]
        #spoke_thickness - thickness of spokes in [mrad]
        #spokes - number of spokes 
        self._aperture_angle = aperture_angle*1e-3 #[rad]
        self._inner_aperture_angle = inner_aperture_angle*1e-3 #[rad]
        self._accelerator = Accelerator(energy=energy)
        self._x_0 = x_0*1e-3 #[rad]
        self._y_0 = y_0*1e-3 #[rad]
        self._spokes = spokes #num of spokes
        self._spoke_thickness = spoke_thickness*1e-3 #[rad]

    def evaluate(self, alpha, phi): # alpha [rad] phi [rad]
        xp = get_array_module(alpha)

        if type(phi) == type(None):
                phi=xp.array([0])

        aperture = xp.ones_like(alpha)

        x = xp.cos(phi)*alpha-self._x_0
        y = xp.sin(phi)*alpha-self._y_0

        r_sq=x**2+y**2

        aperture[ np.logical_or( r_sq > self._aperture_angle**2 , r_sq < self._inner_aperture_angle**2)  ] = 0
        if self._inner_aperture_angle > 0:
            aperture[ r_sq < self._inner_aperture_angle**2 ] = 0
        if self._spokes > 0 and self._spoke_thickness > 0:
            for index_spoke in range(self._spokes):
                phi_spoke = 2*xp.pi/self._spokes * index_spoke
                
                vec = xp.array([xp.cos(phi_spoke),xp.sin(phi_spoke)]) #vec is line intercepting (0,0) and having angle phi_spoke
                normal = xp.array([xp.cos(xp.pi/2+phi_spoke),xp.sin(xp.pi/2+phi_spoke)]) #normal to line along phi_spoke angle (along vec)

                a = normal[0] 
                b = normal[1]
                d = xp.abs(a*x+b*y)/xp.sqrt(a**2+b**2) # dist of a point from a line 

                shape_orig = x.shape
                shape_new = x.size
                x_r= x.reshape(shape_new)
                y_r = y.reshape(shape_new)

                dot_product = np.sum(vec*np.array([x_r,y_r]).T,axis=1)
                dot_product = dot_product.T
                dot_product = dot_product.reshape(shape_orig)

                aperture[ np.logical_and(d < self._spoke_thickness / 2 , dot_product >= 0 ) ] = 0
        
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

class MultipleDeviatedApertures(HasAcceleratorMixin):
    def __init__(self, aperture_angle, energy=None, x_0=np.array([0]), y_0=np.array([0])):
        #x_0 - aperture deviation from center in x direction [mrad]
        #y_0 - aperture deviation from center in y direction [mrad]
        self._aperture_angle = aperture_angle*1e-3 #[rad]
        self._accelerator = Accelerator(energy=energy)
        self._x_0 = np.array(x_0) * 1e-3 #[rad]
        self._y_0 = np.array(y_0) * 1e-3 #[rad]
        
        assert np.shape(x_0) == np.shape(y_0)
        


    def evaluate(self, alpha, phi): # alpha [rad] phi [rad]
        xp = get_array_module(alpha)

        if type(phi) == type(None):
                phi=xp.array([0])

        aperture = xp.zeros_like(alpha)

        for i in range(len(self._x_0)):
            tmp_aperture = DeviatedAperture(self._aperture_angle/1e-3,x_0=self._x_0[i]/1e-3,y_0=self._y_0[i]/1e-3).evaluate(alpha,phi)
            aperture = xp.logical_or(aperture,tmp_aperture)
        
        return aperture

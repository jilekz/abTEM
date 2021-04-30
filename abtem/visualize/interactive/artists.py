import numpy as np
from bqplot import LinearScale, ColorScale, Lines, Scatter, ColorAxis, Figure, Axis
from bqplot_image_gl import ImageGL
from traitlets import HasTraits, observe, default, List, link, Float, Unicode
from traittypes import Array
import ipywidgets as widgets


class ColorBar(widgets.HBox):
    min = Float(0.)
    max = Float(1.)
    label = Unicode()

    def __init__(self, color_scale, label='', *args, **kwargs):
        self._x_scale = LinearScale(allow_padding=False)

        scales = {'x': self._x_scale,
                  'y': LinearScale(allow_padding=False, orientation='vertical'),
                  'image': color_scale}

        self._mark = ImageGL(image=np.linspace(0, 1, 500)[None], scales=scales)
        self._x_axis = Axis(scale=scales['x'], label=label)

        self._x_axis.tick_format = '.2e'
        self._x_axis.num_ticks = 5

        figure = Figure(scales=scales,
                        layout=widgets.Layout(height='80px'),
                        axes=[self._x_axis],
                        fig_margin={'top': 0, 'bottom': 50, 'left': 50, 'right': 10})

        figure.marks = [self._mark]

        super().__init__(children=[figure])

    @observe('label')
    def _observe_label(self, change):
        self._x_axis.label = change['new']

    @observe('min')
    def _observe_min(self, change):
        self._mark.image = np.linspace(change['new'], self.max, 100)[None]
        self._x_scale.min = change['new']
        self._mark.x = [change['new'], self.max]

    @observe('max')
    def _observe_min(self, change):
        self._mark.image = np.linspace(self.min, change['new'], 100)[None]
        self._x_scale.max = change['new']
        self._mark.x = [self.min, change['new']]


class ImageArtist(HasTraits):
    image = Array()
    extent = List(allow_none=True)
    power = Float(1.)

    def __init__(self, **kwargs):
        self._color_scale = ColorScale(colors=['black', 'white'], min=0, max=1)
        self._color_bar = ColorBar(color_scale=self._color_scale)

        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'),
                  'image': self._color_scale}

        self._mark = ImageGL(image=np.zeros((0, 0)), scales=scales)
        super().__init__(**kwargs)

    @property
    def power_scale_slider(self):
        slider = widgets.FloatSlider(description='Power', min=1e-3, max=1, value=1, step=1e-3, )
        slider.observe(self._observe_image, 'value')
        link((slider, 'value'), (self, 'power'))
        return slider

    @property
    def color_scheme_picker(self):
        def update_scheme(change):
            if change['new'] == 'Greys':
                self._color_scale.colors = ['Black', 'White']
            else:
                self._color_scale.colors = []
                self._color_scale.scheme = change['new']

        schemes = ['Greys', 'viridis', 'inferno', 'plasma', 'magma', 'Spectral', 'RdBu']
        dropdown = widgets.Dropdown(description='Scheme', options=schemes)
        dropdown.observe(update_scheme, 'value')
        return dropdown

    @property
    def color_bar(self):
        return self._color_bar

    @default('extent')
    def _default_extent(self):
        return None

    def _add_to_canvas(self, canvas):
        scales = {'x': canvas.figure.axes[0].scale,
                  'y': canvas.figure.axes[1].scale,
                  'image': self._color_scale}

        self._mark.scales = scales
        canvas.figure.marks = [self._mark] + canvas.figure.marks

    @observe('image')
    def _observe_image(self, change):
        image = self.image ** self.power

        if self.extent is None:
            with self._mark.hold_sync():
                self._mark.x = [-.5, image.shape[0] - .5]
                self._mark.y = [-.5, image.shape[1] - .5]
                self._mark.image = image.T
        else:
            self._mark.image = image.T

        if not self.image.size == 0:
            self._mark.scales['image'].min = float(image.min())
            self._mark.scales['image'].max = float(image.max())
            self._color_bar.min = float(image.min())
            self._color_bar.max = float(image.max())

    @observe('extent')
    def _observe_extent(self, change):
        if self.image.size == 0:
            return

        sampling = ((change['new'][0][1] - change['new'][0][0]) / self.image.shape[0],
                    (change['new'][1][1] - change['new'][1][0]) / self.image.shape[1])

        self._mark.x = [value - .5 * sampling[0] for value in change['new'][0]]
        self._mark.y = [value - .5 * sampling[1] for value in change['new'][1]]

    @property
    def display_sampling(self):
        if self.image.size == 0:
            return

        extent_x = self._mark.x[1] - self._mark.x[0]
        extent_y = self._mark.y[1] - self._mark.y[0]
        pixel_extent_x = self.image.shape[0]
        pixel_extent_y = self.image.shape[1]
        return (extent_x / pixel_extent_x, extent_y / pixel_extent_y)

    def position_to_index(self, position):
        sampling = self.display_sampling
        px = int(np.round((position[0] - self._mark.x[0]) / sampling[0] - .5))
        py = int(np.round((position[1] - self._mark.y[0]) / sampling[1] - .5))
        return [px, py]

    def indices_to_position(self, indices):
        sampling = self.display_sampling
        px = (indices[0] + .5) * sampling[0] + self._mark.x[0]
        py = (indices[1] + .5) * sampling[1] + self._mark.y[0]
        return [px, py]

    @property
    def limits(self):
        if (self.extent is None) or (self.display_sampling is None):
            return [(-.5, self.image.shape[1] - .5), (-.5, self.image.shape[0] - .5)]

        return [tuple([l - .5 * s for l in L]) for L, s in zip(self.extent, self.display_sampling)]


class Artist1d(HasTraits):
    x = Array()
    y = Array()

    def __init__(self, mark, **kwargs):
        self._mark = mark
        super().__init__(**kwargs)

    def _add_to_canvas(self, canvas):
        scales = {'x': canvas.figure.axes[0].scale,
                  'y': canvas.figure.axes[1].scale}

        self._mark.scales = scales
        canvas.figure.marks = [self._mark] + canvas.figure.marks


class ScatterArtist(Artist1d):
    x = Array()
    y = Array()

    def __init__(self, **kwargs):
        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'), }
        mark = Scatter(x=np.zeros((1,)), y=np.zeros((1,)), scales=scales, colors=['red'])

        link((mark, 'x'), (self, 'x'))
        link((mark, 'y'), (self, 'y'))
        super().__init__(mark=mark, **kwargs)


class LinesArtist(Artist1d):

    def __init__(self, **kwargs):
        scales = {'x': LinearScale(allow_padding=False),
                  'y': LinearScale(allow_padding=False, orientation='vertical'), }
        mark = Lines(x=np.zeros((1,)), y=np.zeros((1,)), scales=scales, colors=['red'])

        link((mark, 'x'), (self, 'x'))
        link((mark, 'y'), (self, 'y'))
        super().__init__(mark=mark, **kwargs)
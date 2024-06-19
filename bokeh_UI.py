import numpy as np
# import pandas as pd

from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, Select, ColumnDataSource, CustomJS, CrosshairTool, Div, FileInput
from bokeh.plotting import figure
from bokeh.plotting.contour import contour_data
from bokeh.palettes import Viridis256

import io
import base64
import os

import processing as pro

class CreateGUI:
    def __init__(self):
        """
        Create a BokehJS server for preprocessing and postprocessing PDMAC data.
        """
        self.range_const = 0

        # Description
        desc = Div(text=open('description.html').read(), sizing_mode='stretch_width')
        # Text input
        self.input_OD = TextInput(title="Outside Diameter (in)")
        self.input_WT = TextInput(title="Wall Thickness (in)")
        self.input_SMYS = TextInput(title="SMYS (psi)")
        self.input_ILI_format = TextInput(title="ILI Format", value='southern')

        # File input
        self.button_select_file = FileInput(title="Select File", accept='.csv, .xlsx')
        self.button_select_file.on_change('filename', self._file_callback)

        # # File selection Data Source
        # self.file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})
        # self.file_source.on_change('data', self._file_callback)

        # # Create a CustomJS callback to handle file selection
        # select_file = CustomJS(args=dict(file_source=self.file_source), code = """
        #                 function read_file(filename) {
        #                     var reader = new FileReader();
        #                     reader.onload = load_handler;
        #                     reader.onerror = error_handler;
        #                     // readAsDataURL represents the file's data as a base64 encoded string
        #                     reader.readAsDataURL(filename);
        #                 }

        #                 function load_handler(event) {
        #                     var b64string = event.target.result;
        #                     file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
        #                     file_source.trigger("change");
        #                 }

        #                 function error_handler(evt) {
        #                     if(evt.target.error.name == "NotReadableError") {
        #                         alert("Can't read file!");
        #                     }
        #                 }

        #                 var input = document.createElement('input');
        #                 input.setAttribute('type', 'file');
        #                 input.onchange = function(){
        #                     if (window.FileReader) {
        #                         read_file(input.files[0]);
                            
        #                     } else {
        #                         alert('FileReader is not supported in this browser');
        #                     }
        #                 }
        #                 input.click();
        #                 """)

        # # Button for selecting a file
        # self.button_select_file = Button(label="Select File", button_type="success")
        # self.button_select_file.js_on_click(select_file)

        # Smoothing parameters
        self.input_CI = TextInput(title="Circumferential Interval (in)", value='0.5')
        self.input_AI = TextInput(title="Axial Interval (in)", value='0.5')
        self.input_CW = TextInput(title="Circumferential Window", value='5')
        self.input_CS = TextInput(title="Circumferential Smoothing", value='0.001')
        self.input_AW = TextInput(title="Axial Window", value='9')
        self.input_AS = TextInput(title="Axial Smoothing", value='0.00005')

        # Button to perform Smoothing
        self.button_smooth = Button(label="Smooth Data", button_type="success")
        self.button_smooth.on_click(self._smooth_data)

        # Button to reset data
        self.button_reset = Button(label="Reset Data", button_type="default")
        self.button_reset.on_click(self._reset_data)

        # Button to perform strain analysis
        self.button_strain = Button(label="Calculate Strain", button_type="success")
        self.button_strain.on_click(self._strain)

        # Contour plot
        self.axial = np.zeros(10)
        self.circ = np.zeros(10)
        self.radius = np.zeros((10,10))
        self.plot_cont = figure(title="Surface Plot", height=400, width=800,
                    tools='pan, crosshair, wheel_zoom, box_zoom',
                    x_range=[0,1],
                    y_range=[0,360])
        levels = np.linspace(0, 1, 17)
        self.contour_renderer = self.plot_cont.contour(x=self.axial, y=self.circ, z=self.radius, levels=levels, fill_color=Viridis256, line_color="black", line_width=0, line_alpha=0)
        self.colorbar = self.contour_renderer.construct_color_bar()
        self.plot_cont.add_layout(self.colorbar, 'below')
        self.plot_cont.xaxis.axis_label
        self.plot_cont.toolbar.logo = None

        # Circumferential Plot
        self.source_circ = ColumnDataSource(data={'x':self.radius[0,:], 'y':np.deg2rad(self.circ)})
        self.source_circ2 = ColumnDataSource(data={'x':self.radius[0,:], 'y':np.deg2rad(self.circ)})
        self.plot_circ = figure(title="Circumferential Plot", height=600, width=600,
                        tools='pan, wheel_zoom, box_zoom',
                        x_range=[0,1],
                        y_range=[0,1])
        self.plot_circ.scatter(x='x', y='y', source=self.source_circ, size=5, legend_label="Hover View", color='blue')
        self.plot_circ.scatter(x='x', y='y', source=self.source_circ2, size=5, legend_label="Selection", color='orange')
        self.plot_circ.toolbar.logo = None

        # Longitudinal Plot
        self.source_long = ColumnDataSource(data={'x':self.axial, 'y':self.radius[0,:]})
        self.source_long2 = ColumnDataSource(data={'x':self.axial, 'y':self.radius[0,:]})
        self.plot_long = figure(title="Longitudinal Plot", height=400, width=800,
                        tools='pan, wheel_zoom, box_zoom',
                        x_range=[0,1],
                        y_range=[0,1])
                        # tooltips=[("index","$index"),("(x,y)","($x,$y)")])
        self.plot_long.scatter(x='x', y='y', source=self.source_long, size=2, legend_label="Hover View", color='blue')
        self.plot_long.scatter(x='x', y='y', source=self.source_long2, size=2, legend_label="Selection", color='orange')
        self.plot_long.toolbar.logo = None
        self.plot_long.add_tools(CrosshairTool(dimensions="height"))

        # Custom js_on_event functions
        self.plot_cont.js_on_event(events.MouseMove, self._move_update(None))
        self.plot_cont.js_on_event(events.Tap, self._tap_update(None))

        controls = [self.input_OD, 
                    self.input_WT, 
                    self.input_SMYS, 
                    self.input_ILI_format, 
                    self.button_select_file, 
                    self.input_CI, 
                    self.input_AI, 
                    self.input_CW, 
                    self.input_CS, 
                    self.input_AW, 
                    self.input_AS, 
                    self.button_smooth,
                    self.button_reset]
        inputs = column(controls, width=250, height=800)
        plots = column([self.plot_cont, self.plot_long])
        layout = column(desc, row(inputs, self.button_strain, plots, self.plot_circ), height=800)

        # curdoc().add_root(row(self.button_smooth, column(self.plot_cont, self.plot_long), self.plot_circ))
        curdoc().add_root(layout)
        curdoc().title = "PDMAC UI"

    def _file_callback(self, attr, old, new):
        # raw_contents = self.file_source.data['file_contents'][0]
        # prefix, b64_contents = raw_contents.split(",", 1)   # Remove the prefix that JS adds  
        # file_contents = base64.b64decode(b64_contents)
        file_contents = base64.b64decode(self.button_select_file.value)

        rd_path = io.StringIO(bytes.decode(file_contents))

        # print(f"rd_path = {rd_path}")

        # self.df = pro.ImportData(rd_path=rd_path, ILI_format=self.input_ILI_format.value, OD=self.input_OD.value, filename=self.file_source.data['file_name'][0])
        self.df = pro.PreProcess(rd_path=rd_path, ILI_format=str(self.input_ILI_format.value), OD=float(self.input_OD.value), filename=self.button_select_file.filename)

        self.axial  = self.df.o_axial
        self.circ   = self.df.o_circ
        self.radius = self.df.o_radius

        self._update_sources()

    def _pol2cart(self, r, rad):
        # Convert from cylindrical to cartesian coordinates
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        return x, y
    
    def _strain(self, attr):
        depth=0.05
        length=2.00
        self.strain = pro.PostProcess(OD=float(self.input_OD.value), WT=float(self.input_WT.value), d=depth, L=length, axial=self.df.f_axial, circ=self.df.f_circ, radius=self.df.f_radius)

        self.radius = self.strain.ei

        self._update_sources()

    def _smooth_data(self, attr):
        self.df.smooth_data()

        self.axial  = self.df.f_axial
        self.circ   = self.df.f_circ
        self.radius = self.df.f_radius

        self._update_sources()

    def _reset_data(self, attr):
        self.axial  = self.df.o_axial
        self.circ   = self.df.o_circ
        self.radius = self.df.o_radius

        self._update_sources()

    def _move_update(self, attr):
        # Define the callback function for the contour plot
        self.move_update_JS = CustomJS(args=dict(source_long=self.source_long, source_circ=self.source_circ, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius), code="""
                        const xval = cb_obj.x; // xval = axial value
                        const yval = cb_obj.y; // yval = circumferential value

                        // Determine the closest index of the data from the crosshair location
                        const xdiffArr = axial.map(x => Math.abs(xval - x));
                        const xminNumber = Math.min(...xdiffArr);
                        const axial_index = xdiffArr.findIndex(x => x === xminNumber);
                        
                        const ydiffArr = circ.map(y => Math.abs(yval - y));
                        const yminNumber = Math.min(...ydiffArr);
                        const circ_index = ydiffArr.findIndex(y => y === yminNumber);

                        // Extract the desired column from the 1D flattened array
                        var n = axial.length;    // number of rows
                        var m = circ.length;     // number of columns
                        var column = [];
                        for (var i = 0; i < n; i++) {
                            column.push(radius[i * m + circ_index]);
                        }

                        source_long.data['y'] = column;
                        source_long.change.emit();

                        // Extract the desired row from the 1D flattened array
                        var row = radius.slice(axial_index * m, (axial_index + 1) * m);
                        var circ_x = [];
                        var circ_y = [];
                        for (var i = 0; i < row.length; i++) {
                            circ_x.push(row[i] * Math.cos(circ_rad[i]));
                            circ_y.push(row[i] * Math.sin(circ_rad[i]));
                        }

                        source_circ.data['x'] = circ_x;
                        source_circ.data['y'] = circ_y;
                        source_circ.change.emit();
                    """)
        return self.move_update_JS
        
    def _tap_update(self, attr):
        self.tap_update_JS = CustomJS(args=dict(source_long2=self.source_long2, source_circ2=self.source_circ2, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius), code="""
                        const xval = cb_obj.x; // xval = axial value
                        const yval = cb_obj.y; // yval = circumferential value

                        // Determine the closest index of the data from the crosshair location
                        const xdiffArr = axial.map(x => Math.abs(xval - x));
                        const xminNumber = Math.min(...xdiffArr);
                        const axial_index = xdiffArr.findIndex(x => x === xminNumber);
                        
                        const ydiffArr = circ.map(y => Math.abs(yval - y));
                        const yminNumber = Math.min(...ydiffArr);
                        const circ_index = ydiffArr.findIndex(y => y === yminNumber);

                        // Extract the desired column from the 1D flattened array
                        var n = axial.length;    // number of rows
                        var m = circ.length;     // number of columns
                        var column = [];
                        for (var i = 0; i < n; i++) {
                            column.push(radius[i * m + circ_index]);
                        }

                        source_long2.data['y'] = column;
                        source_long2.change.emit();

                        // Extract the desired row from the 1D flattened array
                        var row = radius.slice(axial_index * m, (axial_index + 1) * m);
                        var circ_x = [];
                        var circ_y = [];
                        for (var i = 0; i < row.length; i++) {
                            circ_x.push(row[i] * Math.cos(circ_rad[i]));
                            circ_y.push(row[i] * Math.sin(circ_rad[i]));
                        }

                        source_circ2.data['x'] = circ_x;
                        source_circ2.data['y'] = circ_y;
                        source_circ2.change.emit();
                    """)
        return self.tap_update_JS
    
    def _update_sources(self):
        # Update data sources
        self.plot_cont.x_range.start = self.axial[0]
        self.plot_cont.x_range.end = self.axial[-1]
        self.plot_cont.y_range.start = self.circ[0]
        self.plot_cont.y_range.end = self.circ[-1]
        levels = np.linspace(self.radius.min(), self.radius.max(), 17)
        self.contour_renderer.set_data(contour_data(x=self.axial, y=self.circ, z=self.radius.T, levels=levels))
        # self.colorbar.levels = list(levels)
        # self.colorbar.ticker.ticks = list(levels)

        self.plot_circ.x_range.start = - (self.radius.max() + self.range_const)
        self.plot_circ.x_range.end = self.radius.max() + self.range_const
        self.plot_circ.y_range.start = - (self.radius.max() + self.range_const)
        self.plot_circ.y_range.end = self.radius.max() + self.range_const
        circ_x, circ_y = self._pol2cart(self.radius[0,:], np.deg2rad(self.circ))
        self.source_circ.data = {'x':circ_x, 'y':circ_y}
        self.source_circ2.data = {'x':circ_x, 'y':circ_y}

        self.plot_long.x_range.start = self.axial[0]
        self.plot_long.x_range.end = self.axial[-1]
        self.plot_long.y_range.start = self.radius.min() - self.range_const
        self.plot_long.y_range.end = self.radius.max() + self.range_const
        self.source_long.data = {'x':self.axial, 'y':self.radius[:,0]}
        self.source_long2.data = {'x':self.axial, 'y':self.radius[:,0]}

        self.move_update_JS.args = dict(source_long=self.source_long, source_circ=self.source_circ, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius)
        self.tap_update_JS.args = dict(source_long2=self.source_long2, source_circ2=self.source_circ2, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius)

CreateGUI()
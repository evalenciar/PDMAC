import numpy as np
# import pandas as pd

from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, TextInput, Select, ColumnDataSource, CustomJS, CrosshairTool, Div, FileInput, Span, BoxSelectTool, LinearColorMapper, ColorBar, BasicTicker, FixedTicker, PreText, RadioButtonGroup
from bokeh.plotting import figure
from bokeh.plotting.contour import contour_data
from bokeh.palettes import Viridis256

from tkinter import Tk
from tkinter.filedialog import askdirectory

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
        self.depth=0.05
        self.length=2.00

        # Description
        desc = Div(text=open('description.html').read(), sizing_mode='stretch_width')
        # Text input
        self.input_OD = TextInput(title="Outside Diameter (in)", value="24")
        self.input_WT = TextInput(title="Wall Thickness (in)", value="0.25")
        self.input_SMYS = TextInput(title="SMYS (psi)", value="52000")
        self.input_ILI_format = TextInput(title="ILI Format", value='southern')

        # File input
        self.button_select_file = FileInput(title="Select File", accept='.csv, .xlsx')
        self.button_select_file.on_change('filename', self._file_callback)

        # Results output div
        self.output_div = Div(width=400, height=400)

        # Smoothing parameters
        self.input_CI = TextInput(title="Circumferential Interval (in)", value='0.5')
        self.input_AI = TextInput(title="Axial Interval (in)", value='0.5')
        self.input_CW = TextInput(title="Circumferential Window", value='5')
        self.input_CS = TextInput(title="Circumferential Smoothing", value='0.001')
        self.input_AW = TextInput(title="Axial Window", value='9')
        self.input_AS = TextInput(title="Axial Smoothing", value='0.00005')

        # Radio button group to toggle data selection
        self.button_data_selection = RadioButtonGroup(labels=["Raw"], active=0)
        self.button_data_selection.on_event('button_click', self._switch_data)

        # Button to reset data
        self.button_reset = Button(label="Reset Data", button_type="default")
        self.button_reset.on_click(self._reset_data)

        # Button to perform Smoothing
        self.button_smooth = Button(label="Smooth Data", button_type="primary", width=120)
        self.button_smooth.on_click(self._smooth_data)

        # Button to perform strain analysis
        self.button_strain = Button(label="Calculate Strain", button_type="success", width=120)
        self.button_strain.on_click(self._strain)

        # Button to create Input File
        self.button_create_input = Button(label="Create Input File", button_type="warning", width=120)
        self.button_create_input.on_click(self._create_input)

        # Button to submit the Input File
        self.button_submit_input = Button(label="Submit Input File", button_type="danger", disabled=True, width=120)
        self.button_submit_input.on_click(self._submit_input)

        # Button to review SCF
        self.button_review_SCF = Button(label="Review SCF", button_type="success", disabled=True, width=120)
        self.button_review_SCF.on_click(self._review_SCF)

        # Linked crosshair tool
        span_width = Span(dimension="width")
        span_height = Span(dimension="height")

        # Contour plot
        self.axial = np.zeros(10)
        self.circ = np.zeros(10)
        self.radius = np.zeros((10,10))
        self.plot_cont = figure(title="Surface Plot", height=400, width=800,
                    # tools='pan, crosshair, wheel_zoom, box_zoom',
                    tools='box_select, wheel_zoom, box_zoom',
                    active_drag="box_select",
                    # toolbar_location=None,
                    x_range=[0,1],
                    y_range=[0,360])
        self.levels = np.linspace(0, 1, 17)
        self.color = LinearColorMapper(palette=Viridis256, low=np.min(self.radius), high=np.max(self.radius))
        self.contour_renderer = self.plot_cont.contour(x=self.axial, y=self.circ, z=self.radius, levels=self.levels, fill_color=Viridis256, line_color="black", line_width=0, line_alpha=0)
        self.plot_cont.yaxis.axis_label = "Circumferential (deg)"
        self.plot_cont.add_tools(CrosshairTool(overlay=[span_width, span_height]))
        # self.plot_cont.toolbar.active_drag = None
        self.plot_cont.toolbar.logo = None
        # select_tool = self.plot_cont.select(dict(type=BoxSelectTool))
        
        # self.colorbar = self.contour_renderer.construct_color_bar()
        # self.plot_cont.add_layout(self.colorbar, 'below')
        self.colorbar = ColorBar(color_mapper=self.color, location=(0,0), title='Radius (in)', 
                                #  ticker=BasicTicker(desired_num_ticks=12), 
                                #  ticker=FixedTicker(ticks=list(self.levels)),
                                 bar_line_color='black', 
                                 major_tick_line_color='white')
        self.plot_cont.add_layout(self.colorbar, 'below')

        # Circumferential Plot
        self.source_circ = ColumnDataSource(data={'x':self.radius[0,:], 'y':np.deg2rad(self.circ)})
        self.source_circ2 = ColumnDataSource(data={'x':self.radius[0,:], 'y':np.deg2rad(self.circ)})
        self.plot_circ = figure(title="Circumferential Plot", height=350, width=350,
                        # tools='pan, wheel_zoom, box_zoom',
                        toolbar_location=None,
                        x_range=[0,1],
                        y_range=[0,1])
        self.plot_circ.scatter(x='x', y='y', source=self.source_circ, size=5, legend_label="Hover View", color='blue')
        self.plot_circ.scatter(x='x', y='y', source=self.source_circ2, size=5, legend_label="Selection", color='orange')
        self.plot_circ.axis.visible = False
        # self.plot_circ.toolbar.logo = None

        # Longitudinal Plot
        self.source_long = ColumnDataSource(data={'x':self.axial, 'y':self.radius[0,:]})
        self.source_long2 = ColumnDataSource(data={'x':self.axial, 'y':self.radius[0,:]})
        self.plot_long = figure(title="Longitudinal Plot", height=400, width=800,
                        # tools='pan, wheel_zoom, box_zoom',
                        toolbar_location=None,
                        # x_range=[0,1],
                        x_range=self.plot_cont.x_range,
                        y_range=[0,1])
                        # tooltips=[("index","$index"),("(x,y)","($x,$y)")])
        self.plot_long.scatter(x='x', y='y', source=self.source_long, size=2, legend_label="Hover View", color='blue')
        self.plot_long.scatter(x='x', y='y', source=self.source_long2, size=2, legend_label="Selection", color='orange')
        self.plot_long.yaxis.axis_label = "Radius (in)"
        self.plot_long.xaxis.axis_label = "Axial (in)"
        self.plot_long.add_tools(CrosshairTool(overlay=span_height))
        self.plot_long.toolbar.active_drag = None
        # self.plot_long.toolbar.logo = None

        # Custom js_on_event functions
        self.plot_cont.js_on_event(events.MouseMove, self._move_update(None))
        self.plot_cont.js_on_event(events.Tap, self._tap_update(None, radius_label=self.plot_long.yaxis.axis_label))

        controls = [self.button_data_selection,
                    self.input_OD, 
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
                    self.button_reset]
        button_column = column(row([self.button_smooth, self.button_strain]), row([self.button_create_input, self.button_submit_input]), self.button_review_SCF, sizing_mode="stretch_height")
        first_row = row(desc, button_column)

        inputs = column(controls, width=250, height=800)
        
        center_plots = column([self.plot_cont, self.plot_long])
        last_column = column([self.output_div, self.plot_circ])
        layout = column(first_row, row(inputs, center_plots, last_column), height=800)

        # curdoc().add_root(row(self.button_smooth, column(self.plot_cont, self.plot_long), self.plot_circ))
        curdoc().add_root(layout)
        curdoc().title = "PDMAC UI"

    def _file_callback(self, attr, old, new):
        file_contents = base64.b64decode(self.button_select_file.value)

        rd_path = io.StringIO(bytes.decode(file_contents))

        self.df = pro.Process(rd_path=rd_path, ILI_format=str(self.input_ILI_format.value), OD=float(self.input_OD.value), WT=float(self.input_WT.value), SMYS=float(self.input_SMYS.value), filename=self.button_select_file.filename)

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
        self.df.calculate_strain(d=0.1, L=3)
        self.radius = self.df.ei

        self._update_sources()

        self.button_data_selection.labels = ["Raw","Smooth","Strain"]
        self.button_data_selection.active = 2

    def _create_input(self, attr):
        def select_file():
            root = Tk()
            root.attributes('-topmost', True)
            root.withdraw()
            dirname = askdirectory()  # blocking
            if dirname:
                return dirname + '/'
            else:
                return 'results/'

        self.df.create_input_file(results_path=select_file())
        self.button_submit_input.disabled = False

    def _submit_input(self, attr):
        def f_submit_input():
            self.df.submit_input_file()

            self.button_submit_input.disabled = False
            self.button_review_SCF.disabled = False

        self.button_submit_input.disabled = True
        curdoc().add_next_tick_callback(f_submit_input)

    def _review_SCF(self, attr):
        def f_review_SCF():
            self.df.review_abaqus_results()
            self.radius = self.df.S_SPOS

            self._update_sources()

            self.button_data_selection.labels = ["Raw","Smooth","Strain","SPOS","SNEG"]
            self.button_data_selection.active = 3
        
        curdoc().add_next_tick_callback(f_review_SCF)

    def _switch_data(self, attr):
        if self.button_data_selection.active == 0:
            self.axial  = self.df.o_axial
            self.circ   = self.df.o_circ
            self.radius = self.df.o_radius
            self.plot_long.yaxis.axis_label = "Radius (in)"
            self.colorbar.title = "Radius (in)"
        elif self.button_data_selection.active == 1:
            self.axial  = self.df.f_axial
            self.circ   = self.df.f_circ
            self.radius = self.df.f_radius
            self.plot_long.yaxis.axis_label = "Radius (in)"
            self.colorbar.title = "Radius (in)"
        elif self.button_data_selection.active == 2:
            self.axial  = self.df.f_axial
            self.circ   = self.df.f_circ
            self.radius = self.df.ei
            self.plot_long.yaxis.axis_label = "Strain (in/in)"
            self.colorbar.title = "Strain (in/in)"
        elif self.button_data_selection.active == 3:
            self.axial  = self.df.f_axial
            self.circ   = self.df.f_circ
            self.radius = self.df.S_SPOS
            self.plot_long.yaxis.axis_label = "SPOS SCF"
            self.colorbar.title = "SPOS SCF"
        elif self.button_data_selection.active == 4:
            self.axial  = self.df.f_axial
            self.circ   = self.df.f_circ
            self.radius = self.df.S_SNEG
            self.plot_long.yaxis.axis_label = "SNEG SCF"
            self.colorbar.title = "SNEG SCF"
        
        self._update_sources()

    def _smooth_data(self, attr):
        def f_smooth_data():
            self.df.smooth_data()
            self.button_smooth.disabled = False

            self.axial  = self.df.f_axial
            self.circ   = self.df.f_circ
            self.radius = self.df.f_radius

            self._update_sources()

            self.button_data_selection.labels = ["Raw","Smooth"]
            self.button_data_selection.active = 1

        self.button_smooth.disabled = True
        curdoc().add_next_tick_callback(f_smooth_data)

    def _reset_data(self, attr):
        self._switch_data

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
        
    def _tap_update(self, attr, radius_label = 'Radius (in)', style = 'float:left;clear:left;font_size=13px'):
        self.tap_update_JS = CustomJS(args=dict(div=self.output_div, source_long2=self.source_long2, source_circ2=self.source_circ2, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius), code="""
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
                                      
                        const closestIndex = (num, arr) => {
                            let curr = arr[0], diff = Math.abs(num - curr);
                            let index = 0;
                            for (let val = 0; val < arr.length; val++) {
                                let newdiff = Math.abs(num - arr[val]);
                                if (newdiff < diff) {
                                    diff = newdiff;
                                    curr = arr[val];
                                    index = val;
                                };
                            };
                            return index;
                        };
                                      
                        var x_index = closestIndex(cb_obj.x, source_long2.data['x'])
                        var z_val = source_long2.data['y'][x_index];
                        var x_val = cb_obj.x.toFixed(2)
                        var y_val = cb_obj.y.toFixed(2)
                        
                        const line ="<span style=%r><b>Selection:</b><br /> Axial (in) = " + x_val + " [" + (x_val/12).toFixed(2) + " feet]<br />Circumferential (deg) = " + y_val + "<br />%s = " + z_val.toFixed(3) + "</span>\\n";
                        div.text = line;
                                      
                        // const text = div.text.concat(line);
                        // const lines = text.split("\\n")
                        // if (lines.length > 35)
                           // lines.shift();
                        // div.text = lines.join("\\n");
                    """ % (style, radius_label))
        return self.tap_update_JS
    
    def _update_sources(self):
        # Update data sources
        self.plot_cont.x_range.start = self.axial[0]
        self.plot_cont.x_range.end = self.axial[-1]
        self.plot_cont.y_range.start = self.circ[0]
        self.plot_cont.y_range.end = self.circ[-1]
        self.levels = np.linspace(np.min(self.radius), np.max(self.radius), 17)
        self.contour_renderer.set_data(contour_data(x=self.axial, y=self.circ, z=self.radius.T, levels=self.levels))
        # self.colorbar.levels = list(levels)
        # self.colorbar.ticker.ticks = list(levels)
        self.color.low = np.min(self.radius)
        self.color.high = np.max(self.radius)
        # self.colorbar.ticker = FixedTicker(ticks=list(self.levels))

        self.plot_circ.x_range.start = - (self.radius.max() + self.range_const)
        self.plot_circ.x_range.end = self.radius.max() + self.range_const
        self.plot_circ.y_range.start = - (self.radius.max() + self.range_const)
        self.plot_circ.y_range.end = self.radius.max() + self.range_const
        circ_x, circ_y = self._pol2cart(self.radius[0,:], np.deg2rad(self.circ))
        self.source_circ.data = {'x':circ_x, 'y':circ_y}
        self.source_circ2.data = {'x':circ_x, 'y':circ_y}

        self.plot_long.y_range.start = self.radius.min() - self.range_const
        self.plot_long.y_range.end = self.radius.max() + self.range_const
        self.source_long.data = {'x':self.axial, 'y':self.radius[:,0]}
        self.source_long2.data = {'x':self.axial, 'y':self.radius[:,0]}

        self.move_update_JS.args = dict(source_long=self.source_long, source_circ=self.source_circ, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius)
        self.tap_update_JS.args = dict(div=self.output_div, source_long2=self.source_long2, source_circ2=self.source_circ2, axial=self.axial, circ=self.circ, circ_rad=np.deg2rad(self.circ), radius=self.radius)

CreateGUI()
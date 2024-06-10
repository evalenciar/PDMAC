# Using bokeh to display data

import numpy as np
import pandas as pd

from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, PolarTransform, CustomJS, CrosshairTool
from bokeh.plotting import figure
from bokeh.palettes import Viridis256

# Collect Raw Data
def collect_raw_data(rd_path):
    # Created on 04/11/2024
    # Axial position (m) is in vertical direction, starting at A2
    # There is no Circumferential position or caliper number
    # Internal radial values (mm) start at C2
    rd = pd.read_csv(rd_path, header=None, dtype=float, skiprows=1)
    # Drop column B (index=1) since it is not being used
    rd.drop(rd.columns[1], axis=1, inplace=True)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in COLUMN direction
    rd_axial = rd.loc[0:][0].to_numpy().astype(float)
    # Convert (m) values to relative inches (in)
    rd_axial = [39.3701 * (x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning based on number of caliper data columns
    # Ignore the Axial values column
    rd_circ = np.arange(0, len(rd.loc[0][1:]))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    # Important: Data structure needs to be [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float)
    # Convert from (mm) to (in)
    rd_radius = rd_radius * 0.0393701
    return rd_axial, rd_circ, rd_radius

# Define the callback function for the contour plot
def display_event(attributes):
    return CustomJS(args=dict(source_long=source_long, source_circ=source_circ, rd_axial=rd_axial, rd_circ=rd_circ, rd_radius=rd_radius), code="""
                    const xval = cb_obj.x; // xval = axial value
                    const yval = cb_obj.y; // yval = circumferential value

                    // Determine the closest index of the data from the crosshair location
                    const xdiffArr = rd_axial.map(x => Math.abs(xval - x));
                    const xminNumber = Math.min(...xdiffArr);
                    const axial_index = xdiffArr.findIndex(x => x === xminNumber);
                    
                    const ydiffArr = rd_circ.map(y => Math.abs(yval - y));
                    const yminNumber = Math.min(...ydiffArr);
                    const circ_index = ydiffArr.findIndex(y => y === yminNumber);

                    // Extract the desired column from the 1D flattened array
                    var n = rd_axial.length;    // number of rows
                    var m = rd_circ.length;     // number of columns
                    var column = [];
                    for (var i = 0; i < n; i++) {
                        column.push(rd_radius[i * m + circ_index]);
                    }

                    source_long.data['y'] = column;
                    source_long.change.emit();

                    // Extract the desired row from the 1D flattened array
                    var row = rd_radius.slice(axial_index * m, (axial_index + 1) * m);

                    source_circ.data['radius'] = row;
                    source_circ.change.emit();
                """)

range_const = 0

# Collect the raw data
dent_path = 'ADV Project Folders/5972_RadiiData.csv'
rd_axial, rd_circ, rd_radius = collect_raw_data(dent_path)
rd_circ_rad = np.deg2rad(rd_circ)
# df = pd.DataFrame(data=rd_radius, index=rd_axial, columns=rd_circ)

# Contour plot
# source_cont = ColumnDataSource(data={'x':rd_axial, 'y':rd_circ, 'z':rd_radius.T})
plot_cont = figure(title="Surface Plot", height=400, width=800,
            tools='reset',
            x_range=[rd_axial[0], rd_axial[-1]],
            y_range=[rd_circ[0], rd_circ[-1]])
levels = np.linspace(rd_radius.min(), rd_radius.max(), 17)
contour_renderer = plot_cont.contour(x=rd_axial, y=rd_circ, z=rd_radius.T, levels=levels, fill_color=Viridis256)
colorbar = contour_renderer.construct_color_bar()
plot_cont.add_layout(colorbar, 'below')
plot_cont.xaxis.axis_label
plot_cont.toolbar.logo = None
# plot_cont.toolbar_location = None

# Circumferential Plot
source_circ = ColumnDataSource(data={'radius':rd_radius[0], 'angle':rd_circ_rad})
t = PolarTransform()
plot_circ = figure(title="Circumferential Plot", height=800, width=800,
                   tools='pan, reset, wheel_zoom, box_zoom',
                   x_range=[- (np.max(rd_radius) + range_const), np.max(rd_radius) + range_const],
                   y_range=[- (np.max(rd_radius) + range_const), np.max(rd_radius) + range_const],)
plot_circ.scatter(x=t.x, y=t.y, source=source_circ, size=5)
plot_circ.toolbar.logo = None
# plot_circ.toolbar_location = None

# Longitudinal Plot
source_long = ColumnDataSource(data={'x':rd_axial, 'y':rd_radius[:,0]})
plot_long = figure(title="Longitudinal Plot", height=400, width=800,
                   tools='reset, pan, wheel_zoom, box_zoom',
                   x_range=[rd_axial[0], rd_axial[-1]],
                   y_range=[rd_radius.min() - range_const, rd_radius.max() + range_const])
plot_long.scatter(x='x', y='y', source=source_long, size=2)
plot_long.toolbar.logo = None
# plot_long.toolbar_location = None

plot_cont.js_on_event(events.Tap, display_event(None))

plot_cont.add_tools(CrosshairTool(dimensions="both"))
plot_long.add_tools(CrosshairTool(dimensions="height"))

curdoc().add_root(row(column(plot_cont, plot_long), plot_circ))
curdoc().title = "Visualize Data"
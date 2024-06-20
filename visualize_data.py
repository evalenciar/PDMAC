# Using bokeh to display data

import numpy as np
import pandas as pd

from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import Button, ColumnDataSource, CustomJS, CrosshairTool, Div, HoverTool
from bokeh.plotting import figure
from bokeh.plotting.contour import contour_data
from bokeh.palettes import Viridis256, PiYG

import io
import base64

import dent_process
# from surface3d import Surface3d

def move_update(attributes, div, style = 'float:left;clear:left;font_size=13px'):
    # Define the callback function for the contour plot
    global move_update_JS
    move_update_JS = CustomJS(args=dict(div=div, source_long=source_long, source_circ=source_circ, rd_axial=rd_axial, rd_circ=rd_circ, rd_circ_rad=rd_circ_rad, rd_radius=rd_radius), code="""
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
                    var circ_x = [];
                    var circ_y = [];
                    for (var i = 0; i < row.length; i++) {
                        circ_x.push(row[i] * Math.cos(rd_circ_rad[i]));
                        circ_y.push(row[i] * Math.sin(rd_circ_rad[i]));
                    }

                    source_circ.data['x'] = circ_x;
                    source_circ.data['y'] = circ_y;
                    source_circ.change.emit();

                    // START DEBUGGING
                    const args = [];
                    // args.push('x = ' + Number(xval).toFixed(2));
                    args.push('xid = ' + axial_index);
                    // args.push('y = ' + Number(yval).toFixed(2));
                    args.push('yid = ' + circ_index);
                    // args.push('row_length = ' + row.length);
                    // args.push('row[0] = ' + row[0]);
                    // args.push('row[-1] = ' + row.slice(-1));

                    const line = "<span style=%r><b>" + cb_obj.event_name + "</b>(" + args.join(", ") + ")</span>\\n";
                    const text = div.text.concat(line);
                    const lines = text.split("\\n")
                    if (lines.length > 35)
                        lines.shift();
                    div.text = lines.join("\\n");
                    // END DEBUGGING
                """ % (style))
    return move_update_JS

def tap_update(attributes):
    global tap_update_JS
    tap_update_JS = CustomJS(args=dict(source_long2=source_long2, source_circ2=source_circ2, rd_axial=rd_axial, rd_circ=rd_circ, rd_circ_rad=rd_circ_rad, rd_radius=rd_radius), code="""
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

                    source_long2.data['y'] = column;
                    source_long2.change.emit();

                    // Extract the desired row from the 1D flattened array
                    var row = rd_radius.slice(axial_index * m, (axial_index + 1) * m);
                    var circ_x = [];
                    var circ_y = [];
                    for (var i = 0; i < row.length; i++) {
                        circ_x.push(row[i] * Math.cos(rd_circ_rad[i]));
                        circ_y.push(row[i] * Math.sin(rd_circ_rad[i]));
                    }

                    source_circ2.data['x'] = circ_x;
                    source_circ2.data['y'] = circ_y;
                    source_circ2.change.emit();

                """)
    return tap_update_JS

def pol2cart(r, rad):
    # Convert from cylindrical to cartesian coordinates
    x = r * np.cos(rad)
    y = r * np.sin(rad)
    return x, y
    
def file_callback(attr,old,new):
    # print ('filename:', file_source.data['file_name'])
    raw_contents = file_source.data['file_contents'][0]
    # remove the prefix that JS adds  
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = io.StringIO(bytes.decode(file_contents))
    # df = pd.read_excel(file_io)
    rd_axial, rd_circ, rd_radius = dent_process.collect_raw_data_v7(file_io)

    update_sources(rd_axial, rd_circ, rd_radius)

    # print ("file contents:")
    # print (df)

def update_sources(rd_axial, rd_circ, rd_radius):
    # Update data sources
    plot_cont.x_range.start = rd_axial[0]
    plot_cont.x_range.end = rd_axial[-1]
    plot_cont.y_range.start = rd_circ[0]
    plot_cont.y_range.end = rd_circ[-1]
    levels = np.linspace(rd_radius.min(), rd_radius.max(), 17)
    contour_renderer.set_data(contour_data(x=rd_axial, y=rd_circ, z=rd_radius.T, levels=levels))

    plot_circ.x_range.start = - (rd_radius.max() + range_const)
    plot_circ.x_range.end = rd_radius.max() + range_const
    plot_circ.y_range.start = - (rd_radius.max() + range_const)
    plot_circ.y_range.end = rd_radius.max() + range_const
    rd_circ_rad = np.deg2rad(rd_circ)
    circ_x, circ_y = pol2cart(rd_radius[0,:], rd_circ_rad)
    source_circ.data = {'x':circ_x, 'y':circ_y}
    source_circ2.data = {'x':circ_x, 'y':circ_y}

    plot_long.x_range.start = rd_axial[0]
    plot_long.x_range.end = rd_axial[-1]
    plot_long.y_range.start = rd_radius.min() - range_const
    plot_long.y_range.end = rd_radius.max() + range_const
    source_long.data = {'x':rd_axial, 'y':rd_radius[:,0]}
    source_long2.data = {'x':rd_axial, 'y':rd_radius[:,0]}

    move_update_JS.args = dict(div=div, source_long=source_long, source_circ=source_circ, rd_axial=rd_axial, rd_circ=rd_circ, rd_circ_rad=rd_circ_rad, rd_radius=rd_radius)
    tap_update_JS.args = dict(source_long2=source_long2, source_circ2=source_circ2, rd_axial=rd_axial, rd_circ=rd_circ, rd_circ_rad=rd_circ_rad, rd_radius=rd_radius)

def smooth_data(attr):
    global rd_axial
    global rd_circ
    global rd_radius

    # circ_int = 0.5
    # axial_int = 0.5
    
    # axial_window = 9
    # axial_smooth = 0.00005
    # circ_smooth = 0.001
    # circ_window = 5
    
    # Perform data smoothing on the raw data
    print("Began data smoothing")
    sd_axial, sd_circ, sd_radius = dent_process.data_smoothing(OD, rd_axial, rd_circ, rd_radius) 
                                                            #    circ_int=circ_int, 
                                                            #    axial_int=axial_int, 
                                                            #    circ_window=circ_window,
                                                            #    circ_smooth=circ_smooth,
                                                            #    axial_window=axial_window,
                                                            #    axial_smooth=axial_smooth)
    print("Finished data smoothing")
    update_sources(sd_axial, sd_circ, sd_radius)
    print("Updated data sources")



range_const = 0

# Collect the raw data
dent_path = 'ADV Project Folders/5972_RadiiData.csv'
rd_axial, rd_circ, rd_radius = dent_process.collect_raw_data_v7(dent_path)
rd_circ_rad = np.deg2rad(rd_circ)
# df = pd.DataFrame(data=rd_radius, index=rd_axial, columns=rd_circ)

# File selection
file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})
file_source.on_change('data', file_callback)

# Create a CustomJS callback to handle file selection
select_file = CustomJS(args=dict(file_source=file_source), code = """
                function read_file(filename) {
                    var reader = new FileReader();
                    reader.onload = load_handler;
                    reader.onerror = error_handler;
                    // readAsDataURL represents the file's data as a base64 encoded string
                    reader.readAsDataURL(filename);
                }

                function load_handler(event) {
                    var b64string = event.target.result;
                    file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
                    file_source.trigger("change");
                }

                function error_handler(evt) {
                    if(evt.target.error.name == "NotReadableError") {
                        alert("Can't read file!");
                    }
                }

                var input = document.createElement('input');
                input.setAttribute('type', 'file');
                input.onchange = function(){
                    if (window.FileReader) {
                        read_file(input.files[0]);
                    
                    } else {
                        alert('FileReader is not supported in this browser');
                    }
                }
                input.click();
                """)

# Create a button that triggers the file input dialog
# button_select_file = Button(label="Select File", button_type="success")
# button_select_file.js_on_click(select_file)

# Contour plot
plot_cont = figure(title="Surface Plot", height=400, width=800,
            tools='reset, pan, crosshair, wheel_zoom, box_zoom',
            x_range=[rd_axial[0], rd_axial[-1]],
            y_range=[rd_circ[0], rd_circ[-1]])
levels = np.linspace(rd_radius.min(), rd_radius.max(), 17)
contour_renderer = plot_cont.contour(x=rd_axial, y=rd_circ, z=rd_radius.T, levels=levels, fill_color=Viridis256, line_color="black", line_width=0, line_alpha=0)
colorbar = contour_renderer.construct_color_bar()
plot_cont.add_layout(colorbar, 'below')
plot_cont.xaxis.axis_label
plot_cont.toolbar.logo = None

# Circumferential Plot
circ_x, circ_y = pol2cart(rd_radius[0,:], rd_circ_rad)
source_circ = ColumnDataSource(data={'x':circ_x, 'y':circ_y})
source_circ2 = ColumnDataSource(data={'x':circ_x, 'y':circ_y})

plot_circ = figure(title="Circumferential Plot", height=800, width=800,
                tools='reset, pan, wheel_zoom, box_zoom',
                x_range=[- (np.max(rd_radius) + range_const), np.max(rd_radius) + range_const],
                y_range=[- (np.max(rd_radius) + range_const), np.max(rd_radius) + range_const],)
plot_circ.scatter(x='x', y='y', source=source_circ, size=5, legend_label="Hover View", color='blue')
plot_circ.scatter(x='x', y='y', source=source_circ2, size=5, legend_label="Selection", color='orange')
plot_circ.toolbar.logo = None

# Longitudinal Plot
source_long = ColumnDataSource(data={'x':rd_axial, 'y':rd_radius[:,0]})
source_long2 = ColumnDataSource(data={'x':rd_axial, 'y':rd_radius[:,0]})
plot_long = figure(title="Longitudinal Plot", height=400, width=800,
                tools='reset, pan, wheel_zoom, box_zoom',
                x_range=[rd_axial[0], rd_axial[-1]],
                y_range=[rd_radius.min() - range_const, rd_radius.max() + range_const],
                tooltips=[("index","$index"),("(x,y)","($x,$y)")])
plot_long.scatter(x='x', y='y', source=source_long, size=2, legend_label="Hover View", color='blue')
plot_long.scatter(x='x', y='y', source=source_long2, size=2, legend_label="Selection", color='orange')
plot_long.toolbar.logo = None

div = Div(width=200, height=400)
plot_cont.js_on_event(events.MouseMove, move_update(None, div))
plot_cont.js_on_event(events.Tap, tap_update(None))

# plot_cont.add_tools(CrosshairTool(dimensions="both"))
plot_long.add_tools(CrosshairTool(dimensions="height"))

# Smooth Data
button_smooth = Button(label="Smooth Data", button_type="success")
OD = 24
button_smooth.on_click(smooth_data)

# 3D Plot
# xx, yy = np.meshgrid(rd_axial, rd_circ)
# xx = xx.ravel()
# yy = yy.ravel()
# source_3d = ColumnDataSource(data=dict(x=xx, y=yy, z=rd_radius.ravel()))
# plot_3d = Surface3d(x='x', y='y', z='z', data_source=source_3d, width=400, height=400)

curdoc().add_root(row(button_smooth, column(plot_cont, plot_long), div, plot_circ))
# curdoc().add_root(row(button_smooth, column(plot_cont, plot_long), plot_circ))
# curdoc().add_root(row(button_smooth, column(plot_cont, plot_long, plot_circ), plot_3d))
curdoc().title = "PDMAC UI"
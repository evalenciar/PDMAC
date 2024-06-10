# -*- coding: utf-8 -*-
"""
Collect, Smooth, and Simulate

The objective of this script is to process in-line inspection (ILI) 
geometry/radii data containing a dent feature and generate an SCF value.

@author: evalencia
"""

# Import math and array related modules
import numpy as np
import pandas as pd
import math
# Import data smoothing modules
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
# Import system modules
import time
import shutil
import os
# import readline
import sys

# Prevent the figures from being displayed
plt.ioff()

# =============================================================================
# FUNCTIONS
# =============================================================================

def collect_raw_data_CAMPOS(rd_path, IR):
    # Collect raw data
    rd = pd.read_excel(rd_path, header=None)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in column direction, starting from column B (delete first column)
    rd_axial = rd.loc[0][1:].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning as caliper number in column 
    # Start from 2 since first row is Distance
    # rd_circ = rd[0][1:].to_numpy().astype(float)
    rd_circ = rd[0][1:]
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first column and row
    rd.drop(rd.columns[0], axis=1, inplace=True)
    rd.drop(rd.head(1).index, axis=0, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    # rd_radius = rd_radius - IR
    # Convert from mm to inches
    rd_radius = rd_radius / 25.4
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v1(rd_path):
    # Collect raw data in vendor 1 formatting
    # Data is all in Column A with delimiter ';' with 12 header rows that need to
    # be removed before accessing the actual data
    rd_axial_row = 12  # Row 13 = Index 12
    rd_drop_tail = 2
    rd = pd.read_csv(rd_path, header=None)
    # Drop the first set of rows that are not being used
    rd.drop(rd.head(rd_axial_row).index, inplace=True)
    # Drop the last two rows that are not being used
    rd.drop(rd.tail(rd_drop_tail).index, inplace=True)
    rd = rd[0].str.split(';', expand=True)
    rd = rd.apply(lambda x: x.str.strip())
    # Drop the last column since it is empty
    rd.drop(rd.columns[-1], axis=1, inplace=True)
    # Relative axial positioning values
    rd_axial = rd.loc[rd_axial_row].to_numpy()
    # Delete the first two values which are 'Offset' and '(ft)'
    rd_axial = np.delete(rd_axial, [0,1])
    rd_axial = rd_axial.astype(float)
    # Convert the axial values to inches
    rd_axial = rd_axial*12
    # Drop the two top rows: Offset and Radius
    rd.drop(rd.head(2).index, inplace=True)
    # Circumferential positioning in [degrees]
    rd_circ = rd[0].to_numpy()
    # Convert from clock to degrees
    rd_circ = [x.split(':') for x in rd_circ]
    rd_circ = [round((float(x[0]) + float(x[1])/60)*360/12,1) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the two first columns: Circumferential in o'Clock and in Length inches
    rd.drop(rd.columns[[0,1]], axis=1, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v2(rd_path):
    # Collect raw data in vendor 2 formatting
    rd = pd.read_csv(rd_path, header=None)
    # Axial values are in row direction
    rd_axial = rd[0].to_numpy().astype(float)
    rd_axial = np.delete(rd_axial, 0)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning as caliper number in column direction
    rd_circ = rd.loc[0].to_numpy().astype(float)
    rd_circ = np.delete(rd_circ, 0)
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first row and column
    rd.drop(rd.head(1).index, inplace=True)
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [circ x axial]
    # Make the data negative since it is reading in the opposite direction
    rd_radius = -rd.to_numpy().astype(float)
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_TDW(rd_path, IR):
    # Collect raw data in vedor 3 formatting
    rd = pd.read_csv(rd_path, header=None)
    # Drop any columns with 'NaN' trailing at the end
    rd.dropna(axis=1, how='all', inplace=True)
    # Drop any columns with ' ' trailing at the end
    if rd.iloc[0][rd.columns[-1]] == ' ':
        rd.drop(columns=rd.columns[-1], axis=1, inplace=True)
    # First row gives the original orientation of each sensor, starting from the second column
    rd_circ = rd.iloc[0][1:].to_numpy().astype(float)
    # Since the orientation values are not incremental, will need to roll the data to have the smallest angle starting
    roll_amount = len(rd_circ) - np.argmin(rd_circ)
    rd_circ = np.roll(rd_circ, roll_amount)
    # Drop the first three rows
    rd.drop(rd.head(3).index, inplace=True)
    # Axial values are in row direction, starting from row 4 (delete first 3 rows)
    rd_axial = rd[0].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    rd_radius = rd.to_numpy().astype(float)
    # Also need to roll the ROWS in rd_radius since the circumferential orientation was rolled
    rd_radius = np.roll(rd_radius, roll_amount, axis=0)
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    # rd_radius = rd_radius - IR
    rd_radius = rd_radius
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v4(rd_path, IR):
    # Collect raw data in vedor 3 formatting
    rd = pd.read_csv(rd_path, header=None, dtype=float)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in column direction, starting from column B (delete first column)
    rd_axial = rd.loc[0][1:].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning as caliper number in column 
    # Start from 2 since first row is Distance
    rd_circ = rd[0][1:].to_numpy().astype(float)
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first column and row
    rd.drop(rd.columns[0], axis=1, inplace=True)
    rd.drop(rd.head(1).index, axis=0, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    # rd_radius = rd_radius - IR
    rd_radius = rd_radius
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_TDW2(rd_path, IR):
    # Created on: 09/19/2022
    # Similar to TDW, minus rows 2 and 3
    # Collect raw data
    rd = pd.read_csv(rd_path, header=None)
    # Drop any columns with 'NaN' trailing at the end
    rd.dropna(axis=1, how='all', inplace=True)
    # Drop any columns with ' ' trailing at the end
    if rd.iloc[0][rd.columns[-1]] == ' ':
        rd.drop(columns=rd.columns[-1], axis=1, inplace=True)
    # First row gives the original orientation of each sensor, starting from the second column
    rd_circ = rd.iloc[0][1:].to_numpy().astype(float)
    # Since the orientation values are not incremental, will need to roll the data to have the smallest angle starting
    roll_amount = len(rd_circ) - np.argmin(rd_circ)
    rd_circ = np.roll(rd_circ, roll_amount)
    # Drop the first row
    rd.drop(rd.head(1).index, inplace=True)
    # Axial values are in row direction, starting from row 2 (delete first 1 row)
    rd_axial = rd[0].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    rd_radius = rd.to_numpy().astype(float)
    # Also need to roll the ROWS in rd_radius since the circumferential orientation was rolled
    rd_radius = np.roll(rd_radius, roll_amount, axis=0)
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    rd_radius = rd_radius
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_PBF(rd_path):
    # Created on: 10/21/2022
    # Data has Orientation (oclock) in Horizontal direction starting from B2.
    # Axial information is in Vertical direction, starting from A3
    # Radial values are the IR values
    # Collect raw data and delete the first row
    rd = pd.read_csv(rd_path, header=None, skiprows=1)
    # New first row gives the original orientation of each sensor in oclock, starting from the second column (B)
    rd_circ = rd.iloc[0][1:].to_numpy()
    # Convert from clock to degrees. There are oclock using 12 instead of 0, therefore need to adjust
    rd_circ = [x.split(':') for x in rd_circ]
    rd_circ = [(float(x[0]) * 60 + float(x[1]))/2 for x in rd_circ]
    for i, val in enumerate(rd_circ):
        if val > 360:
            rd_circ[i] = val - 360
    rd_circ = np.array(rd_circ)
    # Since the orientation values are not incremental, will need to roll the data to have the smallest angle starting
    roll_amount = len(rd_circ) - np.argmin(rd_circ)
    rd_circ = np.roll(rd_circ, roll_amount)
    # Drop the first row
    rd.drop(rd.head(1).index, inplace=True)
    # Axial values are in the column direction, starting from A3
    rd_axial = rd[0].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    rd_radius = rd.to_numpy().astype(float)
    # Also need to roll the COLUMNS (axis=1) in rd_radius since the circumferential orientation was rolled
    rd_radius = np.roll(rd_radius, roll_amount, axis=1)
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v7(rd_path):
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
    
def data_smoothing(OD, rd_axial, rd_circ_deg, rd_radius, circ_int=0.5, axial_int=0.5, circ_window=5, circ_smooth=0.001, axial_window=9, axial_smooth=0.00005):
    """
    ASME B31.8-2020 Nonmandatory Appendix R Estimating Strain in Dents recommends 
    the use of suitable data smoothing techniques in order to minimize the effect 
    of random error inherent with all geometric measurement techniques.
    
    This function applies the Savitzky-Golay filter on the data, then generates 
    spline curves that are evaluated at desired intervals.

    Parameters
    ----------
    OD : float
        pipeline nominal outside diameter, in
    rd_axial : array of floats
        1-D array containing the axial displacement, in
    rd_circ : array of floats
        1-D array containing the circumferential displacement, deg
    rd_radius : array of floats
        2-D array containing the radial values with shape (axial x circ), in
    circ_int : float
        the desired circumferential interval length for the output data, in. Default = 0.5
    axial_int : float
        the desired axial interval length for the output data, in. Default = 0.5
    circ_window : int
        the smoothing window (number of points to consider) for the circumferential 
        smoothing filter. Note: this must be an odd number. Default = 5
    circ_smooth : float
        the circumferential smoothing parameter for splines. Default = 0.001
    axial_window : int
        the smoothing window (number of points to consider) for the axial 
        smoothing filter. Note: this must be an odd number. Default = 9
    axial_smooth : float
        the axial smoothing parameter for splines. Default 0.00005
    
    Returns
    -------
    sd_axial : array of floats
        the smoothed axial displacement values in the fixed intervals, in.
    sd_circ : array of floats
        the smoothed circumferential displacement values in the fixed intervals, in.
    sd_radius : array of floats
        the smoothed radial values in the fixed intervals, in.
    """

    # print((time_ref + '====== DATA SMOOTHING ======') % (time.time() - time_start))
    
    filter_polyorder = 3
    filter_mode = 'wrap'
    spline_deg = 3
    
    OR = OD/2
    
    # Convert the circumferential orientation from degrees to radians
    rd_circ = np.deg2rad(rd_circ_deg)
    
    # Smoothed file output interval to 0.50" by 0.50"
    int_len_circ     = circ_int      # Circumferential interval length, in
    int_len_axial    = axial_int     # Axial interval length, in
    
    int_count_circ   = math.ceil((2*math.pi*OR/int_len_circ)/4)*4                       # Find the circumferential interval closest to int_len_circ on a multiple of four
    int_count_axial  = int(max(rd_axial) / int_len_axial)                               # Find the axial interval closest to int_len_axial
    
    int_points_circ  = np.linspace(0, 2*math.pi, int_count_circ, False)                 # Create circumferential interval for one pipe circumference in radians
    int_points_axial = np.linspace(0, round(max(rd_axial), 0), int_count_axial, False)  # Create equally spaced axial points for smoothing
    
    sd_radius_circ1  = np.zeros(rd_radius.shape)                                        # First pass of smoothing will have the same number of data points as the raw data
    sd_radius_axial1 = np.zeros(rd_radius.shape)                                        # First pass of smoothing will have the same number of data points as the raw data
    
    sd_radius_circ2  = np.zeros((len(rd_axial), len(int_points_circ)))                  # Second pass of smoothing will have the desired interval number of data points
    sd_radius_axial2 = np.zeros((len(int_points_axial), len(int_points_circ)))          # Second pass of smoothing will have the desired interval number of data points
    
    
    # sd_radius_circ2  = np.zeros((int_count_circ, int_count_axial))                      # Second pass of smoothing will have the desired interval number of data points
    # sd_radius_axial2 = np.zeros((int_count_circ, int_count_axial))                      # Second pass of smoothing will have the desired interval number of data points
    
    # Step 1: Circumferential profiles smoothing and spline functions
    for axial_index, circ_profile in enumerate(rd_radius[:,0]):
        circ_profile    = rd_radius[axial_index, :]
        circ_filter     = savgol_filter(x=circ_profile, window_length=circ_window, polyorder=filter_polyorder, mode=filter_mode)
        circ_spline     = splrep(x=rd_circ, y=circ_filter, k=spline_deg, s=circ_smooth, per=1) # Data is considered periodic since it wraps around, therefore per=1
        sd_radius_circ1[axial_index, :] = splev(x=rd_circ, tck=circ_spline)
        
    # Step 2: Axial profiles smoothing and spline functions
    for circ_index, axial_profile in enumerate(rd_radius[0,:]):
        axial_profile   = rd_radius[:, circ_index]
        axial_filter    = savgol_filter(x=axial_profile, window_length=axial_window, polyorder=filter_polyorder)
        axial_spline    = splrep(x=rd_axial, y=axial_filter, k=spline_deg, s=axial_smooth)
        sd_radius_axial1[:, circ_index] = splev(x=rd_axial, tck=axial_spline)
        
    # Step 3: Create weighted average profiles from axial and circumferential profiles
    circ_err    = abs(sd_radius_circ1 - rd_radius)
    axial_err   = abs(sd_radius_axial1 - rd_radius)
    sd_radius_avg   = (circ_err * sd_radius_circ1 + axial_err * sd_radius_axial1)/(circ_err + axial_err)
    
    # Step 4: Final profiles with the desired intervals, starting with axial direction
    for axial_index, circ_profile in enumerate(sd_radius_avg[:,0]):
        circ_profile    = sd_radius_avg[axial_index, :]
        # circ_filter     = savgol_filter(x=circ_profile, window_length=circ_window, polyorder=filter_polyorder, mode=filter_mode) # Added this line for testing 04/18/2024
        circ_spline     = splrep(x=rd_circ, y=circ_profile, k=spline_deg, s=circ_smooth, per=1)
        sd_radius_circ2[axial_index, :] = splev(x=int_points_circ, tck=circ_spline)
        
    for circ_index, axial_profile in enumerate(sd_radius_circ2[0,:]):
        axial_profile = sd_radius_circ2[:, circ_index]
        axial_filter = savgol_filter(x=axial_profile, window_length=axial_window, polyorder=filter_polyorder)
        axial_spline = splrep(x=rd_axial, y=axial_filter, k=spline_deg, s=axial_smooth)
        sd_radius_axial2[:, circ_index] = splev(x=int_points_axial, tck=axial_spline)
    
    sd_axial  = int_points_axial
    sd_circ   = np.rad2deg(int_points_circ)
    sd_radius = sd_radius_axial2
    
    return sd_axial, sd_circ, sd_radius

def abaqus_input_file(dent_ID, results_path, OD, WT, SMYS, sd_axial, sd_circ, sd_radius):
    global inp_file_name
    global inp_file_path
    global int_review_path
    
    # lim_cc and lim_ax is the amount of nodes to display applied to both sides in the circumferential and axial directions, respectively
    # For example, using lim_cc = 20 and lim_ax 40 will result in a field of points of (circ x axial) = (40 x 80)
    # lim_cc needs to be half of the circumference
    # lim_cc      = 20
    # circ_interval = []
    # for i in range(len(sd_circ)-1):
    #     circ_interval.append(sd_circ[i+1] - sd_circ[i])
    # circ_interval_avg = np.deg2rad(np.mean(circ_interval))
    # lim_cc = math.ceil(1/2*2*math.pi*(OD/2)/circ_interval_avg/2)
    lim_cc = int(sd_circ.shape[0]/4)
    
    # lim_ax needs to be a span of 2*OD of the axial
    # lim_ax      = 40
    axial_interval = []
    for i in range(len(sd_axial)-1):
        axial_interval.append(sd_axial[i+1] - sd_axial[i])
    axial_interval_avg = np.mean(axial_interval)
    lim_ax = int(math.ceil(2*OD/axial_interval_avg/2))
    
    num_cal     = sd_circ.size
    num_nodes   = sd_radius.size
    def_angl    = 60
    bar_stress  = SMYS # This is based on the SMYS of the pipe
    
    # Create the *Node array
    z_len = sd_axial.size
    theta_len = sd_circ.size
    inp_num_nodes = sd_radius.size
    
    inp_node = []
    inp_node_i = 0
    for iz in range(0,z_len):
        for it in range(0, theta_len):
            inp_node.append(str(inp_node_i + 1) + ", " + str(round(sd_radius[iz,it],3)) + ", " + str(round(sd_circ[it],3)) + ", " + str(round(sd_axial[iz],3)))
            inp_node_i += 1
    
    # Create the *Element and *Elgen arrays
    el1 = 0
    el4 = 0
    j = 0
    inp_element = []
    inp_elgen = []
    
    theta_len = sd_circ.size
    while el4 < inp_node_i:
        # A
        j += 1
        el1 += 1
        el2 = el1 + 1
        el3 = el2 + theta_len
        el4 = el3 - 1
        inp_element.append(str(el1)+", "+str(el1)+", "+str(el2)+", "+str(el3)+", "+str(el4))
        inp_elgen.append(str(el1)+", "+str(theta_len - 1)+", 1, 1")
        # B
        j += 1
        el2 = el1
        el3 = el2 + theta_len
        el4 = el3 + theta_len - 1
        el1 = el1 + theta_len - 1
        inp_element.append(str(el1)+", "+str(el1)+", "+str(el2)+", "+str(el3)+", "+str(el4))
        inp_elgen.append(str(el1)+", 1, 1, 1")
    
    # Create the boundary condition nodes, *BCNodes
    # The first set of nodes at the start Z position, and the second at the last Z position
    inp_bcnode = list(range(1, theta_len + 1)) + list(range(inp_node_i - theta_len, inp_node_i + 1))
    
    # Wall thickness of the pipe
    inp_wt = WT
    
    # Loop through the inp_file and search for the following keywords
    # - #Nodes#
    # - #Elements#
    # - #BCNodes#
    # - #Elgen#
    # - #All_Elements#
    # - #Wall_Thickness#
    
    # Create a copy of the Input Deck Template text file
    inp_file_template_str = templates_path + 'Input Deck Template.inp'
    inp_file_name = "ID" + str(dent_ID)
    inp_file_path = results_path + 'Abaqus Results'
    # Create a folder for the Abaqus files
    os.mkdir(inp_file_path)
    inp_file_path = inp_file_path + '/'
    inp_file_deck = inp_file_path + inp_file_name + '.inp'
    # Load the Input Deck Template text file
    inp_file_template = open(inp_file_template_str, 'r')
    # Create a leaf directory and all intermediate ones
    # Each input deck will have its own folder to store all Abaqus files
    os.makedirs(os.path.dirname(inp_file_deck), exist_ok=True)
    inp_file = open(inp_file_deck, 'w')
    
    # Keep track of the placeholder strings to later remove them
    inp_line_list = ["#Nodes#\n","#Elements#\n","#Elgen#\n","#BCNodes#\n","#All_Elements#\n","#All_Elements#\n","#Wall_Thickness#, 5\n"]
    inp_line_index = []
    inp_file_contents = inp_file_template.readlines()
    for f_index, line in enumerate(inp_file_contents):
        # Search for #Nodes#
        if line == inp_line_list[0]:
            # Print all of the inp_node values to the Input Deck File
            for n_index, n_value in enumerate(inp_node):
                inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
        # Search for #Elements#
        if line == inp_line_list[1]:
            # Print all of the inp_node values to the Input Deck File
            for n_index, n_value in enumerate(inp_element):
                inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
        # Search for #Elgen#
        if line == inp_line_list[2]:
            # Print all of the inp_node values to the Input Deck File
            for n_index, n_value in enumerate(inp_elgen):
                inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
        # Search for #BCNodes#
        if line == inp_line_list[3]:
            # Print all of the inp_node values to the Input Deck File
            for n_index, n_value in enumerate(inp_bcnode):
                inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
        # Search for #All_Elements#
        if line == inp_line_list[4]:
            # Print the value for #All_Elements#
            inp_line_index.append(f_index)
            inp_file_contents.insert(f_index + 1,"1, "+str(inp_num_nodes - theta_len + 1)+", 1\n")
        # Search for #Wall_Thickness#
        if "#Wall_Thickness#" in line:
            # Print the value for #Wall_Thickness#
            inp_line_index.append(f_index)
            inp_file_contents.insert(f_index + 1,str(inp_wt)+", 5 \n")
        if "#Pressure#" in line:
            # Print the value for #Pressure#
            inp_line_index.append(f_index)
            bar_press = (2*WT*bar_stress)/OD
            inp_file_contents[f_index] = inp_file_contents[f_index].replace("#Pressure#", str(round(bar_press,4)))
    # Remove the placeholder strings
    for i in inp_line_list:
        inp_file_contents.remove(i)
    
    inp_file.writelines(inp_file_contents)
    inp_file.close()
    
    # I need to print a file with the essential information needed in the Post-Processing section
    # Part C: Isolated Elements View
    # cc_lim    - Circumferential Limit
    # ax_lim    - Axial Limit
    # num_cal   - Number of Calipers
    # num_nodes - Total Number of Nodes
    # def_angl  - Angle for Isometric View
    
    # Create an Internal Review folder to save all of the images and reports
    int_review_path = inp_file_path + 'Internal Review/'
    os.mkdir(int_review_path)
    
    # Create a node_info.txt file to export theses values
    info_file_name = "node_info"
    info_file_deck = int_review_path + info_file_name + ".txt"
    info_file = open(info_file_deck, "w")
    # Data to write in
    info_file_contents = []
    # # Feature information
    # info_file_contents.append('======== NODE INFORMATION ========' + '\n')
    # info_file_contents.append('Feature ID = ' + str(dent_ID) + '\n\n')
    # info_file_contents.append('The Circumferential Limit (lim_cc) and Axial Limit (lim_ax) specify the window of nodes to display in the Internal Review.\n')
    # info_file_contents.append('For example, using lim_cc = 20 and lim_ax 40 will result in a field of points of (circ x axial) = (40 x 80)\n')
    # info_file_contents.append('For best results, it is recommended to use a field of view containing half of the circumference and 2*OD of the axial.\n')
    # info_file_contents.append('============= VALUES =============' + '\n')
    # lim_cc - Circumferential Limit
    info_file_contents.append('lim_cc     = ' + str(lim_cc) + "\n")
    # lim_ax - Axial Limit
    info_file_contents.append('lim_ax     = ' + str(lim_ax) + "\n")
    # num_cal - Number of Calipers
    info_file_contents.append('num_cal    = ' + str(num_cal) + "\n")
    # num_nodes - Total Number of Nodes
    info_file_contents.append('num_nodes  = ' + str(num_nodes) + "\n")
    # def_angl - Angle for Isometric View
    info_file_contents.append('def_angl   = ' + str(def_angl) + "\n")
    # bar_stress - Barlow's equation for Hoop Stress to calculate SCF
    info_file_contents.append('bar_stress = ' + str(bar_stress) + "\n")
    info_file.writelines(info_file_contents)
    info_file.close()
    
def abaqus_submit(dent_ID):
    """
    This function submits the input file to Abaqus and runs an external Python script to generate the result outputs.

    Parameters
    ----------
    dent_ID : str
        the dent identification found in the Dent Index spreadsheet.

    Returns
    -------
    None.

    """
    print((time_ref + '===== SUBMIT TO ABAQUS =====') % (time.time() - time_start))
    
    # In order to maintain the same Command Prompt environment, need to do both the
    # cd directory change and the Abaqus command in one os.system wrapper.
    command_str = "abaqus job=" + inp_file_name + " cpus=2"
    command_dir = "cd " + os.getcwd() + "/" + inp_file_path
    command = command_dir + " && " + command_str
    # Clear the existing history before running the command
    # readline.clear_history()
    os.system('cls')
    os.system(command)
    os.system('cls')
    # readline.clear_history()
    
    print((time_ref + 'Submitted project: %s') % (time.time() - time_start, str(dent_ID)))
    
    # First check that the file exists, since there may be a delay before it is created
    sta_path = inp_file_path + inp_file_name + ".sta"
    file_check_time = time.time()
    
    while not os.path.exists(sta_path):
        time.sleep(5)
        print((time_ref + 'Waiting for Abaqus to create the .sta file.') % (time.time() - time_start))
        
        # Check that it has not exceeded the time limit to prevent an infite loop
        if (time.time() - file_check_time)>time_limit:
            # End the script
            print((time_ref + '========== ERROR ==========') % (time.time() - time_start))
            sys.exit('Exceeded time limit of %.0f seconds to search for .sta file. Proceess aborted.' % (time_limit))
        
    print((time_ref + 'Abaqus has created the .sta file. Begin monitoring this file until Abaqus concludes.') % (time.time() - time_start))
    
    # Restart the time limit for monitoring the .sta file
    file_check_time = time.time()
    
    sta_file = open(sta_path, "r")
    sta_contents = sta_file.readlines()
    
    # Loop until " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n" shows up at the end
    str_success = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
    while sta_contents[-1] != str_success:
        # Wait for 30 seconds before opening the file again and checking
        time.sleep(10)
        # Reload the .sta file with its contents
        print((time_ref + 'Monitoring the .sta file.') % (time.time() - time_start))
        sta_file = open(sta_path, "r")
        sta_contents = sta_file.readlines()
        
        # Check that it has not exceeded the time limit to prevent an infite loop
        if (time.time() - file_check_time)>time_limit:
            # End the script
            print((time_ref + '========== ERROR ==========') % (time.time() - time_start))
            sys.exit('Exceeded time limit of %.0f seconds to search for .sta file. Proceess aborted.' % (time_limit))
    
    sta_file.close()
    print((time_ref + '===== SCF CALCULATION ======') % (time.time() - time_start))
    
    # Copy the abaqusMacros.py template file to the input deck folder
    script_path = templates_path
    script_name = 'abaqusMacros.py'
    script_file = script_path + script_name
    # Use the same destination as the Input File
    shutil.copy(script_file, inp_file_path)
    # Run the abaqusMacros.py script to do the following:
    # - Print 11 images from the .odb file for analysis
    # - Print the MaxPrincipal value for future SCF calculations
    command_str = "abaqus viewer noGUI=" + script_name
    command_dir = "cd " + os.getcwd() + "/" + inp_file_path
    command = command_dir + " && " + command_str
    # readline.clear_history()
    os.system('cls')
    os.system(command)
    os.system('cls')
    # readline.clear_history()
    
    # Wait for 30 seconds for all images to be created
    time.sleep(15)
    
def abaqus_results(dent_ID):
    """
    This function extracts the results from the job in Abaqus and returns the SCF.

    Parameters
    ----------
    dent_ID : str
        the dent identification found in the Dent Index spreadsheet.

    Raises
    ------
    flag
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    global flag
    file_new_name = "ID" + str(dent_ID) + "_"
    
    # Read the output data
    with open(int_review_path + 'report_MPs.rpt') as f:
        report_MPs = f.readlines()
        f.close()
    # Delete the first 3 rows since they do not contain data
    report_MPs.pop(0)
    report_MPs.pop(0)
    report_MPs.pop(0)
    # Remove the leading and trailing spaces, also the '\n' character
    report_MPs = [s.strip() for s in report_MPs]
    # Split by spaces
    report_MPs = [s.split(" ") for s in report_MPs]
    # Remove the empty list items
    report_MPs = [s for s in report_MPs if s != ['']]
    
    # Remove the spaces and convert string numbers into floats
    x1 = np.zeros(len(report_MPs))
    yMPs = np.zeros(len(report_MPs))
    for i in range(0,len(report_MPs)):
        report_MPs[i] = [float(s) for s in report_MPs[i] if s.strip()]
        x1[i] = report_MPs[i][0]
        yMPs[i] = report_MPs[i][1]
    
    # Read the output data
    with open(int_review_path + 'report_Radius.rpt') as f:
        report_Radius = f.readlines()
        f.close()
    # Delete the first 3 rows since they do not contain data
    report_Radius.pop(0)
    report_Radius.pop(0)
    report_Radius.pop(0)
    # Remove the leading and trailing spaces, also the '\n' character
    report_Radius = [s.strip() for s in report_Radius]
    # Split by spaces
    report_Radius = [s.split(" ") for s in report_Radius]
    # Remove the empty list items
    report_Radius = [s for s in report_Radius if s != ['']]
    
    # Remove the spaces and convert string numbers into floats
    x2 = np.zeros(len(report_Radius))
    R = np.zeros(len(report_Radius))
    for i in range(0,len(report_Radius)):
        report_Radius[i] = [float(s) for s in report_Radius[i] if s.strip()]
        x2[i] = report_Radius[i][0]
        R[i] = report_Radius[i][1]
    
    # Produce the Node Path Image
    plt.rcParams['font.size'] = 8
    plt.rcParams['lines.markersize'] = 0.5
    
    fig9, ax9 = plt.subplots(figsize=(3.43,2), dpi=200)
    ax9_1 = ax9.twinx()
    # First Y Axis
    ax9.plot(x1, yMPs, c='tab:blue', label='Max. Principal Stress',
             marker='o', markerfacecolor='k', markeredgecolor='k', markersize=1)
    ax9.set_xlabel('Position Z Along Pipe [in]')
    ax9.set_ylabel('Maximum Principal Stress [psi]')
    ax9.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax9.tick_params(axis='y', colors='tab:blue')
    ax9.yaxis.label.set_color('tab:blue')
    # Secondary Y Axis
    ax9_1.plot(x2, R, c='tab:orange',label='Axial Radial Profile',
             marker='o', markerfacecolor='r', markeredgecolor='r', markersize=1)
    ax9_1.set_ylabel('Radius [in]')
    ax9_1.tick_params(axis='y', colors='tab:orange')
    ax9_1.yaxis.label.set_color('tab:orange')
    # Save as the OR version
    fig9.savefig(int_review_path + file_new_name + '14_Nodal_Path_OR', bbox_inches='tight')
    
    # Adjust so that it saves as the full size
    plt.rcParams['font.size'] = 8
    plt.rcParams['lines.markersize'] = 0.5
    fig9.set_size_inches(10,4)
    fig9.suptitle('Max. Principal Stress OD along Axial Radial Profile')
    s1,sl1 = ax9.get_legend_handles_labels()
    s2,sl2 = ax9_1.get_legend_handles_labels()
    s = s1 + s2
    sl = sl1 + sl2
    ax9.legend(s,sl)
    fig9.savefig(int_review_path + file_new_name + '14_Nodal_Path', dpi=200)
    fig9.savefig('all_results/' + file_new_name + '_14_Nodal_Path.png')
    
    # Collect the SCF value from the information file
    with open(int_review_path + "node_info.txt") as f:
        node_info = f.readlines()
        f.close()
    
    node_val = [s.strip() for s in node_info]
    node_val = [s.split("=") for s in node_val if "=" in s]
    # Collect the Max Principal Stress Value in the OD
    MPs = [float(s[1]) for s in node_val if "max_val_OD" in s[0]]
    MPs = float(MPs[0])
    # Collect the SCF Values (for ID and OD), but only keep the largest
    SCF_ID = [float(s[1]) for s in node_val if "ID SCF" in s[0]]
    SCF_ID = float(SCF_ID[0])
    SCF_OD = [float(s[1]) for s in node_val if "OD SCF" in s[0]]
    SCF_OD = float(SCF_OD[0])
    SCF = max(SCF_ID, SCF_OD)
    # Collect the Unaveraged Max Principal Stress Value
    uMPs = [float(s[1]) for s in node_val if "Unavg MPs" in s[0]]
    uMPs = float(uMPs[0])
    # Collect the Unaveraged SCF Value
    uSCF = [float(s[1]) for s in node_val if "Unavg SCF" in s[0]]
    uSCF = float(uSCF[0])
    print((time_ref + 'Averaged SCF = %.2f | Unaveraged SCF = %.2f') % (time.time() - time_start,SCF,uSCF))
    
    # Quality Control Point
    # If the values disagree past a limit, then raise a flag
    scf_limit = 0.1
    scf_err = abs(uSCF - SCF)/uSCF
    if scf_err >= scf_limit:
        print((time_ref + 'Error: the comparison between the average and unaveraged SCF values exceeds 10%%') % (time.time() - time_start))
        flag.append('The comparison between the average and unaveraged SCF values exceeds 10%. Review the dent Abaqus .odb file for more information.')
        
    # print((time_ref + 'Saved contents to scf_values.xlsx') % (time.time() - time_start))
    
    return SCF, MPs

def graphing(dent_ID, rd_axial, rd_circ, rd_radius, sd_axial, sd_circ, sd_radius):
    # 08/04/2023 TEMPORARILY ADDED THIS TO REMOVE CROSSHAIR
    crosshair = False
    
    # print((time_ref + '========== GRAPHS ==========') % (time.time() - time_start))
    
    plot_title = "ID " + str(dent_ID) + " "
    file_new_name = "ID" + str(dent_ID) + "_"
    
    # Once this is ready to go, I want to put the entire plotting code in a separate python script.
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 8
    plt.rcParams['lines.markersize'] = 0.5
    # Make the levels based on the min and max from both raw and smooth
    # level_min = min([rd_radius.min(),sd_radius.min()])
    # level_max = max([rd_radius.max(),sd_radius.max()])
    # levels = np.linspace(level_min,level_max,11)
    
    levels_raw = np.linspace(rd_radius.min(),rd_radius.max(),17)
    levels_smooth = np.linspace(sd_radius.min(),sd_radius.max(),17)
    
    # To account for the change in Indexing due to undersampling, need to find the corresponding index of the Raw data
    # [Index, Value] in Circumferential Direction
    min_ind = np.unravel_index(np.argmin(sd_radius), sd_radius.shape)
    min_ind_ic = min(enumerate(rd_axial), key=lambda x: abs(x[1] - sd_axial[min_ind[0]]))
    # [Index, Value] in Longitudinal Direction
    min_ind_ia = min(enumerate(rd_circ), key=lambda x: abs(x[1] - sd_circ[min_ind[1]]))
    # print('min_ind      = [%i, %i]'%(min_ind))
    # Check the errors
    # z_err = 100*abs(sd_axial[min_ind[1]] - min_ind_ic[1])/min_ind_ic[1]
    # theta_err = 100*abs(sd_circ[min_ind[0]] - min_ind_ia[1])/min_ind_ia[1]
    # # Print the [Index, Value]
    # print('sd_axial          = (%i, %.2f)'%(min_ind[1], sd_axial[min_ind[1]]))
    # print('rd_axial          = (%i, %.2f)'%(min_ind_ic))
    # print('z_err        = %.2f %%'%(z_err))
    # print('sd_circ      = (%i, %.2f)'%(min_ind[0], sd_circ[min_ind[0]]))
    # print('rd_circ      = (%i, %.2f)'%(min_ind_ia))
    # print('theta_err    = %.2f %%'%(theta_err))
    
    # For the Longitudinal Plots, put the calipers immediately before and after the sd_circ value
    if sd_circ[min_ind[1]] >= min_ind_ia[1]: #<---------------------- CONFIRM THIS LOGIC
        min_ind_ia_LB = min_ind_ia[0]
        min_ind_ia_UB = min_ind_ia[0] + 1
    elif sd_circ[min_ind[1]] < min_ind_ia[1]:
        min_ind_ia_LB = min_ind_ia[0] - 1
        min_ind_ia_UB = min_ind_ia[0]
    
    # Plots for Internal Review
    
    # Contour Line with Filled Contour for Smooth Data
    graph_surface = int_review_path + file_new_name + '15_Surface_Smooth.png'
    fig1, cp1 = plt.subplots(figsize=(6.85,4), dpi=240)
    cpf1 = cp1.contourf(sd_axial, sd_circ, sd_radius.T, levels=levels_smooth)
    cbar1 = fig1.colorbar(cpf1)
    cbar1.set_label('Radius (in)')
    cp1.set_xlabel('Axial Displacement (in)')
    cp1.set_ylabel('Circumferential Orientation (deg)')
    # cp1.yaxis.set_tick_params(labelleft=False)
    cp1.set_yticks([])
    cp1.set_title(plot_title + 'Surface Plot (Radius) - Smooth')
    # Defect Profile Location Line
    if crosshair == True:
        cp1.axvline(x=sd_axial[min_ind[0]], color='red', linestyle='--', linewidth=0.5)
        cp1.axhline(y=sd_circ[min_ind[1]], color='red', linestyle='--', linewidth=0.5)
    # # Defect Limits
    # cp1.plot(sd_axial[north[1]],sd_circ[north[0]], color='r', marker='D', markersize=1)
    # cp1.plot(sd_axial[east[1]],sd_circ[east[0]], color='r', marker='D', markersize=1)
    # cp1.plot(sd_axial[south[1]],sd_circ[south[0]], color='r', marker='D', markersize=1)
    # cp1.plot(sd_axial[west[1]],sd_circ[west[0]], color='r', marker='D', markersize=1)
    fig1.savefig(graph_surface)
    
    # Contour Line with Filled Contour for Raw Data
    graph_surface2 = int_review_path + file_new_name + '16_Surface_Raw.png'
    fig2, cp2 = plt.subplots(figsize=(6.85,4), dpi=240)
    cpf2 = cp2.contourf(rd_axial, rd_circ, rd_radius.T, levels=levels_raw)
    cbar2 = fig2.colorbar(cpf2)
    cbar2.set_label('Radius (in)')
    cp2.set_xlabel('Axial Displacement (in)')
    cp2.set_ylabel('Circumferential Orientation (deg)')
    cp2.yaxis.set_tick_params(labelleft=False)
    cp2.set_yticks([])
    cp2.set_title(plot_title + 'Surface Plot (Radius) - Raw')
    # Defect Profile Location Line
    if crosshair == True:
        cp2.axvline(x=sd_axial[min_ind[0]], color='red', linestyle='--', linewidth=0.5)
        cp2.axhline(y=sd_circ[min_ind[1]], color='red', linestyle='--', linewidth=0.5)
    # # Defect Limits
    # cp2.plot(sd_axial[north[1]],sd_circ[north[0]], color='r', marker='D', markersize=1)
    # cp2.plot(sd_axial[east[1]],sd_circ[east[0]], color='r', marker='D', markersize=1)
    # cp2.plot(sd_axial[south[1]],sd_circ[south[0]], color='r', marker='D', markersize=1)
    # cp2.plot(sd_axial[west[1]],sd_circ[west[0]], color='r', marker='D', markersize=1)
    fig2.savefig(graph_surface2)
    
    # # Contour Lines
    # fig2, cp2 = plt.subplots(figsize=(6.85,4), dpi=240)
    # cpl2 = cp2.contour(sd_axial, sd_circ, sd_radius)
    # cbar2 = fig2.colorbar(cpl2)
    # cp2.clabel(cpl2, inline=True, fmt='%1.1f', fontsize=8)
    # cp2.set_xlabel('Displacement $Z$')
    # cp2.set_ylabel(r'Orientation $\theta$')
    # cp2.set_title('Contour Plot of Unrolled Surface (Radius)')
    
    # # ----- Dent Profile BEGIN -----
    # graph_dent_profile = int_review_path + 'Profile.png'
    # fig3, lp3 = plt.subplots(nrows=2, ncols=2,figsize=(6.85, 4), dpi=240)
    # fig3.suptitle(plot_title + 'Profile', fontsize=16)
    
    # # Radius vs Orientation Theta
    # lp3[0,0].scatter(sd_circ, sd_radius[:, min_ind[1]])
    # lp3[0,0].set_ylabel('Radius')
    # lp3[0,0].set_ylim([radius_min - 1, radius_max + 1])
    
    # # Radius vs Position Z
    # lp3[0,1].scatter(sd_axial, sd_radius[min_ind[0],:])
    # lp3[0,1].set_ylim([radius_min - 1, radius_max + 1])
    
    # # Radius vs Orientation Theta (Full Radius)
    # lp3[1,0].scatter(sd_circ, sd_radius[:, min_ind[1]])
    # lp3[1,0].set_xlabel(r'Orientation $\theta$')
    # lp3[1,0].set_ylabel('Radius (Starting from 0)')
    # lp3[1,0].set_ylim([0, radius_max + 1])
    
    # # Radius vs Position Z (Full Radius)
    # lp3[1,1].scatter(sd_axial, sd_radius[min_ind[0],:])
    # lp3[1,1].set_xlabel('Position $Z$')
    # lp3[1,1].set_ylim([0, radius_max + 1])
    
    # fig3.savefig(graph_dent_profile)
    
    # # Circumferential Plot
    # graph_circ_all = int_review_path + 'Circumferential_All.png'
    # graph_circ_rel = int_review_path + 'Circumferential_Relative.png'
    # fig4, sp4 = plt.subplots(figsize=(10,4), dpi=1200)
    # fig4.suptitle(plot_title + 'Circumferential Profile', fontsize=16)
    # sp4.scatter(rd_circ, rd_radius[:, min_ind_ic[0]], label='Raw Data at Axial Position ' + str(round(min_ind_ic[1],2)) + ' in')
    # sp4.scatter(sd_circ, sd_radius[:, min_ind[1]], label='Smooth Data at Axial Position ' + str(round(sd_axial[min_ind[1]],2)) + ' in')
    # sp4.set_ylabel('Radius (in)')
    # sp4.set_ylim([0, radius_max + 1])
    # sp4.set_xlabel(r'Orientation $\theta$ (degrees)')
    # sp4.legend()

    # fig4.savefig(graph_circ_all)
    # sp4.set_ylim([radius_min - 1, radius_max + 1])
    # fig4.savefig(graph_circ_rel)
    
    # Circumferential Polar Plot
    graph_circ_polar_all = int_review_path + file_new_name + '17_Circumferential_Polar_All.png'
    graph_circ_polar_rel = int_review_path + file_new_name + 'Circumferential_Polar_Relative.png'
    fig5, sp5 = plt.subplots(figsize=(6.85,4), dpi=1200, subplot_kw={'projection':'polar'})
    fig5.suptitle(plot_title + 'Polar Circumferential Profile', fontsize=12)
    sp5.scatter(rd_circ*np.pi/180, rd_radius[min_ind_ic[0],:], label='Raw at ' + str(round(min_ind_ic[1],2)) + ' in')
    sp5.scatter(sd_circ*np.pi/180, sd_radius[min_ind[0],:], label='Smooth at ' + str(round(sd_axial[min_ind[0]],2)) + ' in')
    sp5.set_ylim([0, sd_radius.max() + 1])
    sp5.set_rlabel_position(0)
    sp5.set_theta_zero_location("N")
    sp5.grid(True)
    angle = np.deg2rad(60)
    sp5.legend(loc="lower left", bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))
    
    fig5.savefig(graph_circ_polar_all)
    fig5.savefig('all_results/' + file_new_name + '_17_Circumferential_Polar_All.png')
    # Adjust the Y Lim to zoom in
    sp5.set_ylim([sd_radius.min() - 0.1, sd_radius.max() + 0.1])
    fig5.savefig(graph_circ_polar_rel)
    fig5.savefig('all_results/' + file_new_name + '_17_Circumferential_Polar_Relative.png')
    
    # Longitudinal Plot
    # graph_long_all = int_review_path + 'Longitudinal_All.png'
    graph_long_rel = int_review_path + file_new_name + '18_Longitudinal_Relative.png'
    fig6, sp6 = plt.subplots(figsize=(6.85,4), dpi=1200)
    fig6.suptitle(plot_title + 'Longitudinal Profile', fontsize=16)
    sp6.scatter(rd_axial, rd_radius[:,min_ind_ia_LB], label='Raw Data - Lower Bound Caliper at Deg ' + str(round(rd_circ[min_ind_ia_LB],2)), s=0.2)
    sp6.scatter(sd_axial, sd_radius[:,min_ind[1]], label='Smooth Data at Deg ' + str(round(sd_circ[min_ind[1]],2)), s=0.2)
    sp6.scatter(rd_axial, rd_radius[:,min_ind_ia_UB], label='Raw Data - Upper Bound Caliper at Deg ' + str(round(rd_circ[min_ind_ia_UB],2)), s=0.2)
    sp6.set_ylabel('Radius (in)')
    # sp6.set_ylim([0,radius_max + 1])
    sp6.set_xlabel('Axial Displacement (in)')
    sp6.legend()
    
    # fig6.savefig(graph_long_all)
    sp6.set_ylim([sd_radius.min() - 0.1, sd_radius.max() + 0.1])
    fig6.savefig(graph_long_rel)
    fig6.savefig('all_results/' + file_new_name + '_18_Longitudinal_Relative.png')
    
    # Plots for the Output Report
    
    # # Circumferential Plot
    # graph_circ_OR = int_review_path + 'Circumferential_OR.png'
    # fig7, ax7 = plt.subplots(figsize=(10,4), dpi=1200)
    # ax7.plot(sd_circ, sd_radius[:, min_ind[1]], label='Smooth Data')
    # ax7.set_ylabel('Radius (in)')
    # ax7.set_ylim([0, radius_max + 1])
    # ax7.set_xlabel(r'Orientation $\theta$ (degrees)')
    # fig7.savefig(graph_circ_OR)
    
    # # Longitudinal Plot
    # graph_long_OR = int_review_path + 'Longitudinal_OR.png'
    # fig8, ax8 = plt.subplots(figsize=(3.43,2), dpi=200)
    # ax8.plot(sd_axial, sd_radius[min_ind[0],:], label='Smooth Data')
    # ax8.set_ylabel('Radius (in)')
    # ax8.set_ylim([0, radius_max + 1])
    # ax8.set_xlabel('Position Z (inches)')
    # fig8.savefig(graph_long_OR)
    
def process_dent(dent_ID, dent_path, results_path_og, ili_format, OD, WT, SMYS, circ_int = 0.5, axial_int = 0.5, smoothing=True, time_start_og=0):
    # ADDED LAST TWO PARAMETERS 6/28 TESTING SMOOTHING 
    # NEED TO UPDATE THE SUMMARY HERE 9/12/2023 <--------
    """
    Specify the individual dent to process and generate an SCF.

    Parameters
    ----------
    dent_ID : str
        the dent identification found in the Dent Index spreadsheet.
    dent_path : str
        the path to the dent raw data.
    results_path : str
        the path to output all of the results from the PDMAC.
    dent_format : str
        the type of dent formatting used depending on the ILI vendor (Baker Hughes, Enduro, Entegra, Onestream, Quest, Rosen, or TDW).
    smoothing : bool
        indicate if to apply smoothing techniques to the dent file, TRUE or FALSE.
    time_start : float
        the time start for the project. Default = 0

    Returns
    -------
    scf : float
        the Stress Concentration Factor
    MPs : float
        the Maximum Principal Stress used to determine the SCF
    df_data : DataFrame
        the DataFrame containing all of the smoothed radial dent data. The shape is: (Axial x Circ)
    flag : TBD

    """
    
    global time_start
    global time_limit
    global templates_path
    global results_path
    global time_ref
    global flag
    
    time_start = time_start_og
    results_path = results_path_og
    time_limit = 60*60 # Normally 10 minutes
    
    templates_path = 'templates/'
    time_ref = '%03d | '
    flag = []
    
    print((time_ref + '========== PDMAC ===========') % (time.time() - time_start))
    
    # Load the raw data information
    if ili_format.lower() == 'baker hughes':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v1(dent_path)
    elif ili_format.lower() == 'enduro':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v1(dent_path)
    elif ili_format.lower() == 'entegra':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v4(dent_path, OD/2)
    elif ili_format.lower() == 'onestream':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v1(dent_path)
    elif ili_format.lower() == 'quest':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v1(dent_path)
    elif ili_format.lower() == 'rosen':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v1(dent_path)
    elif ili_format.lower() == 'tdw':
        rd_axial, rd_circ, rd_radius = collect_raw_data_TDW(dent_path, OD/2)
    elif ili_format.lower() == 'tdw2':
        rd_axial, rd_circ, rd_radius = collect_raw_data_TDW2(dent_path, OD/2)
    elif ili_format.lower() == 'pbf':
        rd_axial, rd_circ, rd_radius = collect_raw_data_PBF(dent_path)
    elif ili_format.lower() == 'campos':
        rd_axial, rd_circ, rd_radius = collect_raw_data_CAMPOS(dent_path, OD/2)
    elif ili_format.lower() == 'southern':
        rd_axial, rd_circ, rd_radius = collect_raw_data_v7(dent_path)
    else:
        raise Exception('ILI format %s was not found.' % (ili_format))
        
    # NEXT STEPS IS TO MAKE THIS ADJUST AUTOMATICALLY DEPENDING ON THE RESOLUATION OF THE DATA COLLECTION <---------------------------------------------------
    # Parameters for smoothing
    # axial_window = 13
    # axial_smooth = 0.001
    # circ_smooth = 0.003
    # circ_window = 5
    axial_window = 9
    axial_smooth = 0.00005
    # circ_smooth = 0.0002
    circ_smooth = 0.001
    circ_window = 5
    
    # # Remove calipers that are having issues
    # remove_calipers = [7]
    # for i in remove_calipers:
    #     rd_radius_1 = rd_radius.copy()
    #     rd_radius_1[:,i] = (rd_radius_1[:,i-1] + rd_radius_1[:,i+1])/2
    
    # Perform data smoothing on the raw data
    if smoothing == True:
        sd_axial, sd_circ, sd_radius = data_smoothing(OD, rd_axial, rd_circ, rd_radius, circ_int, axial_int, circ_window, circ_smooth, axial_window, axial_smooth)
        
    else:
        sd_axial  = rd_axial.copy()
        sd_circ   = rd_circ.copy()
        sd_radius = rd_radius.copy()
        
    df_data = pd.DataFrame(data=sd_radius, index=sd_axial, columns=sd_circ)
        
    # Create the Abaqus Input File
    abaqus_input_file(dent_ID, results_path, OD, WT, SMYS, sd_axial, sd_circ, sd_radius)
    
    # Export radii files for review
    if smoothing == True:
        # Export the raw data file
        test_rd_output = pd.DataFrame(data=rd_radius, index=rd_axial, columns=rd_circ)
        test_rd_output.to_excel(excel_writer = inp_file_path + 'Raw_Data.xlsx')
        
        # # Export the smooth data file
        test_sd_output = pd.DataFrame(data=sd_radius, index=sd_axial, columns=sd_circ)
        test_sd_output.to_excel(excel_writer = inp_file_path + 'Smoothed_Data.xlsx')
        
        # # Export the smooth data to the MD49 folder
        # test_sd_output.to_csv('MD49/' + str(dent_path.split('\\')[-2]) + '-' + str(dent_ID) + '.csv')
    else:
        # # Export the raw data file
        test_rd_output = pd.DataFrame(data=rd_radius, index=rd_axial, columns=rd_circ)
        test_rd_output.to_excel(excel_writer = inp_file_path + 'Raw_Data.xlsx')
        
        # # Export the raw data to the MD49 folder
        # test_rd_output.to_csv('MD49/' + str(dent_path.split('\\')[-2]) + '-' + str(dent_ID) + '.csv')
    
    # Submit to Abaqus
    abaqus_submit(dent_ID)
    
    # Collect the SCF value and generate the figures
    scf, MPs = abaqus_results(dent_ID)
    # scf = 0
    # MPs = 0

    # Export graphs for analysis
    graphing(dent_ID, rd_axial, rd_circ, rd_radius, sd_axial, sd_circ, sd_radius)
    
    return scf, MPs, df_data, flag
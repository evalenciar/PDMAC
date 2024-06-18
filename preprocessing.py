import pandas as pd
import numpy as np
import math
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
import os

def collect_raw_data_v1(rd_path):
    # Vendor 1
    # Works with: Baker Hughes, Enduro, Onestream, Quest, and Rosen

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
    # Vendor 2
    # Works with: 

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

def collect_raw_data_v3(rd_path, IR):
    # Vendor 3
    # Works with: TDW

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
    # Vendor 4
    # Works with: Entegra

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

def collect_raw_data_v5(rd_path, IR):
    # Vendor 5 (created on 09/19/2022)
    # Works with: TDW (similar to original TDW, minus rows 2 and 3)

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

def collect_raw_data_v6(rd_path):
    # Vendor 6 (created on 10/21/2022)
    # Works with: PBF

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
    # Vendor 7 (created on 04/11/2024)
    # Works with: Southern Company

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

def collect_raw_data_v8(rd_path, IR):
    # Vendor 8
    # Works with: Campos

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

class ImportData:
    def __init__(self, rd_path, ILI_format, OD=None):
        self.name = rd_path.split('/')[-1].split('.')[0]    # Get the filename
        self.path = rd_path
        self.ILI_format = str(ILI_format)

        ILI_format_list = ['Baker Hughes', 'Enduro', 'Entegra', 'Onestream', 'Quest',
                           'Rosen', 'TDW', 'TDW (v2)', 'PBF', 'Campos', 'Southern']

        # Load the raw data information
        if ILI_format.lower() == 'baker hughes':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'enduro':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'entegra':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v4(rd_path, OD/2)
        elif ILI_format.lower() == 'onestream':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'quest':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'rosen':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'tdw':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v3(rd_path, OD/2)
        elif ILI_format.lower() == 'tdw2':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v5(rd_path, OD/2)
        elif ILI_format.lower() == 'pbf':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v6(rd_path)
        elif ILI_format.lower() == 'campos':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v8(rd_path, OD/2)
        elif ILI_format.lower() == 'southern':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v7(rd_path)
        else:
            raise Exception('ILI format %s was not found. Use one of the following: %s' % (ILI_format, ', '.join(ILI_format_list)))
        
        self.axial = rd_axial
        self.circ = rd_circ
        self.radius = rd_radius

    def smooth_data(self, OD, circ_int=0.5, axial_int=0.5, circ_window=5, circ_smooth=0.001, axial_window=9, axial_smooth=0.00005):
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
        
        rd_axial = self.axial
        rd_circ_deg = self.circ
        rd_radius = self.radius
        
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
        
        # Save parameters
        self.OD = OD
        self.circ_int = circ_int
        self.axial_int = axial_int
        self.circ_window = circ_window
        self.circ_smooth = circ_smooth
        self.axial_window = axial_window
        self.axial_smooth = axial_smooth

        self.s_axial = sd_axial
        self.s_circ = sd_circ
        self.s_radius = sd_radius

    def create_input_file(self, WT, SMYS, results_path='results/', templates_path='templates/'):
        global inp_file_name
        global inp_file_path
        global int_review_path
        
        dent_ID     = self.name
        OD          = self.OD
        sd_axial    = self.s_axial
        sd_circ     = self.s_circ
        sd_radius   = self.s_radius
        inp_wt      = WT
        num_cal     = sd_circ.size
        num_nodes   = sd_radius.size
        def_angl    = 60
        bar_stress  = SMYS # This is based on the SMYS of the pipe

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
        
        # Loop through the inp_file and search for the following keywords
        # - #Nodes#
        # - #Elements#
        # - #BCNodes#
        # - #Elgen#
        # - #All_Elements#
        # - #Wall_Thickness#
        
        # Create a copy of the Input Deck Template text file
        inp_file_template_str = templates_path + 'Input Deck Template.inp'
        inp_file_name = "ID-" + str(dent_ID)
        inp_file_path = results_path + 'FEA Results'
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
        # int_review_path = inp_file_path + 'Internal Review/'
        int_review_path = inp_file_path
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

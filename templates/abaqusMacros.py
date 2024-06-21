# -*- coding: mbcs -*-
# Do not delete the following import lines
from odbAccess import *
from abaqus import *
from abaqusConstants import *
from visualization import *
from viewerModules import * # Temp
from driverUtils import executeOnCaeStartup # Temp
import xyPlot
import displayGroupOdbToolset as dgo
import __main__
import os
import glob
# import matplotlib.pyplot as plt
import numpy as np

# DEBUGGING
# Change Console working directory
# os.chdir()
# os.chdir('C:\\Users\\intern\\ADV Integrity\\ADV File Share - Software Development\\Excel ILI Dent Processing (PDMAC Tool)\\PDMAC')

# Open the only .odb file in the directory
filePath = glob.glob(os.getcwd() + '\\*.odb')[0]
# Collect the file name, example: Dent1234
fileName = filePath.split('\\')[-1].split('.')[0]
# Keep track of images
img_id = 0
img_labels = ['1_MPS_XYZ_0deg','2_MPS_XYZ_90deg','3_MPS_XYZ_180deg','4_MPS_XYZ_270deg',
              '5_COORD_RTZ_0deg','6_COORD_RTZ_90deg','7_COORD_RTZ_180deg','8_COORD_RTZ_270deg',
              '9_ISO_IDMAX_ID','10_ISO_IDMAX_OD',
              '11_ISO_ODMAX_ID','12_ISO_ODMAX_OD',
              '13_MPS_Dent']
img_labels = [fileName + '_' + s for s in img_labels]
# Create the folder to save all of the images
# int_review_path = 'Internal Review/'
# int_review_path = os.getcwd() + '/'
int_review_path = os.path.join(os.pardir, os.getcwd()) + '/'
# os.mkdir(int_review_path)

session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=300, height=150)

# session.Viewport(name='Viewport: 1', origin=(0.0, 0.0))

o1 = session.openOdb(
    name= filePath,
    readOnly=False)

session.viewports['Viewport: 1'].setValues(displayedObject=o1)
# session.viewports['Viewport: 1'].maximize()

# Common Options -> Adjust Visible Edges to Free Edges
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=FREE)

# Change the perspective so that it is not distorted
session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)

# View the Max. Principal Stress
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(
    INVARIANT, 'Max. Principal'), )

# Switch to view OD data (SPOS)
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    sectionResults=USE_TOP)

# Remove the compass, title block, and state block
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(title=OFF,
    state=OFF, compass=OFF, legendDecimalPlaces=0, legendNumberFormat=FIXED,
    legendFont='-*-verdana-medium-r-normal-*-*-180-*-*-p-*-*-*')

# Remove the legend bounding box
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(
    legendBox=OFF)

# Make the background solid white
session.graphicsOptions.setValues(backgroundStyle=SOLID,
    backgroundColor='#FFFFFF')

# Add annotation with dent name. Original: offset=(240, 125)
t = session.odbs[filePath].userData.Text(
    name='Text-1', text=fileName, offset=(150, 125),
    font='-*-verdana-medium-r-normal-*-*-200-*-*-p-*-*-*', box=ON)
session.viewports['Viewport: 1'].plotAnnotation(annotation=t)

# =============================================================================
# A: 4 Images of the Max. Principal Stress on the OD at 90deg intervals
# =============================================================================

# Initial Position: Normal to the X plane with pos Z going right.
session.viewports['Viewport: 1'].view.setValues(session.views['Left'])
session.viewports['Viewport: 1'].view.zoom(zoomFactor=0.75, mode=ABSOLUTE)
session.viewports['Viewport: 1'].view.pan(xFraction=0.12, yFraction=0)

# Print Output PNG 4 times for 0, 90, 180, 270 deg
for i in range(img_id,img_id + 4):
    # Print Output PNG
    session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    session.printToFile(fileName=int_review_path + img_labels[i], format=PNG, canvasObjects=(
        session.viewports['Viewport: 1'], ))

    # Rotate by 90deg
    session.viewports['Viewport: 1'].view.rotate(xAngle=0, yAngle=0, zAngle=90, mode=MODEL)

# Image ID becomes the last value of i
img_id = i+1

# =============================================================================
# B: 4 Images of COORD 1 (Radius) at the same 90 degree intervals
# =============================================================================

# Create Cylindrical Coordinate System
cylName = 'CSYS-1'
odb = session.odbs[filePath]
scratchOdb = session.ScratchOdb(odb)
scratchOdb.rootAssembly.DatumCsysByThreePoints(name=cylName,
    coordSysType=CYLINDRICAL, origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0),
    point2=(0.0, 1.0, 0.0))

# Transform to User-specified Cylindrical Coordinate System
dtm = session.scratchOdbs[filePath].rootAssembly.datumCsyses[cylName]
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    transformationType=USER_SPECIFIED, datumCsys=dtm)

# View COORD and COOR1, which is the Radius value
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='COORD', outputPosition=NODAL, refinement=(COMPONENT,
    'COOR1'))

# Add more decimal places just for the COORD 1
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(title=OFF,
    state=OFF, compass=OFF, legendDecimalPlaces=2, legendNumberFormat=FIXED,
    legendFont='-*-verdana-medium-r-normal-*-*-180-*-*-p-*-*-*')

# Print Output PNG 4 times for 0, 90, 180, 270 deg
for i in range(img_id,img_id + 4):
    # Print Output PNG
    session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    session.printToFile(fileName=int_review_path + img_labels[i], format=PNG, canvasObjects=(
        session.viewports['Viewport: 1'], ))

    # Rotate by 90deg
    session.viewports['Viewport: 1'].view.rotate(xAngle=0, yAngle=0, zAngle=90,
        mode=MODEL)

# Image ID becomes the last value of i
img_id = i+1

# =============================================================================
# Report with Nodal Position and Max Values
# =============================================================================

# View the Max. Principal Stress
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(
    INVARIANT, 'Max. Principal'), )
# Remove the compass, title block, and state block
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(title=OFF,
    state=OFF, compass=OFF, legendDecimalPlaces=0, legendNumberFormat=FIXED,
    legendFont='-*-verdana-medium-r-normal-*-*-180-*-*-p-*-*-*')

# Create max.rpt report
# odb = session.odbs[filePath]
session.fieldReportOptions.setValues(printXYData=OFF, printTotal=OFF)
session.writeFieldReport(fileName=int_review_path + 'report_max.rpt', append=ON, sortItem='Node Label',
    odb=odb, step=0, frame=10, outputPosition=NODAL, variable=(('S',
    INTEGRATION_POINT, ((INVARIANT, 'Max. Principal'), )), ),
    stepFrame=SPECIFY)
# Read the report
with open(int_review_path + 'report_max.rpt') as f:
    report_lines = f.readlines()
    f.close()

# There should ONLY be ONE "Maximum", will need to create a logic test for this
report_max = [s for s in report_lines if "Maximum" in s][0].split(" ")
report_max = [s for s in report_max if s.strip()]
# Maximum S.Max. Prin value @Loc 1
max_val_ID = float(report_max[1])
# Maximum S.Max. Prin value @Loc 2
max_val_OD = float(report_max[2].replace('\n',''))
# There will be more than one "At Node", one pertaining to the "Minimum" and
# the other for the "Maximum". Since the "Maximum" shows up second, then we
# will collect the second item from the list.
report_node = [s for s in report_lines if "At Node" in s][1].split(" ")
report_node = [s for s in report_node if s.strip()]
# Maximum S.Max. Prin node location @Loc 1
max_node_ID = int(report_node[2])
# Maximum S.Max. Prin node location @Loc 2
max_node_OD = int(report_node[3].replace('\n',''))

# =============================================================================
# C: Isolated Elements View
# =============================================================================

# Put in isometric view
# Brute Force Method
# Open the .inp file and find the matching node. Then check the
# theta component of the RTZ values.
with open(fileName + '.inp') as f:
    inp_lines = f.readlines()
    f.close()

# Need to find the node with the matching max_node_ID in order to collect the theta value
inp_max_node_ID = [s for s in inp_lines if str(max_node_ID) in s][0].split(",")
inp_max_node_ID = [float(s) for s in inp_max_node_ID]
inp_max_node_ID[0] = int(inp_max_node_ID[0])

inp_max_node_OD = [s.strip() for s in inp_lines if str(max_node_OD) in s][0].split(",")
inp_max_node_OD = [float(s) for s in inp_max_node_OD]
inp_max_node_OD[0] = int(inp_max_node_OD[0])

# All of these inputs need to be in interger format
# Open the node_info.txt file to collect theses values
with open(int_review_path + "node_info.txt") as f:
    info_lines = f.readlines()
    f.close()

info_val = [s.strip() for s in info_lines]
info_val = [s.split("=")[1] for s in info_val if "=" in s]
lim_cc      = int(info_val[0])
lim_ax      = int(info_val[1])
num_cal     = int(info_val[2])
num_nodes   = int(info_val[3])
def_angl    = int(info_val[4])
bar_stress  = float(info_val[5])

rot_angl_ID = -(inp_max_node_ID[2] - def_angl)
rot_angl_OD = -(inp_max_node_OD[2] - def_angl)

# Show max node value on screen
session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
    showMaxLocation=ON)
# Common Options -> Adjust Visible Edges to All
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=FREE)
# Move the Triad to the right bottom corner
session.viewports['Viewport: 1'].viewportAnnotationOptions.setValues(
    triadPosition=(95, 9))

# Print Output PNG for ID and then OD
max_node = [max_node_ID, max_node_OD]
rot_angl = [rot_angl_ID,rot_angl_OD]
for i in range(0,2):

    # Switch to view ID data (SNEG)
    session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
        sectionResults=USE_BOTTOM)

    # Return to the default model with everything deselected
    leaf = dgo.Leaf(leafType=DEFAULT_MODEL)
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)

    # Define the Lower and Upper Limits of the boundary
    ax_LB = int(max_node[i] - num_cal*lim_ax)
    ax_UB = int(max_node[i] + num_cal*lim_ax)

    # Need to "Replace" the first segment in order to make everything else disappear
    seg = str(ax_LB - lim_cc) + ':' + str(ax_LB + lim_cc) + ':1'

    leaf = dgo.LeafFromModelElemLabels(elementLabels=(('PART-1-1', (seg,
        )), ))
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)

    # Iterate through all other rings from lower bound to upper bound
    for seg_i in range(1, 2*lim_ax):
        seg = str(ax_LB + num_cal*seg_i - lim_cc) + ':' + str(ax_LB + num_cal*seg_i + lim_cc) + ':1'

        leaf = dgo.LeafFromModelElemLabels(elementLabels=(('PART-1-1', (seg, )), ))
        session.viewports['Viewport: 1'].odbDisplay.displayGroup.add(leaf=leaf)

    # Depending on the Input Deck angle, adjust the rotation
    # Need to convert it as negative since we want to adjust the negative degree difference

    session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
    session.viewports['Viewport: 1'].view.zoom(zoomFactor=0.90, mode=ABSOLUTE)
    session.viewports['Viewport: 1'].view.rotate(xAngle=0, yAngle=0, zAngle=rot_angl[i],
            mode=MODEL)

    # Print Output PNG
    session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    session.printToFile(fileName=int_review_path + img_labels[img_id], format=PNG, canvasObjects=(
        session.viewports['Viewport: 1'], ))

    # Switch to view OD data (SPOS) but on the same location
    session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
        sectionResults=USE_TOP)

    img_id += 1

    # Print Output PNG
    session.printOptions.setValues(vpBackground=ON, reduceColors=False)
    session.printToFile(fileName=int_review_path + img_labels[img_id], format=PNG, canvasObjects=(
        session.viewports['Viewport: 1'], ))

    img_id += 1


# Return to the default model with everything deselected
leaf = dgo.Leaf(leafType=DEFAULT_MODEL)
session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)

# # Switch to view OD data (SPOS)
# session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
#     sectionResults=USE_TOP)


# =============================================================================
# Part D: Print the Max. Principal Stress with the Dent on the Center
# =============================================================================

# Initial Position: Normal to the X plane with pos Z going left.
session.viewports['Viewport: 1'].view.setValues(session.views['Right'])
session.viewports['Viewport: 1'].view.zoom(zoomFactor=0.75, mode=ABSOLUTE)
session.viewports['Viewport: 1'].view.pan(xFraction=0.12, yFraction=0)

# Rotate to have Dent in Center
rot_angl_DENT = -(inp_max_node_OD[2] - 0)
session.viewports['Viewport: 1'].view.rotate(xAngle=0, yAngle=0, zAngle=rot_angl_DENT, mode=MODEL)

# Do not show max node value on screen
session.viewports['Viewport: 1'].odbDisplay.contourOptions.setValues(
    showMaxLocation=OFF)

# Common Options -> Adjust Visible Edges to All (09/21/22 made it FREE)
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=FREE)

# Print Output PNG
session.printOptions.setValues(vpBackground=ON, reduceColors=False)
session.printToFile(fileName=int_review_path + img_labels[img_id], format=PNG, canvasObjects=(
    session.viewports['Viewport: 1'], ))

# =============================================================================
# Part E: Create a Nodal Path down the maximum Max. Prin OD location
# =============================================================================

# Create a Path
# This gives me the remainder of the division, which will tell us how much to
# offset from Node 1. That way, I know what "caliper line" to use that aligns
# with the Max Node that we want to create a path from.
node_start = (max_node_OD % num_cal) + 1
node_path = str(node_start) + ':' + str(num_nodes) + ':' + str(num_cal)
session.Path(name='Path-1', type=NODE_LIST, expression=(('PART-1-1', (
    node_path, )), ))

# Create MPsData
pth = session.paths['Path-1']
session.XYDataFromPath(name='MPsData', path=pth, includeIntersections=False,
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10,
    projectionTolerance=0, shape=DEFORMED, labelType=TRUE_DISTANCE,
    removeDuplicateXYPairs=True, includeAllElements=False)

# Output MPsData
x0 = session.xyDataObjects['MPsData']
session.writeXYReport(fileName=int_review_path + 'report_MPs.rpt', xyData=(x0, ))

# Create RadiusData
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='COORD', outputPosition=NODAL, refinement=(COMPONENT,
    'COOR1'))
pth = session.paths['Path-1']
session.XYDataFromPath(name='RadiusData', path=pth, includeIntersections=False,
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10,
    projectionTolerance=0, shape=DEFORMED, labelType=TRUE_DISTANCE,
    removeDuplicateXYPairs=True, includeAllElements=False)

# Output RadiusData
x0 = session.xyDataObjects['RadiusData']
session.writeXYReport(fileName=int_review_path + 'report_Radius.rpt', xyData=(x0, ))
    
# Output All COORD, Strain, and MPs Data
# odb = session.odbs['C:/Users/ahassanin/ADV Integrity/ADV File Share - Projects/Projects/Analysis Group Management/Software Development/PDMAC/results/100978-015-FINAL/Abaqus Results/ID100978-015.odb']
odb = session.odbs[filePath]
session.writeFieldReport(fileName=int_review_path + 'report_All_Data.rpt', append=ON, 
    sortItem='Node Label', odb=odb, step=0, frame=10, outputPosition=NODAL, 
    variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), )), ('LE', 
    INTEGRATION_POINT, ((INVARIANT, 'Max. In-Plane Principal'), )), ('S', 
    INTEGRATION_POINT, ((INVARIANT, 'Max. In-Plane Principal'), )), ), 
    stepFrame=SPECIFY)

# # Read the output data
# with open('report_MPs.rpt') as f:
#     report_MPs = f.readlines()
#     f.close()
# # Delete the first 3 rows since they do not contain data
# report_MPs.pop(0)
# report_MPs.pop(0)
# report_MPs.pop(0)
# # Remove the leading and trailing spaces, also the '\n' character
# report_MPs = [s.strip() for s in report_MPs]
# # Split by spaces
# report_MPs = [s.split(" ") for s in report_MPs]
# # Remove the empty list items
# report_MPs = [s for s in report_MPs if s != ['']]

# # Remove the spaces and convert string numbers into floats
# x1 = np.zeros(len(report_MPs))
# MPs = np.zeros(len(report_MPs))
# for i in range(0,len(report_MPs)):
#     report_MPs[i] = [float(s) for s in report_MPs[i] if s.strip()]
#     x1[i] = report_MPs[i][0]
#     MPs[i] = report_MPs[i][1]

# # Read the output data
# with open('report_Radius.rpt') as f:
#     report_Radius = f.readlines()
#     f.close()
# # Delete the first 3 rows since they do not contain data
# report_Radius.pop(0)
# report_Radius.pop(0)
# report_Radius.pop(0)
# # Remove the leading and trailing spaces, also the '\n' character
# report_Radius = [s.strip() for s in report_Radius]
# # Split by spaces
# report_Radius = [s.split(" ") for s in report_Radius]
# # Remove the empty list items
# report_Radius = [s for s in report_Radius if s != ['']]

# # Remove the spaces and convert string numbers into floats
# x2 = np.zeros(len(report_Radius))
# R = np.zeros(len(report_Radius))
# for i in range(0,len(report_Radius)):
#     report_Radius[i] = [float(s) for s in report_Radius[i] if s.strip()]
#     x2[i] = report_Radius[i][0]
#     R[i] = report_Radius[i][1]

# =============================================================================
# Run the Previous SCF Calculation Method for Comparison Purposes
# =============================================================================

lastFrame = odb.steps['Pressure Up'].frames[-1]
maxPrincipal = 0
for step in odb.steps.values(-1):
    for stressValue in lastFrame.fieldOutputs['S'].values:
        if (stressValue.maxPrincipal > maxPrincipal):
            maxPrincipal = stressValue.maxPrincipal
    uSCF = maxPrincipal/bar_stress
odb.close()
uMPs = maxPrincipal

# =============================================================================
# Append the Dent Information File with the New Values
# =============================================================================

SCF_OD = max_val_OD / bar_stress
SCF_ID = max_val_ID / bar_stress
append_title = '============ RESULTS ============='
append_error = 'Script ran without errors.'
append_info = ['\n' + append_title + '\n',
               append_error + '\n',
               'inp_max_node_ID = ' + str(inp_max_node_ID) + '\n',
               'inp_max_node_OD = ' + str(inp_max_node_OD) + '\n',
               'max_val_ID      = ' + str(max_val_ID) + '\n',
               'max_val_OD      = ' + str(max_val_OD) + '\n',
               'OD SCF          = ' + str(SCF_OD) + '\n',
               'ID SCF          = ' + str(SCF_ID) + '\n',
               'Unavg MPs       = ' + str(uMPs) + '\n',
               'Unavg SCF       = ' + str(uSCF) + '\n']

with open(int_review_path + "node_info.txt", "a") as f:
    f.writelines(append_info)
    f.close()

# =============================================================================
# Create Output Graph
# =============================================================================

# # Create a graph for the Nodal Path
# # Adjust the formatting parameters of the graph
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.size'] = 8
# plt.rcParams['lines.markersize'] = 0.5

# fig1, ax1 = plt.subplots(figsize=(10,4), dpi=1200)
# ax2 = ax1.twinx()
# fig1.suptitle('Max. Principal Stress OD along Axial Radial Profile', fontsize=16)
# # First Y Axis
# ax1.plot(x1, MPs, c='tab:blue',label='Max. Principal Stress')
# ax1.set_xlabel('Position Z Along Pipe [in]')
# ax1.set_ylabel('Maximum Principal Stress [psi]')
# # Secondary Y Axis
# ax2.plot(x2, R, c='tab:orange',label='Axial Radial Profile')
# ax2.set_ylabel('Radius [in]')

# s1,sl1 = ax1.get_legend_handles_labels()
# s2,sl2 = ax2.get_legend_handles_labels()
# s = s1 + s2
# sl = sl1 + sl2
# ax1.legend(s,sl)

# fig1.savefig(int_review_path + '/' + img_labels[img_id])

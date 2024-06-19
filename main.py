import processing as prep
import pandas as pd

dent_path = 'ADV Project Folders/5972_RadiiData.csv'
ILI_format = 'southern'
WT = 0.25
SMYS = 52000

# Import Data
df = prep.ImportData(rd_path=dent_path, ILI_format=ILI_format, OD=24)
# Perform data smoothing
df.smooth_data()
# Create an input file
df.create_input_file(WT, SMYS)
# Submit to Abaqus
df.submit_input_file()
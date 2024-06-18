import preprocessing as prep
import pandas as pd

dent_path = 'ADV Project Folders/5972_RadiiData.csv'
ILI_format = 'southern'
df = prep.ImportData(rd_path=dent_path, ILI_format=ILI_format)
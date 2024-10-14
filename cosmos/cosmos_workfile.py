import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system("Py_CoSMoS.py")

data = pd.read_csv('urban_demand.csv')
analyzeTS(data, lags=10)
import collections
from copy import deepcopy
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import scipy.interpolate as intrp
import scipy.signal as sig
import scipy.stats as st
from skimage.filters import threshold_otsu as otsu

import Utility.ABF_util as abf

# if __name__ == "__main__":

# 	dataDir = "../EA_Code/Data/FlourakisData/2P/"

# 	fastDT = []

# 	for yearFolder in os.listdir(dataDir):
# 		yearPath = os.path.join(dataDir, yearFolder)
# 		if not os.path.isdir(yearPath):
# 			continue
# 		if int(yearFolder) > 2012:
# 			continue
# 		for monthFolder in os.listdir(yearPath):
# 			monthPath = os.path.join(yearPath, monthFolder)
# 			if not os.path.isdir(monthPath):
# 				continue
# 			for dayFolder in os.listdir(monthPath):
# 				dayPath = os.path.join(monthPath, dayFolder)
# 				if not os.path.isdir(dayPath):
# 					continue
# 				print(dayPath)
# 				for dayFile in os.listdir(dayPath):
# 					if dayFile[-4:] != ".abf":
# 						continue
# 					try:
# 						print(dayFile)
# 						[d, h] = abf.ABF_read(dayFile, datadir = dayPath)
# 						tb = abf.GetTimebase(h, 0)
# 						dt = np.mean(np.diff(tb.squeeze()))
# 						if dt < 0.1:
# 							print(f"Found dt = {dt}!")
# 							fastDT.append([dayPath, dayFile, dt])
# 					except:
# 						pass
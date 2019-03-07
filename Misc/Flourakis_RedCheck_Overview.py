"""
================================================================================
	Red Check Data Over view
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, March 6, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This script will contain plotting procedures to show the derived feature 
	values from the red-checked Flourakis data.

	Specifically, we will load and fit each red-checked data point and plot it
	over the course of the day.  It may be good to save these derived values
	to a csv for easy use later.

	The features/objectives to be measured include:
		- Firing Rate (Rest)
		- Membrane Voltage (Rest)
		- Firing Rate for Current Injections
		- F-I slope
		- Spike Amplitude
		- AHP
		- Input Resistance
		- Relaxation Time Constant for Slow Channel
	Others I have forgotten...

================================================================================
================================================================================
"""
from copy import deepcopy
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns

import Utility.ABF_util as abf
import Utility.DataIO_util as DIO
import Utility.runfile_util as rfu
import Utility.utility as utl

################################################################################
## Text Processing Functions for Loading CSV
################################################################################
def strip(text):
	try:
		return text.strip()
	except AttributeError:
		return text

def make_float(text):
	try:
		return float(strip(text))
	except TypeError:
		return strip(text)
	except ValueError:
		text = strip(text)
		if len(text) == 0:
			return np.NaN
		else:
			return text

def check_list(text):
	text = strip(text)
	if text.lower() != 'all':
		return make_list(text)
	else:
		return [0, -1]

def make_list(text):
	strip(text)
	text = text[1:-1].split(",")
	text = [int(t.strip()) for t in text]
	return text

def convert_time(text):
	hours, minutes = text.split(":")
	hours = int(hours)
	minutes = float(minutes)/60.
	return hours + minutes

def isWT(entry):
	if entry['Geno'] in ['WT', 'WT_OA', 'WT_PP', 'WT_DL', 'WTcer']:
		return True
	else:
		return False


################################################################################
## Runtime Commands
################################################################################
if __name__ == "__main__":

	plt.close('all')
	sns.set({'legend.frameon':True}, color_codes=True)
	matplotlib.rc("font", size=20)
	matplotlib.rc("xtick", labelsize=16)
	matplotlib.rc("ytick", labelsize=16)
	matplotlib.rc("axes", labelsize=20)
	matplotlib.rc("axes", titlesize=24)
	matplotlib.rc("legend", fontsize=18)
	matplotlib.rc("figure", titlesize=24)

	figDir = "./Figures/RedCheck_Overview/"

	gen_new_data = False
	check_for_data = True

	infoDir = "./Runfiles/RedCheck_Overview/"

	infoDict = rfu.getInfo(infoDir, verbose=1)

################################################################################
## Load csv
################################################################################
	csvfile = "../EA_Code/RedChecksData_Flourakis.csv"

	converters = {
		'ZT': convert_time,
		'Neuron':make_float,
		'Page No.':strip,
		'Geno':strip,
		'FR':make_float,
		'Vm':make_float,
		'RI':make_float, 
		'Check':make_float,
		'Flag':make_float,
		"Recordings":check_list}

	df = pd.read_csv(csvfile, sep=',', header=0,
		index_col=0, converters=converters)
	df['WT'] = df.apply(isWT, axis=1)

	WT = df.loc[df['WT']]

################################################################################
## Iterate through WT dates
################################################################################
	for dateNo, date in enumerate(WT.index.values):

		if not check_for_data:
			continue

		print(f"\nLoading data from {date} ({dateNo+1}/{len(WT.index.values)})")

		dateStr = date[-2:] + "/" + date[-5:-3] + "/" + date[:4]
		print(dateStr)

		print(WT.ZT[date])
		infoDict['data']['dates'] = {dateStr:None}

		dataDict = DIO.load_data(infoDict['data'], verbose=2)

		break

################################################################################
## Show plots!
################################################################################
	plt.show()
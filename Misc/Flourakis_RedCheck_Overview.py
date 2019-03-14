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

import Objectives.Electrophysiology.ephys_objs as epo
import Objectives.Electrophysiology.ephys_util as epu

import Utility.ABF_util as abf
import Utility.DataIO_util as DIO
import Utility.runfile_util as rfu
import Utility.utility as utl

################################################################################
##	Set Protocol ID Numbers
################################################################################
if True:
	EPHYS_PROT_REST				= 0
	EPHYS_PROT_DEPOLSTEP		= 1
	EPHYS_PROT_HYPERPOLSTEP		= 2
	EPHYS_PROT_DEPOLSTEPS		= 3
	EPHYS_PROT_HYPERPOLSTEPS	= 4
	EPHYS_PROT_CONSTHOLD		= 5

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
	if text.lower() == 'all':
		return [0, -1]
	elif text.lower() == 'none':
		return []
	else:
		return make_list(text)

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

	gen_new_data = True
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

		dateStr = date[-2:] + "/" + date[-5:-3] + "/" + date[:4]

		dateStr = "01/04/2011"
		# dateStr = "24/10/2012"
		print(f"\nLoading data from {dateStr} ({dateNo+1}/"+
			f"{len(WT.index.values)})")

		try:
			print(f"ZT = {WT.ZT[date]:.2f}\n")
		except:
			for ZT in WT.ZT[date]:
				print(f"ZT = {ZT:.2f}\n")
		infoDict['data']['dates'] = {dateStr:None}

		dataDict = DIO.load_data(infoDict['data'], verbose=0)

		dataDir = DIO.get_data_dir(dataDir="")

		dataFeatDir = DIO.find_date_folder(dateStr, dataDir)
		dataFeatPath = os.path.join(dataFeatDir, "dataFeat.pkl")

		if not gen_new_data:
			try:
				print(f"\nTrying to load {dataFeatPath}...")
				with open(dataFeatPath, "rb") as f:
					dataFeat = pkl.load(f)
				print(f"Loaded dataFeat.pkl")
				continue
			except:
				print(f"Could not load dataFeat.pkl...")
				pass

	############################################################################
	## Iterate through Objectives
	############################################################################
		dataFeat = {}

		keys = sorted(list(dataDict[dateStr].keys()))

		prots = []

		for key in keys:

			# if key != 17:
			# 	continue

			data = dataDict[dateStr][key]['data']
			hdr = dataDict[dateStr][key]['header']

			protocol = epu.getExpProtocol(hdr)

			prots.append([key, protocol])

			print(f"{key}: {protocol}")

		########################################################################
		## Rest Protocol
		########################################################################
			if protocol == EPHYS_PROT_REST:

				if hdr['lActualEpisodes'] > 1:
					print(f"WARNING: Many ({hdr['lActualEpisodes']}) episodes."+
						".. skipping!")
					continue

				dataFeat = epu.getRestFeatures(data, hdr, infoDict, dataFeat,
					key=key, verbose=2)

		########################################################################
		## Depolarization Step Protocol
		########################################################################
			elif protocol == EPHYS_PROT_DEPOLSTEP:

				# print(f"{key}: Depolarization Step Protocol")

				dataFeat = epu.getDepolFeatures(data, hdr, infoDict, dataFeat,
					key=key, verbose=2)

		########################################################################
		## Hyperpolarization Step Protocol
		########################################################################
			elif protocol == EPHYS_PROT_HYPERPOLSTEP:

				# print(f"{key}: Hyperpolarization Step Protocol")

				dataFeat = epu.getHyperpolFeatures(data, hdr, infoDict,
					dataFeat, key=key, verbose=2)

		########################################################################
		## Hyperpolarization Step Protocol
		########################################################################
			elif protocol == EPHYS_PROT_DEPOLSTEPS:

				# print(f"{key}: Depolarization Steps Protocol")
				dataFeat = epu.getDepolStepsFeatures(data, hdr, infoDict,
					dataFeat, key=key, verbose=2)

		########################################################################
		## Hyperpolarization Step Protocol
		########################################################################
			elif protocol == EPHYS_PROT_HYPERPOLSTEPS:

				# print(f"{key}: Hyperpolarization Steps Protocol")
				dataFeat = epu.getHyperpolStepsFeatures(data, hdr, infoDict,
					dataFeat, key=key, verbose=2)

		########################################################################
		## Hyperpolarization Step Protocol
		########################################################################
			elif protocol == EPHYS_PROT_CONSTHOLD:

				# print(f"{key}: Constant Holding Current Protocol")
				pass

		########################################################################
		## Hyperpolarization Step Protocol
		########################################################################
			else:
				# print(f"{key}: UNKNOWN PROTOCOL")
				pass

		break

		print("\n".join([", ".join([f"{p}" for p in pline]) for pline in prots]))

################################################################################
## Show plots!
################################################################################
	plt.show()
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

from Misc.dip import *

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
	if entry['Geno'] == 'WT':
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

	plotEveryone = True

	infoDir = "./Runfiles/RedCheck_Overview/"

	infoDict = rfu.getInfo(infoDir, verbose=1)

################################################################################
## Load csv
################################################################################
	csvfile = "./Misc/RedChecksData_Flourakis.csv"

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

		if dateNo != 6:
			continue

		# dateStr = "01/04/2011"
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

		try:
			print(f"\nTrying to load {dataFeatPath}...")
			with open(dataFeatPath, "rb") as f:
				dataFeat = pkl.load(f)
			print(f"Loaded dataFeat.pkl")
		except:
			print(f"Could not load dataFeat.pkl...")
			dataFeat = {}
			pass

	############################################################################
	## Iterate through Objectives
	############################################################################

		keys = sorted(list(dataDict[dateStr].keys()))

		goodRecs = WT['Recordings'].values[dateNo]
		if goodRecs[1] == -1:
			goodRecs[1] = keys[-1]
		goodRecs = np.arange(goodRecs[0], goodRecs[1]+1, 1).astype(int)

		prots = []

		for key in keys:

			if key not in goodRecs:
				continue

			print(f"\nKey = {key}\n")

			# if key != 22:
			# 	continue

			## IF I WANT TO GENERATE NEW DATA...
			## Delete all the entries that exist already
			if gen_new_data:
				for feat in dataFeat:
					try:
						_ = dataFeat[feat][key]
						del dataFeat[feat][key]
						continue
					except KeyError:
						pass
						
					for prot in dataFeat[feat]:
						try:
							_ = dataFeat[feat][prot][key]
							del dataFeat[feat][prot][key]
							continue
						except KeyError:
							pass

			## If I don't want to force make new data, check that all the 
			## features for all the protocols for all the keys are present.
			else:
				skipKey = False
				for feat in dataFeat:
					try:
						_ = dataFeat[feat][key]
						skipKey = True
					except KeyError:
						pass
						
					for prot in dataFeat[feat]:
						try:
							_ = dataFeat[feat][prot][key]
							skipKey = True
							break
						except KeyError:
							pass
				## If skipKey, go to next key.
				if skipKey:
					continue

			## Load the data, hdr, get the protocol.
			data = dataDict[dateStr][key]['data']
			hdr = dataDict[dateStr][key]['header']

			protocol = epu.getExpProtocol(hdr)

	############################################################################
	## Parse Protocols
	############################################################################

		########################################################################
		## Rest Protocol
		########################################################################
			if protocol == EPHYS_PROT_REST:

				if hdr['lActualEpisodes'] > 1:
					print(f"WARNING: Many ({hdr['lActualEpisodes']}) episodes."+
						".. skipping!")
					continue

				dataFeat = epu.getRestFeatures(data, hdr, infoDict, dataFeat,
					key=key, verbose=3)

		########################################################################
		## Depolarization Step Protocol
		########################################################################
			elif protocol == EPHYS_PROT_DEPOLSTEP:

				# print(f"{key}: Depolarization Step Protocol")

				dataFeat = epu.getDepolFeatures(data, hdr, infoDict, dataFeat,
					key=key, verbose=3)

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

				if len(data) > 30000:
					continue

				# print(f"{key}: Constant Holding Current Protocol")
				dataFeat = epu.getConstHoldFeatures(data, hdr, infoDict,
					dataFeat, key=key, verbose=2)

		########################################################################
		## Hyperpolarization Step Protocol
		########################################################################
			else:
				# print(f"{key}: UNKNOWN PROTOCOL")
				pass


	############################################################################
	## Plot MAYBE
	############################################################################

			if plotEveryone:

				if (protocol == EPHYS_PROT_REST) or (protocol == EPHYS_PROT_CONSTHOLD):
					fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
						sharex=True)

					data = data[:, 0].squeeze()


					dt = infoDict['data']['dt']
					grid = np.arange(len(data))*dt

					ax1.plot(grid, data)

					medArr =  epu.getRollPerc(data, window=100, perc=50)
					noMed = data - medArr

					ax2.plot(grid, noMed)

					if protocol == EPHYS_PROT_CONSTHOLD:
						waveform = abf.GetWaveform(hdr, hdr['lActualEpisodes'])
						holdCurr = np.unique(waveform)[0]*1000
						title = r"$I_{app} = " + f"{holdCurr:.2f}" + r"$mV"

						prot = 'hold'
					else:
						title = r"$I_{app} = 0$mV"

						prot = 'rest'

					ax1.set_title(f"ZT = {WT.ZT.iloc[dateNo]:.2f}: " + title)

					ax2.set_title("Median-Subtracted")

					ax2.set_xlabel("Time (s)")

					ax1.set_ylabel("Voltage (mV)")
					ax2.set_ylabel("Voltage (mV)")

					try:
						spikeDict = infoDict['objectives']['Spikes'].copy()
						spikeIdx, spikeVals = epo.getSpikeIdx(data, 
							dt=infoDict['data']['dt'], **spikeDict)


						if len(spikeIdx) > 0:
							ax1.scatter(spikeIdx*dt, spikeVals, c='r',
								zorder=20, label=(r'$FR = '+
									f"{1./dataFeat['ISI'][prot][key]:.4g}"+ r"$Hz"))

							spikeIdx = spikeIdx.astype(int)

							ax2.scatter(grid[spikeIdx], noMed[spikeIdx], c='r',
								zorder=20, label=(r'$FR = '+
									f"{1./dataFeat['ISI'][prot][key]:.4g}"+ r"$Hz"))

						if not np.isnan(dataFeat['Amp'][prot][key]):
							ax1.hlines(y=dataFeat['Amp'][prot][key], xmin=grid[0],
								xmax=grid[-1], color='orange',
								label=f"Amp = {dataFeat['Amp'][prot][key]:.4g}mV")
					except:
						pass

					if not np.isnan(dataFeat['PSD'][prot][key]):
						ax1.hlines(y=dataFeat['PSD'][prot][key], xmin=grid[0],
							xmax=grid[-1], color='purple',
							label=f"PSD = {dataFeat['PSD'][prot][key]:.4g}mV")

					ax1.legend(fontsize=8)
					ax2.legend(fontsize=8)

				else:
					fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8),
						sharex=True)

					ax1.set_title(f"ZT = {WT.ZT.iloc[dateNo]:.2f}")
					
					dpData, dpIdx, dpI = epu.getDepolIdx(data[:, 0].squeeze(), 
						hdr, protocol=EPHYS_PROT_DEPOLSTEP)

					dt = infoDict['data']['dt']
					grid = np.arange(len(dpData))*dt

					waveform = abf.GetWaveform(hdr, hdr['lActualEpisodes'])[:,0]
					waveform *= 1000

					ax1.plot(grid, dpData)

					medArr =  epu.getRollPerc(dpData, window=100, perc=50)
					noMed = dpData - medArr

					ax2.plot(grid, noMed)

					wavegrid = (np.arange(len(waveform)) - dpIdx[0])*dt

					ax3.plot(wavegrid, waveform, label=r'$\Delta I = '+
						f"{waveform.max()-waveform.min():.1f}" + r"$pA")

					if protocol == EPHYS_PROT_DEPOLSTEP:
						prot = 'depol'
					else:
						prot = ''

					ax1.set_title("Voltage Trace: ")

					ax2.set_title("Median-Subtracted")

					ax3.set_xlabel("Time (s)")

					ax1.set_ylabel("Voltage (mV)")
					ax2.set_ylabel("Voltage (mV)")
					ax3.set_ylabel("Current (pA)")

					try:
						spikeDict = infoDict['objectives']['Spikes'].copy()
						spikeIdx, spikeVals = epo.getSpikeIdx(dpData, 
							dt=infoDict['data']['dt'], **spikeDict)

						if len(spikeIdx) > 0:
							label = r'$FR_{beg} = '
							label += f"{1./dataFeat['ISI'][prot][key][0]:.4g}"
							label += r"$Hz" +f"\n" +r"$FR_{end} ="
							label += f"{1./dataFeat['ISI'][prot][key][1]:.4g}"
							label += r"$Hz"
							ax1.scatter(spikeIdx*dt, spikeVals, c='r',
								zorder=20, label=label)

							spikeDict['exact'] = False
							spikeIdx, spikeVals = epo.getSpikeIdx(dpData, 
								dt=infoDict['data']['dt'], **spikeDict)
							spikeIdx = spikeIdx.astype(int)

							ax2.scatter(grid[spikeIdx], noMed[spikeIdx], c='r',
								zorder=20, label=label)

						if not np.isnan(dataFeat['Amp'][prot][key]):
							ax1.hlines(y=dataFeat['Amp'][prot][key], xmin=grid[0],
								xmax=grid[-1], color='orange',
								label=f"Amp = {dataFeat['Amp'][prot][key]:.4g}mV")
					except:
						pass

					if not np.isnan(dataFeat['PSD'][prot][key]):
						ax1.hlines(y=dataFeat['PSD'][prot][key], xmin=grid[0],
							xmax=grid[-1], color='purple',
							label=f"PSD = {dataFeat['PSD'][prot][key]:.4g}mV")

					ax1.legend(fontsize=10)
					ax2.legend(fontsize=10)
					ax3.legend(fontsize=10)



				fig.tight_layout()

				figName = f"Figures/DataVisual_{date}_{key:04d}.pdf"
				figPath = os.path.join(dataFeatDir, figName)

				fig.savefig(figPath, format='pdf')

		break

	# print("\n".join([", ".join([f"{p}" for p in pline]) for pline in prots]))

	with open(dataFeatPath, "wb") as f:
		pkl.dump(dataFeat, f)


################################################################################
## Show plots!
################################################################################
	plt.show()
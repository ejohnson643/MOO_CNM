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

		if dateNo != 36:
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

		dataFeat = {}

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

			# if key > 24:
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
						except (KeyError, IndexError):
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

			print(f"Rec Time = {hdr['lFileStartDate']}, "+
				f"{hdr['lFileStartTime']/60./60.:.2f}\n")

			if key == 22:
				data22 = data.copy()
				hdr22 = hdr.copy()
			elif key == 23:
				data23 = data.copy()
				hdr23 = hdr.copy()
			elif key == 24:
				data24 = data.copy()
				hdr24 = hdr.copy()

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

				elif protocol == EPHYS_PROT_DEPOLSTEP:
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
						spikeDict['exact'] = True
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


				try:
					fig.tight_layout()

					figName = f"Figures/DataVisual_{date}_{key:04d}.pdf"
					figPath = os.path.join(dataFeatDir, figName)

					fig.savefig(figPath, format='pdf')
				except:
					pass

		break

	# print("\n".join([", ".join([f"{p}" for p in pline]) for pline in prots]))

	with open(dataFeatPath, "wb") as f:
		pkl.dump(dataFeat, f)


# ################################################################################
# ## Show plots!
# ################################################################################

# 	dt = 0.001
# 	maxRate = 100.
# 	minISI = int(1./dt/maxRate)

# 	data22 = data22[:, 0].squeeze()

# 	window22 = int(len(data22)/100)
# 	window22 = window22 + 1 if (window22 % 2 == 0) else window22

# 	med22 = epu.getRollPerc(data22, window=int(window22), perc=50)
# 	noMed22 = data22-med22

# 	tGrid22 = np.arange(len(data22))*dt

# 	data23 = data23[:, 0].squeeze()
# 	old23 = data23.copy()
# 	data23, dpIdx, dpI = epu.getDepolIdx(data23, hdr,
# 		protocol=EPHYS_PROT_DEPOLSTEP)

# 	window23 = int(len(data23)/100)
# 	window23 = window23 + 1 if (window23 % 2 == 0) else window23

# 	med23 = epu.getRollPerc(data23, window=int(window23), perc=50)
# 	noMed23 = data23-med23

# 	tGrid23 = np.arange(len(data23))*dt

# 	data24 = data24[:, 0].squeeze()
# 	old24 = data24.copy()
# 	data24, dpIdx, dpI = epu.getDepolIdx(data24, hdr,
# 		protocol=EPHYS_PROT_DEPOLSTEP)

# 	window24 = int(len(data24)/100)
# 	window24 = window24 + 1 if (window24 % 2 == 0) else window24

# 	med24 = epu.getRollPerc(data24, window=int(window24), perc=50)
# 	noMed24 = data24-med24

# 	tGrid24 = np.arange(len(data24))*dt


# 	wLenMax22, wLenMax23, wLenMax24 = len(noMed22), len(noMed23), len(noMed24)
# 	t22, t23 = np.percentile(noMed22, 90), np.percentile(noMed23, 90)
# 	t24 = np.percentile(noMed24, 90)
# 	prom22 = np.diff(np.percentile(noMed22, [1, 90]))[0]
# 	prom23 = np.diff(np.percentile(noMed23, [1, 90]))[0]
# 	prom24 = np.diff(np.percentile(noMed24, [1, 90]))[0]

# ################################################################################
# ## Show Median of 22
# ################################################################################

# 	fig1, [ax1, ax2] = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 	ax1.plot(tGrid22, data22, label='Data')
# 	ax1.plot(tGrid22, med22, label='Median Filter')

# 	ax2.plot(tGrid22, noMed22, label='Median-Subtracted Data')

# 	ax1.set_ylabel("Voltage (mV)")
# 	ax1.legend()
# 	ax1.set_title(f"{date}: Rec 22")

# 	ax2.set_xlabel("Time (s)")
# 	ax2.set_ylabel("Voltage (mV)")
# 	ax2.legend()

# 	fig1.tight_layout()

# ################################################################################
# ## Show Median of 23
# ################################################################################

# 	fig2, [ax2_1, ax2_2] = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# 	ax2_1.plot(tGrid23, data23, label='Data')
# 	ax2_1.plot(tGrid23, med23, label='Median Filter')

# 	ax2_2.plot(tGrid23, noMed23, label='Median-Subtracted Data')

# 	ax2_2.hlines(y=np.percentile(noMed23, 90), xmin=0, xmax=10)

# 	ax2_1.set_ylabel("Voltage (mV)")
# 	ax2_1.legend()
# 	ax2_1.set_title(f"{date}: Rec 23")

# 	ax2_2.set_xlabel("Time (s)")
# 	ax2_2.set_ylabel("Voltage (mV)")
# 	ax2_2.legend()

# 	fig2.tight_layout()

# ################################################################################
# ## Show Median Edge Corrections
# ################################################################################
	
# 	fig3, [[ax3_1, ax3_2], [ax3_3, ax3_4]] = plt.subplots(2, 2, figsize=(12, 8))

# 	rawMed22 = sig.order_filter(data22, np.ones(window22), int((window22-1)/2))

# 	cmap = sns.color_palette()

# 	xlimMin = -.1
# 	xlimMax = 21.
# 	xlimPad = .3


# 	ax3_1.plot(tGrid22, data22, label='Data')
# 	ax3_1.plot(tGrid22, rawMed22, label='No Edge Correction', color=cmap[1])
# 	ax3_1.plot(tGrid22, med22, label='Edge-Corrected Median', color=cmap[2])

# 	ax3_1.set_xlim(xlimMin, xlimPad)
# 	ax3_1.legend(fontsize=10)
# 	ax3_1.set_ylabel('Voltage (mV)')

# 	ax3_3.plot(tGrid22, data22-rawMed22, label='No Edge Correction',
# 		color=cmap[1])
# 	ax3_3.plot(tGrid22, data22-med22, label='Edge-Corrected Median',
# 		color=cmap[2])

# 	ax3_3.set_xlim(xlimMin, xlimPad)
# 	ax3_3.legend(fontsize=10)
# 	ax3_3.set_ylabel('Voltage (mV)')

# 	ax3_2.plot(tGrid22, data22, label='Data')
# 	ax3_2.plot(tGrid22, rawMed22, label='No Edge Correction', color=cmap[1])
# 	ax3_2.plot(tGrid22, med22, label='Edge-Corrected Median', color=cmap[2])

# 	ax3_2.set_xlim(xlimMax-xlimPad, xlimMax-xlimMin)
# 	ax3_2.legend(fontsize=10)
# 	ax3_2.set_ylabel('Voltage (mV)')

# 	ax3_4.plot(tGrid22, data22-rawMed22, label='No Edge Correction',
# 		color=cmap[1])
# 	ax3_4.plot(tGrid22, data22-med22, label='Edge-Corrected Median',
# 		color=cmap[2])

# 	ax3_4.set_xlim(xlimMax-xlimPad, xlimMax-xlimMin)
# 	ax3_4.legend(fontsize=10)
# 	ax3_4.set_ylabel('Voltage (mV)')

# 	fig3.suptitle(f"{date}: Rec 22")

# 	fig3.tight_layout()



# 	fig4, [[ax4_1, ax4_2], [ax4_3, ax4_4]] = plt.subplots(2, 2, figsize=(12, 8))

# 	rawMed23 = sig.order_filter(data23, np.ones(window23), int((window23-1)/2))

# 	cmap = sns.color_palette()

# 	xlimMin = -.1
# 	xlimMax = 10.
# 	xlimPad = .3


# 	ax4_1.plot(tGrid23, data23, label='Data')
# 	ax4_1.plot(tGrid23, rawMed23, label='No Edge Correction', color=cmap[1])
# 	ax4_1.plot(tGrid23, med23, label='Edge-Corrected Median', color=cmap[2])

# 	ax4_1.set_xlim(xlimMin, xlimPad)
# 	ax4_1.legend(fontsize=10)
# 	ax4_1.set_ylabel('Voltage (mV)')

# 	ax4_3.plot(tGrid23, data23-rawMed23, label='No Edge Correction',
# 		color=cmap[1])
# 	ax4_3.plot(tGrid23, data23-med23, label='Edge-Corrected Median',
# 		color=cmap[2])

# 	ax4_3.set_xlim(xlimMin, xlimPad)
# 	ax4_3.legend(fontsize=10)
# 	ax4_3.set_ylabel('Voltage (mV)')

# 	ax4_2.plot(tGrid23, data23, label='Data')
# 	ax4_2.plot(tGrid23, rawMed23, label='No Edge Correction', color=cmap[1])
# 	ax4_2.plot(tGrid23, med23, label='Edge-Corrected Median', color=cmap[2])

# 	ax4_2.set_xlim(xlimMax-xlimPad, xlimMax-xlimMin)
# 	ax4_2.legend(fontsize=10)
# 	ax4_2.set_ylabel('Voltage (mV)')

# 	ax4_4.plot(tGrid23, data23-rawMed23, label='No Edge Correction',
# 		color=cmap[1])
# 	ax4_4.plot(tGrid23, data23-med23, label='Edge-Corrected Median',
# 		color=cmap[2])

# 	ax4_4.set_xlim(xlimMax-xlimPad, xlimMax-xlimMin)
# 	ax4_4.legend(fontsize=10)
# 	ax4_4.set_ylabel('Voltage (mV)')

# 	fig4.suptitle(f"{date}: Rec 23")

# 	fig4.tight_layout()

# ################################################################################
# ## Show Threshold, Prominences, WLen Defs
# ################################################################################
	
# 	from scipy.misc import electrocardiogram as ecg

# 	x = ecg()[17000:18000]

# 	fig5, [ax5_1, ax5_2, ax5_3] = plt.subplots(3, 1, figsize=(12, 8),
# 		sharex=True)


# 	ax5_1.plot(x, color=cmap[0])
# 	ax5_2.plot(x, color=cmap[0])
# 	ax5_3.plot(x, color=cmap[0])

# 	## RESTRICTIVE PROMINENCE THRESHOLD
# 	peaks, props = sig.find_peaks(x, prominence=1.5, width=0, wlen=2*len(x),
# 		rel_height=1)

# 	ax5_1.plot(peaks, x[peaks], "x", color=cmap[1])
# 	ax5_1.vlines(x=peaks, ymin=x[peaks]-props['prominences'], ymax=x[peaks],
# 		color=cmap[1])
# 	ax5_1.hlines(y=props['width_heights'], xmin=props['left_ips'],
# 		xmax=props['right_ips'], color=cmap[1])

# 	## LOOSE PROM THRESHOLD
# 	peaks, props = sig.find_peaks(x, prominence=0.5, width=0, wlen=2*len(x),
# 		rel_height=1)

# 	ax5_2.plot(peaks, x[peaks], "x", color=cmap[1])
# 	ax5_2.vlines(x=peaks, ymin=x[peaks]-props['prominences'], ymax=x[peaks],
# 		color=cmap[1])
# 	ax5_2.hlines(y=props['width_heights'], xmin=props['left_ips'],
# 		xmax=props['right_ips'], color=cmap[1])

# 	## RESTRICTIVE WLEN THRESHOLD
# 	peaks, props = sig.find_peaks(x, prominence=0.5, width=0, wlen=50,
# 		rel_height=1)

# 	ax5_3.plot(peaks, x[peaks], "x", color=cmap[1])
# 	ax5_3.vlines(x=peaks, ymin=x[peaks]-props['prominences'], ymax=x[peaks],
# 		color=cmap[1])
# 	ax5_3.hlines(y=props['width_heights'], xmin=props['left_ips'],
# 		xmax=props['right_ips'], color=cmap[1])

# 	ax5_3.set_xlabel("Time")
# 	ax5_3.set_ylabel("Voltage")
# 	ax5_3.set_title(f"Prom = 0.5, wlen = 50")

# 	ax5_2.set_ylabel("Voltage")
# 	ax5_2.set_title(f"Prom = 0.5, wlen = {2*len(x)}")

# 	ax5_1.set_ylabel("Voltage")
# 	ax5_1.set_title(f"Prom = 1.5, wlen = {2*len(x)}")


# 	fig5.tight_layout()



# 	fig6, ax6 = plt.subplots(1, 1, figsize=(12, 6))

# 	ax6.plot(x, color=cmap[0])

# 	peaks, props = sig.find_peaks(x)

# 	ax6.plot(peaks, x[peaks], "X", color=cmap[1])

# 	ax6.set_xlabel("Time")
# 	ax6.set_ylabel("Voltage")
# 	fig6.tight_layout()


# 	fig7, ax7 = plt.subplots(1, 1, figsize=(12, 6))

# 	ax7.plot(x, color=cmap[0])

# 	peaks, props = sig.find_peaks(x)

# 	ax7.plot(peaks, x[peaks], "X", color=cmap[8], alpha=0.5)
# 	xlim = ax7.get_xlim()

# 	prom, wlen1 = 1.5, 2*len(x)
# 	peaks1, props1 = sig.find_peaks(x, prominence=0.5, width=0, wlen=wlen1,
# 		rel_height=1)
# 	wlen2, wlen3 = 100, 50
# 	peaks2, props2 = sig.find_peaks(x, prominence=0.5, width=0, wlen=wlen2,
# 		rel_height=1)
# 	peaks3, props3 = sig.find_peaks(x, prominence=0.5, width=0, wlen=wlen3,
# 		rel_height=1)

# 	pLoc = 691
# 	pId1 = np.where(peaks1 == pLoc)[0][0]
# 	pId2 = np.where(peaks2 == pLoc)[0][0]
# 	pId3 = np.where(peaks3 == pLoc)[0][0]

# 	ax7.vlines(x=pLoc, ymin=x[pLoc]-props1['prominences'][pId1], ymax=x[pLoc],
# 		color=cmap[1], label=f"Prom = {props1['prominences'][pId1]:.2f}")
# 	ax7.hlines(y=x[pLoc]-props1['prominences'][pId1], xmin=pLoc-wlen1/2,
# 		xmax=pLoc+wlen1/2, color=cmap[1])
# 	ax7.plot([pLoc-wlen1/2, pLoc, pLoc+wlen1/2],
# 		[x[pLoc]-props1['prominences'][pId1], x[pLoc],
# 			x[pLoc]-props1['prominences'][pId1]], color=cmap[1],
# 			label=f"Min Secant = {props1['prominences'][pId1]/wlen1*2:.2g}")

# 	ax7.vlines(x=pLoc, ymin=x[pLoc]-props2['prominences'][pId2], ymax=x[pLoc],
# 		color=cmap[2], label=f"Prom = {props2['prominences'][pId2]:.2f}")
# 	ax7.hlines(y=x[pLoc]-props2['prominences'][pId2], xmin=pLoc-wlen2/2,
# 		xmax=pLoc+wlen2/2, color=cmap[2])
# 	ax7.plot([pLoc-wlen2/2, pLoc, pLoc+wlen2/2],
# 		[x[pLoc]-props2['prominences'][pId2], x[pLoc],
# 			x[pLoc]-props2['prominences'][pId2]], color=cmap[2],
# 			label=f"Min Secant = {props1['prominences'][pId2]/wlen2*2:.2g}")

# 	ax7.vlines(x=pLoc, ymin=x[pLoc]-props3['prominences'][pId3], ymax=x[pLoc],
# 		color=cmap[3], label=f"Prom = {props3['prominences'][pId3]:.2f}")
# 	ax7.hlines(y=x[pLoc]-props3['prominences'][pId3], xmin=pLoc-wlen3/2,
# 		xmax=pLoc+wlen3/2, color=cmap[3])
# 	ax7.plot([pLoc-wlen3/2, pLoc, pLoc+wlen3/2],
# 		[x[pLoc]-props3['prominences'][pId3], x[pLoc],
# 			x[pLoc]-props3['prominences'][pId3]], color=cmap[3],
# 			label=f"Min Secant = {props1['prominences'][pId3]/wlen3*2:.2g}")

# 	ax7.set_xlim(xlim)

# 	ax7.legend(fontsize=10)

# 	ax7.set_xlabel("Time")
# 	ax7.set_ylabel("Voltage")
# 	fig7.tight_layout()

# ################################################################################
# ## Show Percentiles
# ################################################################################

# 	fig8, [ax8_1, ax8_2] = plt.subplots(2, 1, figsize=(12, 8))

# 	sns.distplot(noMed22, ax=ax8_1, kde=False)

# 	ax8_1.set_yscale('log')

# 	ylim = ax8_1.get_ylim()

# 	t1, t90, t99 = np.percentile(noMed22, [1, 90, 99])
# 	ax8_1.vlines(x=t1, ymin=ylim[0], ymax=ylim[1], color=cmap[1],
# 		label=f"1st Percentile")
# 	ax8_1.vlines(x=t90, ymin=ylim[0], ymax=ylim[1], color=cmap[2],
# 		label=f"90th Percentile")
# 	ax8_1.vlines(x=t99, ymin=ylim[0], ymax=ylim[1], color=cmap[3],
# 		label=f"99th Percentile")
# 	ax8_1.hlines(y=100, xmin=t1, xmax=t90, color='k',
# 		label=f"Min Prom = {t90-t1:.2g}mV")
# 	ax8_1.set_ylim(ylim)

# 	ax8_1.set_xlabel("Voltage (mV)")
# 	ax8_1.set_ylabel("No. Measurements")
# 	ax8_1.set_title(f"{date}: Recording 22")
# 	ax8_1.legend(fontsize=10)

# 	sns.distplot(noMed23, ax=ax8_2, kde=False)

# 	ax8_2.set_yscale('log')

# 	ylim = ax8_1.get_ylim()

# 	t1, t90, t99 = np.percentile(noMed23, [1, 90, 99])
# 	ax8_2.vlines(x=t1, ymin=ylim[0], ymax=ylim[1], color=cmap[1],
# 		label=f"1st Percentile")
# 	ax8_2.vlines(x=t90, ymin=ylim[0], ymax=ylim[1], color=cmap[2],
# 		label=f"90th Percentile")
# 	ax8_2.vlines(x=t99, ymin=ylim[0], ymax=ylim[1], color=cmap[3],
# 		label=f"99th Percentile")
# 	ax8_2.hlines(y=100, xmin=t1, xmax=t90, color='k',
# 		label=f"Min Prom = {t90-t1:.2g}mV")
# 	ax8_2.set_ylim(ylim)

# 	ax8_2.set_xlabel("Voltage (mV)")
# 	ax8_2.set_ylabel("No. Measurements")
# 	ax8_2.set_title(f"{date}: Recording 23")
# 	ax8_2.legend(fontsize=10)

# 	fig8.tight_layout()

# ################################################################################
# ## Show Peaks Found with Max wLen
# ################################################################################

# 	peakIdx22, _ = sig.find_peaks(noMed22, height=t22, distance=minISI,
# 		prominence=prom22, wlen=wLenMax22)
# 	peakVals22 = noMed22[peakIdx22]

# 	peakIdx23, _ = sig.find_peaks(noMed23, height=t23, distance=minISI,
# 		prominence=prom23, wlen=wLenMax23)
# 	peakVals23 = noMed23[peakIdx23]

# 	peakIdx24, _ = sig.find_peaks(noMed24, height=t24, distance=minISI,
# 		prominence=prom24, wlen=wLenMax24)
# 	peakVals24 = noMed24[peakIdx24]

# 	fig9, [ax9_1, ax9_2] = plt.subplots(2, 1, figsize=(12, 8))

# 	ax9_1.plot(tGrid22, noMed22)
# 	ax9_1.plot(tGrid22[peakIdx22], peakVals22, "X", color=cmap[1])
# 	ax9_1.hlines(y=t22, xmin=0, xmax=tGrid22[-1], color='k',
# 		label=f"Threshold = {t22:.3g}", zorder=10)

# 	ax9_1.set_xlabel('Time (s)')
# 	ax9_1.set_ylabel("Voltage (mV)")
# 	ax9_1.set_title(f"{date}: Recording 22: Med-Sub, wlen={wLenMax22}")
# 	ax9_1.legend(fontsize=10)

# 	ax9_2.plot(tGrid23, noMed23)
# 	ax9_2.plot(tGrid23[peakIdx23], peakVals23, "X", color=cmap[1])
# 	ax9_2.hlines(y=t23, xmin=0, xmax=tGrid23[-1], color='k',
# 		label=f"Threshold = {t23:.3g}", zorder=10)

# 	ax9_2.set_xlabel('Time (s)')
# 	ax9_2.set_ylabel("Voltage (mV)")
# 	ax9_2.set_title(f"{date}: Recording 23: Med-Sub, wlen={wLenMax23}")
# 	ax9_2.legend(fontsize=10)

# 	fig9.tight_layout()

# ################################################################################
# ## Show Peaks Found with Max wLen and Iterative Thresholding
# ################################################################################

# 	fig10, [[ax10_1, ax10_1b],
# 		[ax10_2, ax10_2b]] = plt.subplots(2, 2, figsize=(12, 8),
# 								gridspec_kw={'width_ratios':[2,1]})

# 	ax10_1.plot(tGrid22, noMed22)
# 	ax10_1.plot(tGrid22[peakIdx22], peakVals22, "X", color=cmap[1])

# 	ax10_2.plot(tGrid24, noMed24)
# 	ax10_2.plot(tGrid24[peakIdx24], peakVals24, "X", color=cmap[1])

# 	bins22 = np.linspace(peakVals22.min(), peakVals22.max(), 10)
# 	bins24 = np.linspace(peakVals24.min(), peakVals24.max(), 10)

# 	nThresh = 5

# 	cmapHLS = sns.color_palette('hls', nThresh)

# 	itr, pThresh = 0, 75
# 	while itr < nThresh:

# 		thresh22 = np.percentile(noMed22, pThresh)
# 		ax10_1.hlines(y=thresh22, xmin=0, xmax=tGrid22[-1], color=cmapHLS[itr],
# 			zorder=10)

# 		goodVals22 = peakVals22[peakVals22 > thresh22]
# 		sns.distplot(goodVals22, ax=ax10_1b, color=cmapHLS[itr], kde=False,
# 			norm_hist=False, bins=bins22, label=r"$T_{"+f"{int(pThresh)}"+
# 			r"} = $"+f"{thresh22:.2f}mV")

# 		thresh24 = np.percentile(noMed24, pThresh)
# 		ax10_2.hlines(y=thresh24, xmin=0, xmax=tGrid24[-1], color=cmapHLS[itr],
# 			zorder=10)

# 		goodVals24 = peakVals24[peakVals24 > thresh24]
# 		sns.distplot(goodVals24, ax=ax10_2b, color=cmapHLS[itr], kde=False,
# 			norm_hist=False, bins=bins24, label=r"$T_{"+f"{int(pThresh)}"+
# 			r"} = $"+f"{thresh24:.2f}mV")

# 		pThresh += (100-pThresh)/2.
# 		itr += 1

# 	ax10_1.set_xlabel('Time (s)')
# 	ax10_1.set_ylabel("Voltage (mV)")
# 	ax10_1.set_title(f"{date}: Recording 22", fontsize=20)
# 	ax10_1b.set_title(f"Med-Sub, wlen={wLenMax22}", fontsize=20)
# 	ax10_1b.legend(fontsize=10)

# 	ax10_2.set_xlabel('Time (s)')
# 	ax10_2.set_ylabel("Voltage (mV)")
# 	ax10_2.set_title(f"{date}: Recording 24", fontsize=20)
# 	ax10_2b.set_title(f"Med-Sub, wlen={wLenMax24}", fontsize=20)
# 	ax10_2b.legend(fontsize=10)

# 	fig10.tight_layout()

# ################################################################################
# ## Show N Peaks As Function of Threshold
# ################################################################################	
	
# 	plt.close('all')

# 	pThreshArr = np.arange(50, 100)
# 	threshArr = np.percentile(noMed22, pThreshArr)
# 	nPeaks22 = np.zeros_like(pThreshArr)
# 	peakValsArr22 = []
# 	for ii, pThresh in enumerate(pThreshArr):

# 		thresh22 = np.percentile(noMed22, pThresh)

# 		peakIdx, _ = sig.find_peaks(noMed22, height=thresh22, distance=minISI,
# 			prominence=prom22, wlen=wLenMax22)

# 		nPeaks22[ii] = len(peakIdx)

# 		peakValsArr22.append(noMed22[peakIdx])

# 	fig11, [ax11_1, ax11_2] = plt.subplots(2, 1, figsize=(12, 8))

# 	ax11_1.scatter(threshArr, nPeaks22)


################################################################################
## Show plots!
################################################################################

	plt.show()
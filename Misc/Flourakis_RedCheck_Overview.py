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

		if dateNo != 4:
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

			if key != 10:
				continue

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
					if skipKey:
						break
				if skipKey:
					continue

			data = dataDict[dateStr][key]['data']
			hdr = dataDict[dateStr][key]['header']

			protocol = epu.getExpProtocol(hdr)

	############################################################################
	## TRYING TO FIGURE OUT SPIKES
	############################################################################

			if protocol == EPHYS_PROT_DEPOLSTEP:
				dpData, dpIdx, dpI = epu.getDepolIdx(data[:, 0].squeeze(), hdr,
					protocol=EPHYS_PROT_DEPOLSTEP)
			else:
				dpData = deepcopy(data[:, 0].squeeze())

			grid = np.arange(len(dpData))



			window, perc = 201, 50
			order = int(window*perc/100.)
			medArr = sig.order_filter(dpData, np.ones(window), order)

			leftEnd = np.array([sig.order_filter(dpData[:window*2],
				np.ones(int(wds)), int((wds-1)/2))[ii]
				for (ii, wds) in zip(np.arange(window),
					np.arange(window) + np.arange(window)%2. + 1)])

			rightEnd = np.array([sig.order_filter(dpData[-window*2-1:],
				np.ones(int(wds)), int((wds-1)/2))[ii]
				for (ii, wds) in zip(-1*np.arange(window)[::-1]-1,
					(np.arange(window) + np.arange(window)%2. + 1)[::-1])])

			medArr[:window] = leftEnd
			medArr[-window:] = rightEnd

			noMed = dpData - medArr

			dt, maxRate = 0.001, 50.
			minISI = int(1./dt/maxRate)

			pThresh = 90
			thresh = np.percentile(noMed, pThresh)

			pmin, pProm = 1, 90
			prom = np.diff(np.percentile(noMed, [pmin, pProm]))[0]

			print("\nStarting with...")
			print(f'T{pThresh} =\t{thresh:.4g} mV\nP{pProm} =\t{prom:.4g} mV')

			wLen_Max = len(noMed)

			threshLevel = 90
			while True:

				peakIdx, _ = sig.find_peaks(noMed, height=thresh,
					distance=minISI, prominence=prom, wlen=wLen_Max)

				peakVals = noMed[peakIdx]

				if len(peakVals) < 2:
					break


				oThr = otsu(peakVals)

				lowMed = np.median(peakVals[peakVals <= oThr])
				highMed = np.median(peakVals[peakVals > oThr])

				print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals <= oThr)} on left"+
					f" {sum(peakVals > oThr)} on right)")

				print(f"Low Median {lowMed:.4g}, High Median {highMed:.4g}")

				if (highMed - lowMed <= 5):
					break

				threshLevel += (100 - threshLevel)/2.
				thresh = np.percentile(noMed, threshLevel)
				print(f"The threshold level is {threshLevel} ({thresh:.4g})")

			NSp_Max = len(peakIdx)
			print(f"\n{NSp_Max} Spikes found!")


			minSlope = 20.
			wLen_MinSlope = int(np.ceil(prom/minSlope/dt))
			if wLen_MinSlope % 2 == 0:
				wLen_MinSlope += 1

			print(f"\nwLen for Min Slope = {wLen_MinSlope:.4g}")

			threshLevel = 90
			thresh = np.percentile(noMed, pThresh)
			while True:

				peakIdx, foo = sig.find_peaks(noMed, height=thresh, 
					distance=minISI, prominence=prom, wlen=wLen_MinSlope)

				peakVals = noMed[peakIdx]

				if len(peakVals) < 2:
					break


				oThr = otsu(peakVals)

				lowMed = np.median(peakVals[peakVals <= oThr])
				highMed = np.median(peakVals[peakVals > oThr])

				print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals <= oThr)} on left"+
					f" {sum(peakVals > oThr)} on right)")

				print(f"Low Median {lowMed:.4g}, High Median {highMed:.4g}")

				if (highMed - lowMed <= 5):
					break

				threshLevel += (100 - threshLevel)/2.
				thresh = np.percentile(noMed, threshLevel)
				print(f"The threshold level is {threshLevel} ({thresh:.4g})")


			NSp_MinSlope = len(peakIdx)
			print(f"\n{NSp_MinSlope} Spikes found!")

			if NSp_Max == NSp_MinSlope:

				print("\nwLen_MinSlope yields same no. of spikes as wLen_Max")

				wLen = wLen_MinSlope

				peakWids = sig.peak_widths(noMed, peakIdx, wlen=wLen_MinSlope,
					rel_height=1.)

				leftWids, rightWids = peakIdx-peakWids[2], peakWids[3]-peakIdx

				NSpLeft = np.sum(leftWids <= (wLen-1)/2.)
				NSpRight = np.sum(rightWids <= (wLen-1)/2.)

				while np.minimum(NSpLeft, NSpRight) == NSp_Max:

					wLen -= 2

					NSpLeft = np.sum(leftWids <= (wLen-1)/2.)
					NSpRight = np.sum(rightWids <= (wLen-1)/2.)

				wLen += 2

				peakIdx, foo = sig.find_peaks(noMed, height=thresh,
					distance=minISI, prominence=prom, wlen=wLen)

				print(f"\nDetermined that the minimal width is {wLen}!")


			else:

				print("\nwLen_MinSlope yields diff no. of spikes as wLen_Max")

				wLen_Arr = np.arange(3, wLen_MinSlope+1, 2)
				NSp_Arr = np.zeros_like(wLen_Arr)

				for ii, wLen in enumerate(wLen_Arr):

					if ii % 10 == 0:
						print(f"{ii}: wLen = {wLen}")

					threshLevel = 90
					thresh = np.percentile(noMed, pThresh)
					while True:
						peakIdx, foo = sig.find_peaks(noMed, height=thresh, 
							distance=minISI, prominence=prom, wlen=wLen)

						peakVals = noMed[peakIdx]

						if len(peakVals) < 2:
							break


						oThr = otsu(peakVals)

						lowMed = np.median(peakVals[peakVals <= oThr])
						highMed = np.median(peakVals[peakVals > oThr])

						if (highMed - lowMed <= 5):
							break

						threshLevel += (100 - threshLevel)/2.
						thresh = np.percentile(noMed, threshLevel)

					NSp_Arr[ii] = len(peakIdx)

				NSp_Mode = st.mode(NSp_Arr).mode[0]

				wLen = wLen_Arr[(NSp_Arr == NSp_Mode).nonzero()[0][0]]

				print(f"\nNSp_Mode = {NSp_Mode} giving wLen = {wLen}")

				threshLevel = 90
				thresh = np.percentile(noMed, pThresh)
				while True:
					peakIdx, foo = sig.find_peaks(noMed, height=thresh, 
						distance=minISI, prominence=prom, wlen=wLen)

					peakVals = noMed[peakIdx]

					if len(peakVals) < 2:
						break


					oThr = otsu(peakVals)

					lowMed = np.median(peakVals[peakVals <= oThr])
					highMed = np.median(peakVals[peakVals > oThr])

					if (highMed - lowMed <= 5):
						break

					threshLevel += (100 - threshLevel)/2.
					thresh = np.percentile(noMed, threshLevel)

				print(f"\n{len(peakIdx)} Peaks Found..."+
					f"\nThresh = {thresh:.4g}"+
					f"\nProm = {prom:.4g}"+
					f"\nwLen = {wLen}")



			fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(16, 7))

			# if protocol == EPHYS_PROT_DEPOLSTEP:
			# 	dpData, dpIdx, dpI = epu.getDepolIdx(data[:, 0].squeeze(), hdr,
			# 		protocol=EPHYS_PROT_DEPOLSTEP)
			# else:
			# 	dpData = deepcopy(data[:, 0].squeeze())
			# grid = np.arange(len(dpData))

			ax1.plot(grid, dpData)

			# window, perc = 201, 50
			# order = int(window*perc/100.)
			# medArr = sig.order_filter(dpData, np.ones(window), order)

			# noMed = dpData - medArr

			ax2.plot(grid, noMed)

			# smoother = intrp.UnivariateSpline(grid, noMed,
			# 	s=min(int(len(data)/10.), 5000))
			# smthNoMed = smoother(grid)

			# # ax2.plot(grid, smthNoMed, color='orange')

			# dt, maxRate = 0.001, 50.
			# minISI = int(1./dt/maxRate)


			# noThresh = None
			# tMed90 = np.percentile(noMed, 90)
			# tMed99 = np.percentile(noMed, 99)
			# print(f"\nThresholds\n\ttMed90: {tMed90:.4g}\n\t"+
			# 	f"tMed99: {tMed99:.4g}")


			# # plt.hlines(y=[tSmth90, tSmth99], xmin=0, xmax=len(grid),
			# # 	color='orange')

			# pmin = 1
			# noProm = None
			# minProm = 5
			# pMed90 = np.diff(np.percentile(noMed, [pmin, 90]))[0]
			# pMed99 = np.diff(np.percentile(noMed, [pmin, 99]))[0]

			# print(f"Prominences\n\tpMed90: {pMed90:.4g}\n\t"+
			# 	f"pMed99: {pMed99:.4g}")

			# thresh = tMed90
			# prom = pMed90
			# wlenThresh = 500

			# wlens = np.arange(2,1000)
			# n_peaks_arr = np.zeros_like(wlens)

			# for ii, wlen in enumerate(wlens):
			# 	# n_peaks_arr[ii] = len(sig.find_peaks(noMed, height=thresh,
			# 	# 	distance=minISI, prominence=prom, wlen=wlen)[0])

			# 	if ii % 100 == 0:
			# 		print(f"{ii}: wlen = {wlen}")
			# 	threshLevel = 90
			# 	while True:

			# 		if wlen == wlens[-1]:
			# 			plt.hlines(y=thresh, xmin=0, xmax=len(grid), color='k')

			# 		peakIdx, foo = sig.find_peaks(noMed, height=thresh,
			# 			distance=minISI, prominence=prom, wlen=wlen)


			# 		peakVals = noMed[peakIdx]

			# 		# D, P = dipTest(peakVals)
			# 		# print(f"\nThe likelihood that this data is unimodal is {P:.4g}"+
			# 		# 	f" (D = {D:.4g})")

			# 		if len(peakVals) < 2:
			# 			break


					# oThr = otsu(peakVals)

					# lowMed = np.median(peakVals[peakVals <= oThr])
					# highMed = np.median(peakVals[peakVals > oThr])
					# # print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals <= oThr)} on left"+
					# # 	f" {sum(peakVals > oThr)} on right)")

					# # print(f"Low Median {lowMed:.4g}, High Median {highMed:.4g}")

					# # if (P > .3) and (highMed - lowMed <= 5):
					# # 	break

					# if (highMed - lowMed <= 5):
					# 	break

					# threshLevel += (100 - threshLevel)/2.
					# thresh = np.percentile(noMed, threshLevel)
					# # print(f"The threshold level is {threshLevel} ({thresh:.4g})")


			# 	n_peaks_arr[ii] = len(peakIdx)

			ax1.scatter(peakIdx, dpData[peakIdx],
				c='r', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			ax2.scatter(peakIdx, noMed[peakIdx],
				c='r', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")


			ax1.plot(grid, thresh+medArr, color='k')

			ax2.hlines(y=thresh, xmin=grid[0], xmax=grid[-1], color='k')

			# # peakIdx2, foo2 = sig.find_peaks(noMed, height=tMed99,
			# # 	distance=minISI, prominence=prom, wlen=200)

			# # D2, P2 = dipTest(noMed[peakIdx2])
			# # print(f"The likelihood that this data is unimodal is {P2:.4g}")

			# # ax1.scatter(peakIdx2+10, dpData[peakIdx2]+0.5,
			# # 	c='g', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			# # ax2.scatter(peakIdx2+10, noMed[peakIdx2]+0.5,
			# # 	c='g', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			# ax1.legend()
			# ax2.legend()

			# # threshs = {'None':noThresh,
			# # 	"tMed90":tMed90,
			# # 	"tMed99":tMed99,
			# # 	"tSmth90":tSmth90,
			# # 	"tSmth99":tSmth99}

			# # proms = {'None':noProm,
			# # 	"Fixed":minProm,
			# # 	"pMed90":pMed90,
			# # 	"pMed99":pMed99,
			# # 	"pSmth90":pSmth90,
			# # 	"pSmth99":pSmth99}

			# # for tNo, (tName, thresh) in enumerate(threshs.items()):
			# # 	marker = "o*^sX"[tNo]
			# # 	for pNo, (pName, prom) in enumerate(proms.items()):
			# # 		color = "krgymc"[pNo]

			# # 		print(f"Finding peaks for {tName}, {pName}")

			# # 		peakIdx, foo = sig.find_peaks(noMed, height=thresh,
			# # 			distance=minISI, prominence=prom, wlen=200)

			# # 		ax1.scatter(peakIdx+tNo*10, dpData[peakIdx]+pNo*.5,
			# # 			c=color, marker=marker, label=f"{tName}, {pName}")

			# # 		ax2.scatter(peakIdx+tNo*10, noMed[peakIdx]+pNo*.5,
			# # 			c=color, marker=marker, label=f"{tName}, {pName}")


			ax1.legend(fontsize=10)


			ax2.legend(fontsize=10)




			fig.tight_layout()

			fig.suptitle(f"Key = {key}", fontsize=24)

			# fig.tight_layout()


			# # allIdx, allFoo = sig.find_peaks(noMed, height=None,
			# # 	distance=minISI, prominence=prom, wlen=200)

			# # bins = np.linspace(np.min(noMed[allIdx]), np.max(noMed[allIdx]), 11)

			# # lowIdx, lowFoo = sig.find_peaks(noMed, height=tMed90,
			# # 	distance=minISI, prominence=prom, wlen=200)
			# # highIdx, highFoo = sig.find_peaks(noMed, height=thresh,
			# # 	distance=minISI, prominence=prom, wlen=200)


			# fig2, axWLen = plt.subplots(1, 1, figsize=(8, 6))

			# axWLen.scatter(wlens, n_peaks_arr)

			# axWLen.set_xscale('log')

			# minWLen = wlens[(n_peaks_arr == n_peaks_arr.max()).nonzero()[0][0]]
			# axWLen.vlines(x=minWLen, ymin=0, ymax=n_peaks_arr.max(),
			# 	label=r'$w_{len} = '+f"{minWLen}" + r"$", color='g')

			# print(f"\n\nMinimum Spike Slope = {prom/minWLen*1000:.4g}")

			# minSlope = 20.
			# minSlopeW = prom*1000/minSlope

			# axWLen.vlines(x=minSlopeW, ymin=0, ymax=n_peaks_arr.max(), color='r',
			# 	label=r"$w_{minslope = "+f"{minSlope:.4g}"+r"} = "+f"{minSlopeW:.4g}" + r"$")

			# axWLen.vlines(x=wlenThresh, ymin=0, ymax=n_peaks_arr.max(),
			# 	label=r'$w_{len}^{Thresh} = '+f"{wlenThresh:.4g}" + r"$", color='k')


			# axWLen.legend()

			# # fig2, axDist = plt.subplots(1,1, figsize=(8, 6))

			# # sns.distplot(noMed[allIdx], hist_kws=dict(cumulative=False), bins=bins,
			# # 	kde_kws=dict(cumulative=False), ax=axDist, label='All Idx')

			# # sns.distplot(noMed[lowIdx], hist_kws=dict(cumulative=False), bins=bins,
			# # 	kde_kws=dict(cumulative=False), ax=axDist, label='Low Idx')

			# # sns.distplot(noMed[highIdx], hist_kws=dict(cumulative=False), bins=bins,
			# # 	kde_kws=dict(cumulative=False), ax=axDist, label='High Idx')

			# # lowThresh = otsu(noMed[lowIdx])
			# # highThresh = otsu(noMed[highIdx])

			# # axDist.vlines(x=[lowThresh, highThresh], ymin=0, ymax=0.5, color=['r', 'g'])

			# # axDist.legend()

			# # fig2.suptitle(f"Key = {key}", fontsize=24)

			# # fig2.tight_layout()

			break

			# prots.append([key, protocol])

			# print(f"{key}: {protocol}")

	############################################################################
	## TRYING TO FIGURE OUT SPIKES
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

		break

	# print("\n".join([", ".join([f"{p}" for p in pline]) for pline in prots]))

	with open(dataFeatPath, "wb") as f:
		pkl.dump(dataFeat, f)


################################################################################
## Show plots!
################################################################################
	plt.show()
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
from Misc.dip import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import scipy.interpolate as intrp
import scipy.signal as sig
from skimage.filters import threshold_otsu as otsu

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

		if dateNo != 1:
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

			fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(16, 7))

			if protocol == EPHYS_PROT_DEPOLSTEP:
				dpData, dpIdx, dpI = epu.getDepolIdx(data[:, 0].squeeze(), hdr,
					protocol=EPHYS_PROT_DEPOLSTEP)
			else:
				dpData = deepcopy(data[:, 0].squeeze())
			grid = np.arange(len(dpData))

			ax1.plot(grid, dpData)

			window, perc = 201, 50
			order = int(window*perc/100.)
			medArr = sig.order_filter(dpData, np.ones(window), order)

			noMed = dpData - medArr

			ax2.plot(grid, noMed)

			smoother = intrp.UnivariateSpline(grid, noMed,
				s=min(int(len(data)/10.), 5000))
			smthNoMed = smoother(grid)

			# ax2.plot(grid, smthNoMed, color='orange')

			dt, maxRate = 0.001, 50.
			minISI = int(1./dt/maxRate)



			noThresh = None
			tMed90 = np.percentile(noMed, 90)
			tMed99 = np.percentile(noMed, 99)
			tSmth90 = np.percentile(smthNoMed, 90)
			tSmth99 = np.percentile(smthNoMed, 99)
			print(f"\nThresholds\n\ttMed90: {tMed90:.4g}\n\t"+
				f"tMed99: {tMed99:.4g}\n\ttSmth90: {tSmth90:.4g}\n\t"+
				f"tSmth99: {tSmth99:.4g}")

			plt.hlines(y=[tMed90, tMed99], xmin=0, xmax=len(grid), color='b')

			# plt.hlines(y=[tSmth90, tSmth99], xmin=0, xmax=len(grid),
			# 	color='orange')

			pmin = 1
			noProm = None
			minProm = 5
			pMed90 = np.diff(np.percentile(noMed, [pmin, 90]))[0]
			pMed99 = np.diff(np.percentile(noMed, [pmin, 99]))[0]
			pSmth90 = np.diff(np.percentile(smthNoMed, [pmin, 90]))[0]
			pSmth99 = np.diff(np.percentile(smthNoMed, [pmin, 99]))[0]

			print(f"Prominences\n\tpMed90: {pMed90:.4g}\n\t"+
				f"pMed99: {pMed99:.4g}\n\tpSmth90: {pSmth90:.4g}\n\t"+
				f"pSmth99: {pSmth99:.4g}")

			thresh = tMed90
			prom = pMed90

			peakIdx, foo = sig.find_peaks(noMed, height=thresh,
				distance=minISI, prominence=prom, wlen=200)

			ax1.scatter(peakIdx, dpData[peakIdx],
				c='r', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			ax2.scatter(peakIdx, noMed[peakIdx],
				c='r', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			peakIdx2, foo2 = sig.find_peaks(noMed, height=tMed99,
				distance=minISI, prominence=prom, wlen=200)

			ax1.scatter(peakIdx2+10, dpData[peakIdx2]+0.5,
				c='g', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			ax2.scatter(peakIdx2+10, noMed[peakIdx2]+0.5,
				c='g', marker='o', label=f"h = {thresh:.4g}, p = {prom:.4g}")

			ax1.legend()
			ax2.legend()

			# threshs = {'None':noThresh,
			# 	"tMed90":tMed90,
			# 	"tMed99":tMed99,
			# 	"tSmth90":tSmth90,
			# 	"tSmth99":tSmth99}

			# proms = {'None':noProm,
			# 	"Fixed":minProm,
			# 	"pMed90":pMed90,
			# 	"pMed99":pMed99,
			# 	"pSmth90":pSmth90,
			# 	"pSmth99":pSmth99}

			# for tNo, (tName, thresh) in enumerate(threshs.items()):
			# 	marker = "o*^sX"[tNo]
			# 	for pNo, (pName, prom) in enumerate(proms.items()):
			# 		color = "krgymc"[pNo]

			# 		print(f"Finding peaks for {tName}, {pName}")

			# 		peakIdx, foo = sig.find_peaks(noMed, height=thresh,
			# 			distance=minISI, prominence=prom, wlen=200)

			# 		ax1.scatter(peakIdx+tNo*10, dpData[peakIdx]+pNo*.5,
			# 			c=color, marker=marker, label=f"{tName}, {pName}")

			# 		ax2.scatter(peakIdx+tNo*10, noMed[peakIdx]+pNo*.5,
			# 			c=color, marker=marker, label=f"{tName}, {pName}")


			# ax1.legend(fontsize=6, ncol=5)


			# ax2.legend(fontsize=6, ncol=5)





			fig.suptitle(f"Key = {key}", fontsize=24)

			fig.tight_layout()


			allIdx, allFoo = sig.find_peaks(noMed, height=None,
				distance=minISI, prominence=prom, wlen=200)

			bins = np.linspace(np.min(noMed[allIdx]), np.max(noMed[allIdx]), 11)

			lowIdx, lowFoo = sig.find_peaks(noMed, height=tMed90,
				distance=minISI, prominence=prom, wlen=200)
			highIdx, highFoo = sig.find_peaks(noMed, height=tMed99,
				distance=minISI, prominence=prom, wlen=200)

			fig2, axDist = plt.subplots(1,1, figsize=(8, 6))

			sns.distplot(noMed[allIdx], hist_kws=dict(cumulative=False), bins=bins,
				kde_kws=dict(cumulative=False), ax=axDist, label='All Idx')

			sns.distplot(noMed[lowIdx], hist_kws=dict(cumulative=False), bins=bins,
				kde_kws=dict(cumulative=False), ax=axDist, label='Low Idx')

			sns.distplot(noMed[highIdx], hist_kws=dict(cumulative=False), bins=bins,
				kde_kws=dict(cumulative=False), ax=axDist, label='High Idx')

			lowThresh = otsu(noMed[lowIdx])
			highThresh = otsu(noMed[highIdx])

			axDist.vlines(x=[lowThresh, highThresh], ymin=0, ymax=0.5, color=['r', 'g'])

			axDist.legend()

			fig2.suptitle(f"Key = {key}", fontsize=24)

			fig2.tight_layout()

			break

			prots.append([key, protocol])

			print(f"{key}: {protocol}")

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



	plt.close('all')


	pltBins = bins[:-1] + np.diff(bins)

	pdf, _ = np.histogram(noMed[lowIdx], bins=bins, density=True)
	pdf /= np.sum(pdf)

	cdf = np.cumsum(pdf)/np.sum(pdf)


	tmp_cdf = cdf - pdf

	plt.plot(pltBins, tmp_cdf)

	tmp_idx = pltBins.copy()

	gcm = [tmp_cdf[0]]
	l_tps = [0]

	while len(tmp_cdf) > 1:
		dists = tmp_idx[1:] - tmp_idx[0]
		slopes = (tmp_cdf[1:] - tmp_cdf[0])/dists

		minslope = slopes.min()
		minIdx = np.where(slopes == minslope)[0][0] + 1

		gcm.extend(tmp_cdf[0] + dists[:minIdx]*minslope)

		l_tps.append(l_tps[-1] + minIdx)

		tmp_cdf = tmp_cdf[minIdx:]
		tmp_idx = tmp_idx[minIdx:]

	plt.plot(pltBins[:len(gcm)], gcm)

	print(l_tps)

	gcm = np.array(gcm)
	l_tps = np.array(l_tps).astype(int)
	plt.scatter(pltBins[l_tps], gcm[l_tps])


	tmp_cdf = 1 - cdf
	tmp_idx = pltBins.max() - pltBins[::-1]

	plt.figure()
	plt.plot(tmp_idx, tmp_cdf)

	lcm = [tmp_cdf[0]]
	r_tps = [0]

	while len(tmp_cdf) > 1:
		dists = tmp_idx[1:] - tmp_idx[0]
		slopes = (tmp_cdf[1:] - tmp_cdf[0])/dists

		minslope = slopes.min()
		minIdx = np.where(slopes == minslope)[0][0] + 1

		lcm.extend(tmp_cdf[0] + dists[:minIdx]*minslope)

		r_tps.append(r_tps[-1] + minIdx)

		tmp_cdf = tmp_cdf[minIdx:]
		tmp_idx = tmp_idx[minIdx:]

	lcm = np.array(lcm)
	r_tps = np.array(r_tps).astype(int)

	tmp_idx = pltBins.max() - pltBins[::-1]
	plt.plot(tmp_idx[:len(lcm)], lcm)

	lcm = 1 - lcm[::-1]
	r_tps = len(cdf) - 1 - r_tps[::-1]

	left_diffs = np.abs((gcm[l_tps] - lcm[l_tps]))
	right_diffs = np.abs((gcm[r_tps] - lcm[r_tps]))

	d_left, d_right = left_diffs.max(), right_diffs.max()


	plt.close('all')

################################################################################
## Show plots!
################################################################################
	plt.show()
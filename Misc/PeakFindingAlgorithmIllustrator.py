"""
================================================================================
	Action Potential Locator Algorithm Illustrator
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, May 8, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This script will generate figures explaining how the peak finding algorithm
	in ephys_obj ("getSpikeIdx") works.

	Key elements that will be illustrated:
	 - Rolling Median Subtraction
	 - Median filter edge correction
	 - Sensitivity of peak detection to the size of the median filter
	 - Figure illustrating the parameters thresh, prom, wlen, and how a slope
	   parameter is enforced
	 - Figure showing histograms and where thresh, prom can be set.
	 - Figure showing how the number of peaks changes at max wlen as thresh and
	   prom are modified
	 - Figure showing how the number of peaks changes for given thresh, prom,
	   as wlen is reduced.
	 - Figure showing how for given prom, wlen, the number of peaks changes as 
	   a function of thresh (also violin plots vs thresh?)


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

	figDir = "./Figures/PeakFindingAlgorithmIllustrator/"

	gen_new_data = True

	## Plotting flags
	plot_fig1_rawData = False
	plot_fig2_medSubtract = False
	plot_fig3_medEdgeCorrect = False
	plot_fig4_medWindowSens = False
	plot_fig5_paramDefns = False
	plot_fig6_voltageHists = False
	plot_fig7_NPeaks_vs_thresh_prom = False
	plot_fig8_NPeaks_vs_wlen = False
	plot_fig9_NPeak_vs_ALL = True

	infoDir = "./Runfiles/PeakFindingAlgorithmIllustrator/"

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
## Load dateNo
################################################################################
	
	dateNo = 0

	date = WT.index.values[dateNo]

	dateStr = date[-2:] + "/" + date[-5:-3] + "/" + date[:4]

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

################################################################################
## Iterate through Recordings
################################################################################

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

		## Load the data, hdr, get the protocol.
		data = dataDict[dateStr][key]['data']
		hdr = dataDict[dateStr][key]['header']

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

	dt = infoDict['data']['dt']
	maxRate = infoDict['objectives']['Spikes']['maxRate']
	minISI = 1./dt/maxRate

	data22 = data22[:, 0].squeeze()
	tGrid22 = np.arange(len(data22))*dt
	wave22 = abf.GetWaveformEx(hdr22, 0, 0)[:, 0]*1000.

	data23 = data23[:, 0].squeeze()
	dpData23, dpIdx23, dpI23 = epu.getDepolIdx(data23, hdr23,
		protocol=EPHYS_PROT_DEPOLSTEP)
	tGrid23 = np.arange(len(data23))*dt
	dpTGrid23 = np.arange(len(dpData23))*dt + tGrid23[dpIdx23[0]]
	wave23 = abf.GetWaveformEx(hdr23, 0, 0)[:, 0]*1000.

	data24 = data24[:, 0].squeeze()
	dpData24, dpIdx24, dpI24 = epu.getDepolIdx(data24, hdr24,
		protocol=EPHYS_PROT_DEPOLSTEP)
	tGrid24 = np.arange(len(data24))*dt
	dpTGrid24 = np.arange(len(dpData24))*dt + tGrid24[dpIdx24[0]]
	wave24 = abf.GetWaveformEx(hdr24, 0, 0)[:, 0]*1000.

################################################################################
## Plot Raw Data
################################################################################

	if plot_fig1_rawData:

		## Recording 22
		fig1a, [ax1a_1, ax1a_2] = plt.subplots(2, 1, figsize=(12, 7),
			sharex=True)

		ax1a_1.plot(tGrid22, data22)

		ax1a_1.set_ylabel("Voltage (mV)")

		ax1a_2.plot(tGrid22, wave22)

		ax1a_2.set_xlabel("Time (s)")
		ax1a_2.set_ylabel(r"$I_{Injected}$ (pA)")

		ax1a_1.set_title(f"{date}: Recording 22")

		fig1a.tight_layout()

		fig1aName = "Recording22.pdf"
		fig1aPath = os.path.join(figDir, fig1aName)
		fig1a.savefig(fig1aPath, format='pdf', dpi=600)


		## Recording 23
		fig1b, [ax1b_1, ax1b_2] = plt.subplots(2, 1, figsize=(12, 7),
			sharex=True)

		ax1b_1.plot(tGrid23, data23)

		ax1b_1.set_ylabel("Voltage (mV)")

		ax1b_2.plot(tGrid23, wave23)

		ax1b_2.set_xlabel("Time (s)")
		ax1b_2.set_ylabel(r"$I_{Injected}$ (pA)")

		ax1b_1.set_title(f"{date}: Recording 23")

		fig1b.tight_layout()

		fig1bName = "Recording23.pdf"
		fig1bPath = os.path.join(figDir, fig1bName)
		fig1b.savefig(fig1bPath, format='pdf', dpi=600)


		## Recording 24
		fig1c, [ax1c_1, ax1c_2] = plt.subplots(2, 1, figsize=(12, 7),
			sharex=True)

		ax1c_1.plot(tGrid24, data24)

		ax1c_1.set_ylabel("Voltage (mV)")

		ax1c_2.plot(tGrid24, wave24)

		ax1c_2.set_xlabel("Time (s)")
		ax1c_2.set_ylabel(r"$I_{Injected}$ (pA)")
		ax1c_2.set_yticks(np.arange(-15, 15, 5))

		ax1c_1.set_title(f"{date}: Recording 24")

		fig1c.tight_layout()

		fig1cName = "Recording24.pdf"
		fig1cPath = os.path.join(figDir, fig1cName)
		fig1c.savefig(fig1cPath, format='pdf', dpi=600)

################################################################################
## Plot Median Subtraction
################################################################################
	
	## Set filter window for 22
	window22 = 100 #int(len(data22)/100)
	window22 = window22 + 1 if (window22 % 2 == 0) else window22

	## Get median for 22 (with edge correction!)
	med22 = epu.getRollPerc(data22, window=int(window22), perc=50)
	noMed22 = data22-med22 ## Subtract the median from the data

	## Set filter window for 23
	window23 = 100 #int(len(dpData23)/100)
	window23 = window23 + 1 if (window23 % 2 == 0) else window23

	## Get median for 23 (with edge correction!)
	med23 = epu.getRollPerc(dpData23, window=int(window23), perc=50)
	noMed23 = dpData23-med23 ## Subtract the median from the data

	## Set filter window for 24
	window24 = 100 #nt(len(dpData24)/100)
	window24 = window24 + 1 if (window24 % 2 == 0) else window24

	## Get median for 24 (with edge correction!)
	med24 = epu.getRollPerc(dpData24, window=int(window24), perc=50)
	noMed24 = dpData24-med24 ## Subtract the median from the data

	if plot_fig2_medSubtract:

		## Recording 22
		fig2a, [ax2a_1, ax2a_2] = plt.subplots(2, 1, figsize=(12, 7),
			sharex=True)

		ax2a_1.plot(tGrid22, data22)

		ax2a_1.set_ylabel("Voltage (mV)")

		ax2a_2.plot(tGrid22, noMed22)

		ax2a_2.set_xlabel("Time (s)")
		ax2a_2.set_ylabel("Median-Sub.\nVoltage (mV)")

		ax2a_1.set_title(f"{date}: Recording 22  "+r"($I_{Inj} = $"+
			f"{wave22[0]:.2f} pA)")

		fig2a.tight_layout()

		fig2aName = "NoMedianRecording22.pdf"
		fig2aPath = os.path.join(figDir, fig2aName)
		fig2a.savefig(fig2aPath, format='pdf', dpi=600)


		## Recording 23
		fig2b, [ax2b_1, ax2b_2] = plt.subplots(2, 1, figsize=(12, 7),
			sharex=True)

		ax2b_1.plot(dpTGrid23, dpData23)

		ax2b_1.set_ylabel("Voltage (mV)")

		ax2b_2.plot(dpTGrid23, noMed23)

		ax2b_2.set_xlabel("Time (s)")
		ax2b_2.set_ylabel("Median-Sub.\nVoltage (mV)")

		ax2b_1.set_title(f"{date}: Recording 23  "+r"($I_{Inj} = $"+
			f"{dpI23[0]*1000:.2f} pA)")

		fig2b.tight_layout()

		fig2bName = "NoMedianRecording23.pdf"
		fig2bPath = os.path.join(figDir, fig2bName)
		fig2b.savefig(fig2bPath, format='pdf', dpi=600)


		## Recording 24
		fig2c, [ax2c_1, ax2c_2] = plt.subplots(2, 1, figsize=(12, 7),
			sharex=True)

		ax2c_1.plot(dpTGrid24, dpData24)

		ax2c_1.set_ylabel("Voltage (mV)")

		ax2c_2.plot(dpTGrid24, noMed24)

		ax2c_2.set_xlabel("Time (s)")
		ax2c_2.set_ylabel("Median-Sub.\nVoltage (mV)")

		ax2c_1.set_title(f"{date}: Recording 24  "+r"($I_{Inj} = $"+
			f"{dpI24[0]*1000:.2f} pA)")

		fig2c.tight_layout()

		fig2cName = "NoMedianRecording24.pdf"
		fig2cPath = os.path.join(figDir, fig2cName)
		fig2c.savefig(fig2cPath, format='pdf', dpi=600)

################################################################################
## Show Median Filter Edge Correction
################################################################################
	if plot_fig3_medEdgeCorrect:

		rawMed22 = sig.order_filter(data22, np.ones(window22),
			int((window22-1)/2))

		rawMed23 = sig.order_filter(dpData23, np.ones(window23),
			int((window23-1)/2))

		rawMed24 = sig.order_filter(dpData24, np.ones(window24),
			int((window24-1)/2))

		outerPad = 0.1
		innerPad = 0.3

		fig3, ax3 = plt.subplots(3, 2, figsize=(10, 14))

		ax3[0, 0].plot(tGrid22, noMed22, label='Median Correction')

		ax3[0, 0].plot(tGrid22, data22-rawMed22, label='No Correction')

		ax3[0, 0].set_xlim(tGrid22[0]-outerPad, tGrid22[0]+innerPad)


		ax3[0, 1].plot(tGrid22, noMed22, label='Median Correction')

		ax3[0, 1].plot(tGrid22, data22-rawMed22, label='No Correction')

		ax3[0, 1].set_xlim(tGrid22[-1]-innerPad, tGrid22[-1]+outerPad)


		ax3[1, 0].plot(dpTGrid23, noMed23, label='Median Correction')

		ax3[1, 0].plot(dpTGrid23, dpData23-rawMed23, label='No Correction')

		ax3[1, 0].set_xlim(dpTGrid23[0]-outerPad, dpTGrid23[0]+innerPad)


		outerPad = 0.05
		innerPad = 0.15


		ax3[1, 1].plot(dpTGrid23, noMed23, label='Median Correction')

		ax3[1, 1].plot(dpTGrid23, dpData23-rawMed23, label='No Correction')

		ax3[1, 1].set_xlim(dpTGrid23[-1]-innerPad, dpTGrid23[-1]+outerPad)


		ax3[2, 0].plot(dpTGrid24, noMed24, label='Median Correction')

		ax3[2, 0].plot(dpTGrid24, dpData24-rawMed24, label='No Correction')

		ax3[2, 0].set_xlim(dpTGrid24[0]-outerPad, dpTGrid24[0]+innerPad)


		ax3[2, 1].plot(dpTGrid24, noMed24, label='Median Correction')

		ax3[2, 1].plot(dpTGrid24, dpData24-rawMed24, label='No Correction')

		ax3[2, 1].set_xlim(dpTGrid24[-1]-innerPad, dpTGrid24[-1]+outerPad)


		ax3[0, 0].set_ylabel("Recording 22\nVoltage (mV)")
		ax3[0, 0].legend(fontsize=12)

		ax3[1, 0].set_ylabel("Recording 23\nVoltage (mV)")

		ax3[2, 0].set_ylabel("Recording 24\nVoltage (mV)")

		ax3[2, 0].set_xlabel("Time (s)")

		ax3[2, 1].set_xlabel("Time (s)")


		fig3.tight_layout()

		fig3Name = "EdgeCorrectionIllustration_NoMed.pdf"
		fig3Path = os.path.join(figDir, fig3Name)
		fig3.savefig(fig3Path, format='pdf', dpi=600)

################################################################################
## Show Median Filter Window Size Sensitivity
################################################################################
	if plot_fig4_medWindowSens:

		windowFracArr = np.asarray([50, 100, 200, 500])

		fig4, [ax4_1, ax4_2, ax4_3] = plt.subplots(3, 1, figsize=(12, 8))
		
		for windFrac in windowFracArr:

			## Set filter window
			wind22 = int(len(data22)/windFrac)
			wind22 = wind22 + 1 if (wind22 % 2 == 0) else wind22

			med_22 = epu.getRollPerc(data22, window=wind22, perc=50., verbose=3)

			ax4_1.plot(tGrid22, med_22, #data22-med_22,
				label=f'WindFrac = 1/{windFrac}')


			wind23 = int(len(dpData23)/windFrac)
			wind23 = wind23 + 1 if (wind23 % 2 == 0) else wind23

			med_23 = epu.getRollPerc(dpData23, window=wind23, perc=50.,
				verbose=3)

			ax4_2.plot(dpTGrid23, med_23, #dpData23-med_23,
				label=f'WindFrac = 1/{windFrac}')


			wind24 = int(len(dpData24)/windFrac)
			wind24 = wind24 + 1 if (wind24 % 2 == 0) else wind24

			med_24 = epu.getRollPerc(dpData24, window=wind24, perc=50.,
				verbose=3)

			ax4_3.plot(dpTGrid24, med_24, #dpData24-med_24,
				label=f'WindFrac = 1/{windFrac}')

		ax4_1.set_xlabel("Time (s)")
		ax4_1.set_ylabel("Recording 22\nVoltage (mV)")
		ax4_1.legend(fontsize=10)

		ax4_2.set_xlabel("Time (s)")
		ax4_2.set_ylabel("Recording 23\nVoltage (mV)")
		ax4_2.legend(fontsize=10)

		ax4_3.set_xlabel("Time (s)")
		ax4_3.set_ylabel("Recording 24\nVoltage (mV)")
		ax4_3.legend(fontsize=10)

		fig4.tight_layout()

		fig4Name = "MedianFilter_vs_WindowSize_NoMed.pdf"
		fig4Path = os.path.join(figDir, fig4Name)
		fig4.savefig(fig4Path, format='pdf', dpi=600)

################################################################################
## Show Parameter Definitions
################################################################################
	if plot_fig5_paramDefns:

		lIdx, rIdx = 1150, 1400
		t = dpTGrid24[lIdx:rIdx]
		x = dpData24[lIdx:rIdx]

		wlen = 100
		minProm = 5
		thresh = -24.

		peakIdx, peakInfo = sig.find_peaks(x, prominence=minProm, width=0,
			wlen=wlen, rel_height=1)

		pkId = 1
		pkIdx = peakIdx[pkId]
		pkHgt = x[pkIdx]
		pkTime = t[pkIdx]
		pkProm = peakInfo['prominences'][pkId]
		lB, rB = peakInfo['left_bases'][pkId], peakInfo['right_bases'][pkId]
		pkWdHgt = peakInfo['width_heights'][pkId]
		pkWid = peakInfo['widths'][pkId]

		fig5, ax5 = plt.subplots(1, 1, figsize=(12, 8))

		ax5.plot(t, x)

		ax5.plot(pkTime, pkHgt, "X", color='r')

		ax5.vlines(x=pkTime, ymin=pkWdHgt, ymax=pkHgt, color='r')

		ax5.hlines(y=pkWdHgt, xmin=pkTime-wlen*dt/2, xmax=pkTime+wlen*dt/2,
			color='k')
		ax5.hlines(y=pkWdHgt, xmin=t[lB], xmax=t[rB], color='g')

		ax5.text(pkTime-0.005, pkWdHgt+0.2, f"Peak Width = {pkWid*dt:.2f}s",
			verticalalignment='bottom', horizontalalignment='right',
			bbox={'facecolor':'g', 'alpha':0.5}, fontsize=10)

		ax5.text(pkTime+0.01, pkWdHgt + pkProm/2, f"Prominence = {pkProm:.2f}mV",
			verticalalignment='center', horizontalalignment='left',
			bbox={'facecolor':'r', 'alpha':0.5}, fontsize=10)

		ax5.plot(2*[pkTime+0.02], [pkWdHgt, pkWdHgt+pkProm/2-0.3], "-+",
			color='r')
		ax5.plot(2*[pkTime+0.02], [pkHgt, pkWdHgt+pkProm/2+0.3], "-+",
			color='r')

		ax5.text(pkTime, pkWdHgt-0.3, r"$wlen$ = "+f"{wlen*dt:.2f}s",
			verticalalignment='top', horizontalalignment='center',
			bbox={'facecolor':'k', 'alpha':0.3}, fontsize=10)

		ax5.plot([t[lB], pkTime-0.014], 2*[pkWdHgt-0.5], '-+', color='k')
		ax5.plot([pkTime+wlen/2*dt, pkTime+0.014], 2*[pkWdHgt-0.5], '-+',
			color='k')

		xlim = ax5.get_xlim()
		ax5.hlines(y=thresh, xmin=xlim[0], xmax=xlim[1], color='k',
			linestyle='dashed')
		ax5.set_xlim(xlim)

		ax5.text(6.7, thresh+.2, r"$thresh = $"+f"{thresh:.1f}mV",
			verticalalignment='bottom', horizontalalignment='left',
			bbox={'facecolor':'k', 'alpha':0.3}, fontsize=10)

		ax5.text(pkTime-0.01, pkHgt - minProm/2.,
			r"$minProm = $"+f"{minProm:.2f}mV",
			verticalalignment='center', horizontalalignment='right',
			bbox={'facecolor':'k', 'alpha':0.3}, fontsize=10)

		ax5.plot(2*[pkTime-0.02], [pkHgt-minProm/2.+0.27, pkHgt], "-+",
			color='k')
		ax5.plot(2*[pkTime-0.02], [pkHgt-minProm/2.-0.27, pkHgt-minProm], "-+",
			color='k')

		ax5.plot([pkTime, pkTime+wlen/2*dt, pkTime],
			[pkHgt, pkHgt-minProm, pkHgt-minProm], color='k')

		ax5.text(6.64, pkHgt - minProm/2.,
			r"$minSlope = $"+f"{minProm/wlen/dt:.2f}mV/s",
			verticalalignment='center', horizontalalignment='center',
			bbox={'facecolor':'k', 'alpha':0.3}, fontsize=10, rotation=-50)
		ax5.set_xlabel("Time (s)")
		ax5.set_ylabel("Voltage (mV)")

		fig5.tight_layout()

		fig5Name = "PeakFind_ParamIllustration.pdf"
		fig5Path = os.path.join(figDir, fig5Name)
		fig5.savefig(fig5Path, format='pdf', dpi=600)
	
################################################################################
## Show Common Thresholds on Histograms
################################################################################
	if plot_fig6_voltageHists:

		tArr = [75, 90, 95, 99, 99.9]
		pArr = [(5, 90), (1, 90), (1, 99)]

		fig6, ax6 = plt.subplots(3, 1, figsize=(12, 10))

		cmap = sns.color_palette()

		for figNo in range(3):

			if figNo == 0:
				data = noMed22
			elif figNo == 1:
				data = noMed23
			else:
				data = noMed24

			sns.distplot(data, ax=ax6[figNo], kde=False, bins=25)

			ax6[figNo].set_yscale('log')

			ylim = ax6[figNo].get_ylim()
			for ii, pT in enumerate(tArr):

				t = np.percentile(data, pT)
				ax6[figNo].vlines(x=t, ymin=ylim[0], ymax=ylim[1],
					color=cmap[1])

				ax6[figNo].text(t+0.05, 2000,
					r"$t_{"+f"{pT:.1f}"+r"}=$"+f"{t:.2f}", fontsize=8,
					verticalalignment='top', horizontalalignment='left')

			for ii, (pP1, pP2) in enumerate(pArr):
				p1, p2 = np.percentile(data, pP1), np.percentile(data, pP2)

				ax6[figNo].hlines(y=10+10*ii, xmin=p1, xmax=p2, color=cmap[2])

				ax6[figNo].text((p1+p2)/2., 11+10*ii,
					r"$Prom_{("+f"{pP1:.0f},{pP2:.0f}"+r")}=$"+f"{p2-p1:.1f}",
					fontsize=8, verticalalignment='bottom',
					horizontalalignment='center')

			ax6[figNo].set_ylim(ylim)

			ax6[figNo].set_xlabel("Voltage (mV)")
			ax6[figNo].set_ylabel("No. Measurements")

		fig6.tight_layout()

		fig6Name = "HistogramThresholdExamples.pdf"
		fig6Path = os.path.join(figDir, fig6Name)
		fig6.savefig(fig6Path, format='pdf', dpi=600)
	
################################################################################
## Show Dependence of Num Peaks on Thresh, Prom, (wlen max!)
################################################################################
	if plot_fig7_NPeaks_vs_thresh_prom:

		tPercArr = np.arange(80, 100, 0.2)
		pPercArr = [75, 90, 95, 99, 99.9]

		fig7, ax7 = plt.subplots(3, 2, figsize=(12, 12),
			gridspec_kw={'width_ratios':[1,1]})

		cmap = sns.color_palette()

		for figNo in range(3):

			if figNo == 0:
				data = noMed22
				time = tGrid22
				recNo = 22
			elif figNo == 1:
				data = noMed23
				time = dpTGrid23
				recNo = 23
			else:
				data = noMed24
				time = dpTGrid24
				recNo = 24

			ax7[figNo, 1].plot(time, data, color='k', lw=0.5, alpha=0.5)

			wlen = 2*len(data)

			for ii, pPerc in enumerate(pPercArr):

				minProm = np.diff(np.percentile(data, (1, pPerc)))[0]

				NPeaks = []
				for jj, tPerc in enumerate(tPercArr):

					thresh = np.percentile(data, tPerc)

					peakIdx, _ = sig.find_peaks(data, height=thresh,
						prominence=minProm, wlen=wlen)

					NPeaks.append(len(peakIdx))


				ax7[figNo, 0].scatter(tPercArr, NPeaks,
					label=r'$Prom_{(1,'+f"{pPerc:.0f}"+")}=$"+f"{minProm:.2f}",
					color=cmap[ii])

				thresh = np.percentile(data, tPercArr[0])

				peakIdx, _ = sig.find_peaks(data, height=thresh,
					prominence=minProm, wlen=wlen)

				ax7[figNo, 1].scatter(time[peakIdx], data[peakIdx],
					color=cmap[ii])

			ax7[figNo, 0].set_xlabel("Threshold Percentile")
			ax7[figNo, 0].set_ylabel(f"Recording {recNo}\nNo. Peaks")
			ax7[figNo, 0].legend(fontsize=10, loc=2)

			ax7[figNo, 1].set_xlabel("Time (s)")
			ax7[figNo, 1].set_ylabel("Voltage (mV)")

		fig7.tight_layout()

		fig7Name = "NPeaks_vs_Thresh_Prom_at_WLenMax.pdf"
		fig7Path = os.path.join(figDir, fig7Name)
		fig7.savefig(fig7Path, format='pdf', dpi=600)
	
################################################################################
## Show Dependence of Num Peaks on wlen (given Thresh, Prom)
################################################################################
	if plot_fig8_NPeaks_vs_wlen:

		tPercMin = 75
		pPerc = 75

		paramTuples = [(75, 75),
			(90, 75),
			(90, 90)]

		fig8, ax8 = plt.subplots(3, 2, figsize=(12, 12))
		cmap = sns.color_palette()

		for figNo in range(3):

			if figNo == 0:
				data = noMed22
				time = tGrid22
				recNo = 22
			elif figNo == 1:
				data = noMed23
				time = dpTGrid23
				recNo = 23
			else:
				data = noMed24
				time = dpTGrid24
				recNo = 24

			ax8[figNo, 1].plot(time, data, color='k', lw=0.5, alpha=0.5)

			wlenArr = np.arange(3, 2*len(data))
			wlenArr = wlenArr[wlenArr % 2 == 1]

			for ii, pTuple in enumerate(paramTuples):
				thresh = np.percentile(data, pTuple[0])
				minProm = np.diff(np.percentile(data, [1, pTuple[1]]))[0]

				print(f"\nT{pTuple[0]} = {thresh:.2f}")
				print(f"Prom{pTuple[1]} = {minProm:.2f}")

				peakIdx, _ = sig.find_peaks(data, height=thresh,
					distance=minISI, prominence=minProm, wlen=wlenArr[-1])
				maxPeaks = len(peakIdx)

				maxWLen = 1000

				NPeaks, NPeaksList = 0, []
				itr = 0
				wlen = wlenArr[itr]
				while (wlen <= maxWLen) or (NPeaks < maxPeaks):

					wlen = wlenArr[itr]

					peakIdx, _ = sig.find_peaks(data, height=thresh,
						distance=minISI, prominence=minProm, wlen=wlen)

					NPeaks = len(peakIdx)

					NPeaksList.append(NPeaks)

					itr += 1

					if itr > len(data):
						raise ValueError

				label = r"$T_{"+f"{pTuple[0]}"+r"}="+f"{thresh:.2f}"+r"$mV, "
				label += r"$Prom_{(1,"+f"{pTuple[1]}"+r")}="+f"{minProm:.2f}"
				label += r"$mV"
				ax8[figNo, 0].scatter(wlenArr[:itr], NPeaksList,
					label=label)

				if ii >= 1:
					minSlope = 20
					minWLen = minProm/minSlope/dt

					ylim = ax8[figNo, 0].set_ylim()

					ax8[figNo, 0].vlines(x=minWLen, ymin=ylim[0], ymax=ylim[1],
						color=cmap[ii], label=r'$wlen_{minSlope}$')

				ax8[figNo, 1].scatter(time[peakIdx], data[peakIdx],
					color=cmap[ii])


			ax8[figNo, 0].set_xscale('log')

			ax8[figNo, 0].set_xlabel(r"$wlen$")
			ax8[figNo, 0].set_ylabel(f"Recording {recNo}\nNo. Peaks")

			ax8[figNo, 0].legend(fontsize=6, loc=4)

			ax8[figNo, 1].set_xlabel("Time (s)")
			ax8[figNo, 1].set_ylabel("Voltage (mV)")



		fig8.tight_layout()

		fig8Name = "NPeaks_vs_wlen.pdf"
		fig8Path = os.path.join(figDir, fig8Name)
		fig8.savefig(fig8Path, format='pdf', dpi=600)

	
################################################################################
## Show Dependence of Num Peaks on wlen, Thresh, Prom
################################################################################
	if plot_fig9_NPeak_vs_ALL:

		fig9, ax9 = plt.subplots(3, 2, figsize=(10, 12))

		fig10, ax10 = plt.subplots(3, 1, figsize=(6, 12))

		tPercArr = np.arange(75, 100)
		pPercArr = np.arange(75, 100)

		minSlope = 20

		for figNo in range(3):

			if figNo == 0:
				data = noMed22
				time = tGrid22
				recNo = 22
			elif figNo == 1:
				data = noMed23
				time = dpTGrid23
				recNo = 23
			else:
				data = noMed24
				time = dpTGrid24
				recNo = 24

			wlenMax = int(len(data)/10.)

			NPeaks = np.zeros((len(pPercArr), len(pPercArr)))

			peakIdx, peakInfo = sig.find_peaks(data, prominence=0, width=0,
				distance=minISI, wlen=wlenMax, rel_height=1)

			proms = peakInfo['prominences']
			hgts = data[peakIdx]

			for ii, tP in enumerate(tPercArr):

				thresh = np.percentile(data, tP)

				goodThresh = hgts > thresh

				for jj, pP in enumerate(pPercArr):

					minProm = np.diff(np.percentile(data, [1, pP]))[0]

					goodProms = proms > minProm

					NPeaks[ii, jj] = np.sum(goodThresh*goodProms)

			hand9 = ax9[figNo, 0].imshow(NPeaks)

			cax = fig9.colorbar(hand9, ax=ax9[figNo, 0], label='No. Peaks')

			ax9[figNo, 0].set_xlabel(r"$thresh$ Percentile")
			ax9[figNo, 0].set_ylabel(r"$prom$ Percentile")

			xticks = np.linspace(0, len(tPercArr)-1, 5).astype(int)
			ax9[figNo, 0].set_xticks(xticks)
			ax9[figNo, 0].set_xticklabels(tPercArr[xticks])

			yticks = np.linspace(0, len(pPercArr)-1, 5).astype(int)
			ax9[figNo, 0].set_yticks(yticks)
			ax9[figNo, 0].set_yticklabels(pPercArr[yticks])

			NPeaksMin = np.zeros((len(pPercArr), len(pPercArr)))

			for jj, pP in enumerate(pPercArr):
				
				minProm = np.diff(np.percentile(data, [1, pP]))[0]

				wlen = minProm/minSlope/dt

				peakIdx, peakInfo = sig.find_peaks(data, prominence=minProm,
					width=0, distance=minISI, wlen=wlen, rel_height=1)

				proms = peakInfo['prominences']
				hgts = data[peakIdx]

				goodProms = proms > minProm

				for ii, tP in enumerate(tPercArr):

					thresh = np.percentile(data, tP)

					goodThresh = hgts > thresh

					NPeaksMin[ii, jj] = np.sum(goodThresh*goodProms)

			hand9b = ax9[figNo, 1].imshow(NPeaksMin)

			caxb = fig9.colorbar(hand9b, ax=ax9[figNo, 1], label='No. Peaks')

			ax9[figNo, 1].set_xlabel(r"$thresh$ Percentile")
			ax9[figNo, 1].set_ylabel(r"$prom$ Percentile")

			xticks = np.linspace(0, len(tPercArr)-1, 5).astype(int)
			ax9[figNo, 1].set_xticks(xticks)
			ax9[figNo, 1].set_xticklabels(tPercArr[xticks])

			yticks = np.linspace(0, len(pPercArr)-1, 5).astype(int)
			ax9[figNo, 1].set_yticks(yticks)
			ax9[figNo, 1].set_yticklabels(pPercArr[yticks])



			h10 = ax10[figNo].imshow(NPeaks/NPeaksMin)

			cax10 = fig9.colorbar(h10, ax=ax10[figNo], label='Ratio No. Peaks')

			ax10[figNo].set_xlabel(r"$thresh$ Percentile")
			ax10[figNo].set_ylabel(r"$prom$ Percentile")

			xticks = np.linspace(0, len(tPercArr)-1, 5).astype(int)
			ax10[figNo].set_xticks(xticks)
			ax10[figNo].set_xticklabels(tPercArr[xticks])

			yticks = np.linspace(0, len(pPercArr)-1, 5).astype(int)
			ax10[figNo].set_yticks(yticks)
			ax10[figNo].set_yticklabels(pPercArr[yticks])



		fig9.tight_layout()

		fig9Name = "NPeaks_vs_ALL.pdf"
		fig9Path = os.path.join(figDir, fig9Name)
		fig9.savefig(fig9Path, format='pdf', dpi=600)

		fig10.tight_layout()

		fig10Name = "NPeaks_vs_ALL_RATIO.pdf"
		fig10Path = os.path.join(figDir, fig10Name)
		fig10.savefig(fig10Path, format='pdf', dpi=600)



################################################################################
## Show Plots!
################################################################################
	plt.show()
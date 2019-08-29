"""
================================================================================
	Red Check Data Illustrator
================================================================================

	Author: Eric Johnson
	Date Created: Thursday, July 11, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	We parsed the red checks data using "Flourakis_RedCheck_Overview" and now 
	want to plot some summary information.


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
		return str(text.strip())
	except AttributeError:
		return str(text)

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
## LET'S VISUALIZE
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

	## Plotting flags
	fig1_plot_restFR = True
	fig1_plot_restPSD = True
	fig1_plot_restAmp = True

################################################################################
## Load csv
################################################################################
	csvfile = "./Misc/RedChecksData_Flourakis_Fits.tsv"

	converters = {
		"dateNo":make_float,
		"date":strip,
		"ZT":make_float,
		"key":make_float,
		"startZT":make_float,
		"cell":strip,
		"protocol":strip,
		"Imin":make_float,
		"Imax":make_float,
		"FRbeg":make_float,
		"FRennd":make_float,
		"Amp":make_float,
		"PSD":make_float}

	df = pd.read_csv(csvfile, sep='\t', header=0, converters=converters)
	# df = df.drop([281, 282])
	# df = df.drop(247)

	"""
	In FR:
	 - (12, 30Hz)
	 - (17, 18Hz)

	In PSD:
	 - Everyone with PSD > -15:
	 	dateNo  date  ZT     key
		233    	35.0   9.6   0
		240    	35.0   9.6   9
		247    	37.0  10.0   0
		254    	38.0   2.2   9
		261    	39.0   1.9  13
		274    	41.0  11.0   0
		281    	41.0  11.0   9
		282    	41.0  11.0  10
		289    	43.0   4.1   0
		290    	43.0   4.1   1
		297    	44.0   2.0   1
		304    	45.0   3.3  10
	"""
	cellAttachedDF = df.loc[df.PSD > -20]
	df = df.drop([233, 240, 247, 254, 261, 268, 274, 281, 282, 289, 290, 297, 304])


	constDF = df.loc[df['protocol'] == "Const"]
	depolDF = df.loc[df['protocol'].isin(["Depol Step", "DepolSteps"])]

	restDF = constDF.loc[constDF['Imin'] == 0]
	# restDF = restDF.loc[restDF['PSD'] < -15]

	cellAttachedDF = cellAttachedDF.loc[cellAttachedDF.protocol == "Const"]

################################################################################
## Plot Figure 1
################################################################################
	if fig1_plot_restFR:

		fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))

		ax1.scatter(restDF['ZT'], restDF['FRbeg'])

		ax1.set_xlabel("ZT")
		ax1.set_ylabel("Firing Rate (Hz)")

		fig1.tight_layout()

		saveName = "Resting_FR_vs_ZT.pdf"
		savePath = os.path.join(figDir, saveName)

		fig1.savefig(savePath, format='pdf', dpi=600)

		fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))

		binWid = 4

		restDF['ZTBin'] = (restDF['ZT']/binWid).astype(int)

		sns.boxplot(x='ZTBin', y='FRbeg', ax=ax2, data=restDF)
		sns.swarmplot(x='ZTBin', y='FRbeg', ax=ax2, data=restDF, color='k',
			alpha=0.5)

		for ZTBin in range(int(24/binWid)):
			N = len(restDF.loc[restDF["ZTBin"] == ZTBin])
			ax2.text(ZTBin, np.median(restDF.loc[restDF["ZTBin"]==ZTBin].FRbeg),
				f"N = {N}", ha='center', va='bottom', fontsize=12)

		ticklabs = [f"{ii*binWid} - {(ii+1)*binWid}"
			for ii in range(int(24/binWid))]
		ax2.set_xticklabels(ticklabs)

		fig2.tight_layout()

		saveName = "Resting_FR_vs_ZT_Boxplot.pdf"
		savePath = os.path.join(figDir, saveName)

		fig2.savefig(savePath, format='pdf', dpi=600)

################################################################################
## Plot Figure 2
################################################################################
	if fig1_plot_restPSD:

		fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))

		ax1.scatter(restDF['ZT'], restDF['PSD'])

		ax1.set_xlabel("ZT")
		ax1.set_ylabel("Post-Spike Depth (mV)")

		fig1.tight_layout()

		saveName = "Resting_PSD_vs_ZT.pdf"
		savePath = os.path.join(figDir, saveName)

		fig1.savefig(savePath, format='pdf', dpi=600)

		fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))

		binWid = 4

		restDF['ZTBin'] = (restDF['ZT']/binWid).astype(int)

		sns.boxplot(x='ZTBin', y='PSD', ax=ax2, data=restDF)
		sns.swarmplot(x='ZTBin', y='PSD', ax=ax2, data=restDF, color='k',
			alpha=0.5)

		for ZTBin in range(int(24/binWid)):
			N = len(restDF.loc[restDF["ZTBin"] == ZTBin])
			ax2.text(ZTBin, np.median(restDF.loc[restDF["ZTBin"]==ZTBin].PSD),
				f"N = {N}", ha='center', va='bottom', fontsize=12)

		ticklabs = [f"{ii*binWid} - {(ii+1)*binWid}"
			for ii in range(int(24/binWid))]
		ax2.set_xticklabels(ticklabs)

		fig2.tight_layout()

		saveName = "Resting_PSD_vs_ZT_Boxplot.pdf"
		savePath = os.path.join(figDir, saveName)

		fig2.savefig(savePath, format='pdf', dpi=600)

################################################################################
## Plot Figure 2
################################################################################
	if fig1_plot_restAmp:

		fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))

		ax1.scatter(restDF['ZT'], restDF['Amp'])

		ax1.set_xlabel("ZT")
		ax1.set_ylabel("Spike Amplitude (mV)")

		fig1.tight_layout()

		saveName = "Resting_Amp_vs_ZT.pdf"
		savePath = os.path.join(figDir, saveName)

		fig1.savefig(savePath, format='pdf', dpi=600)

		fig2, ax2 = plt.subplots(1, 1, figsize=(12, 7))

		binWid = 4

		restDF['ZTBin'] = (restDF['ZT']/binWid).astype(int)

		sns.boxplot(x='ZTBin', y='Amp', ax=ax2, data=restDF)
		sns.swarmplot(x='ZTBin', y='Amp', ax=ax2, data=restDF, color='k',
			alpha=0.5)

		for ZTBin in range(int(24/binWid)):
			N = len(restDF.loc[restDF["ZTBin"] == ZTBin].loc[~np.isnan(restDF['Amp'])])
			print(np.nanmedian(restDF.loc[restDF["ZTBin"]==ZTBin]["Amp"]))
			ax2.text(ZTBin, np.nanmedian(restDF.loc[restDF["ZTBin"]==ZTBin].Amp),
				f"N = {N}", ha='center', va='bottom', fontsize=12)

		ticklabs = [f"{ii*binWid} - {(ii+1)*binWid}"
			for ii in range(int(24/binWid))]
		ax2.set_xticklabels(ticklabs)

		fig2.tight_layout()

		saveName = "Resting_Amp_vs_ZT_Boxplot.pdf"
		savePath = os.path.join(figDir, saveName)

		fig2.savefig(savePath, format='pdf', dpi=600)




	plt.show()



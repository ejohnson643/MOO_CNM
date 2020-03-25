"""
================================================================================
	Flourakis Data Reviewer
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, July 24, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This script will contain precedures to generate a list of all of Matt's
	abf recordings, which we can then parse and assess.

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

import matplotlib.pyplot as plt
import seaborn as sns

import Utility.DataIO_util as DIO

import Objectives.Electrophysiology.ephys_objs_v2 as epo




if __name__ == "__main__":

	plt.close('all')
	sns.set(color_codes=True)

	dataDir = "./Data/TaeHeeData/"
	figDir = "./Figures/TaeHeeOverview/"

	makeNewFigs = False

	fileName = "04/05/2012"
	# fileName = "12/06/2012"
	dataNum = 9

	window = 101
	dt = 0.001
	maxRate = 100
	minISI = int(1./dt/maxRate)
	minSlope = 20.
	pad = 1
	minWidth = 3
	pPromAdj = 0.1

	verbose = 1

	# data = DIO.loadABF(fileName, dirName=dataDir, dirType='Han',
	# 	dataNum=dataNum)

	# dataFolder = DIO.formatDateFolder(DIO.parseDate(fileName), dirName=dataDir,
	# 	dirType='han')

	for yearDir in os.listdir(dataDir):
		for dayDir in os.listdir(os.path.join(dataDir, yearDir)):
			dirPath = os.path.join(dataDir, yearDir, dayDir)
			for fileName in sorted(os.listdir(dirPath)):
				filePath = os.path.join(dirPath, fileName)

				figName = f"RecordingPlot_{''.join(fileName[:-4].split('/'))}.pdf"
				print("Looking at", fileName)

				if not makeNewFigs:
					if figName in os.listdir(dirPath):
						print("Already made figure ", figName)
						continue
				if filePath[-4:] != '.abf':
					print("Not an ABF", filePath)
					continue

				try:
					data = DIO.quickloadABF(filePath)


					fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 8),
						sharex=True)

					cmap = sns.color_palette('hls', data.sweepCount)

					for ii in range(data.sweepCount):
						data.setSweep(ii)

						dt = np.mean(np.diff(data.sweepX))

						spikeIdx, spikeVals = epo.getSpikeIdx(data, dt=dt,
							maxRate=maxRate, minSlope=minSlope, minWidth=minWidth,
							pad=pad, window=window, verbose=verbose)

						ISI = epo.getISI(spikeIdx, dt=dt)

						FR = epo.getFR(ISI)

						# AmpP, AmpCov = epo.getSpikeAmp(spikeIdx, spikeVals, dt=dt,
						# 	returnAll=True, verbose=verbose)

						print(f"\nThe Firing Rate is {FR:.2f}Hz (ISI = {ISI:.2f}s)\n")

						ax1.plot(data.sweepX, data.sweepY, color=cmap[ii],
							label = f"FR = {FR:.2f}Hz")

						ax1.scatter(spikeIdx*dt, spikeVals, color=cmap[ii])

						try:
							ax2.plot(data.sweepX, data.sweepC, color=cmap[ii],
								label = r"$I_{APP} = $" + f"{data.sweepEpochs.levels[2]:.1f} pA")
						except IndexError:
							ax2.plot(data.sweepX, data.sweepC, color=cmap[ii])

					ax1.set_ylabel(data.sweepLabelY, fontsize=18)
					ax1.set_title("Patch Clamp Recording", fontsize=24)
					ax1.legend(fontsize=10)

					ax2.set_xlabel(data.sweepLabelC, fontsize=18)
					ax2.set_ylabel(data.sweepLabelY, fontsize=18)
					ax2.set_title("Applied Current", fontsize=24)
					ax2.legend(fontsize=10)

					fig.tight_layout()

					figName = f"RecordingPlot_{''.join(fileName[:-4].split('/'))}.pdf"
					print("Trying to save", figName)
					fig.savefig(os.path.join(dirPath, figName), format='pdf')
					plt.close('all')
				except:
					plt.close('all')
					print("\n\n\n\n" + 40*"*" + "\n\n" + "COULD NOT MAKE FIGURE")
					print(filePath)
					print("\n" + 40*"*" + "\n\n\n\n")
					# print("\n")
					pass


	plt.show()
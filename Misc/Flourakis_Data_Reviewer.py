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

	figDir = "./Figures/FlourakisDataReview/"

	gen_new_data = True
	check_for_data = True

	plotEveryone = True

	infoDir = "./Runfiles/RedCheck_Overview/"

	infoDict = rfu.getInfo(infoDir, verbose=1)
	
	dataDir = DIO.get_data_dir(dataDir="")

################################################################################
## Procedure Flags
################################################################################

	load_and_plot_txt_files = False
	view_one_at_a_time = True
	make_recording_list = False

################################################################################
## Load txt files from Nov 11, 2010
################################################################################

	if load_and_plot_txt_files:

		txtFileName1 = "2010/nov 2010/11112010/2010-11-11-0003.txt"
		txtFileName2 = "2010/nov 2010/11112010/2010-11-11-0007.txt"

		txtFile1 = []
		with open(os.path.join(dataDir, txtFileName1), 'r') as f:
			headers1 = f.readline().strip().split("\t")
			for line in f:
				txtFile1.append([float(l) for l in line.strip().split("\t")])

		txtFile2 = []
		with open(os.path.join(dataDir, txtFileName2), 'r') as f:
			headers2 = f.readline().strip().split("\t")
			for line in f:
				txtFile2.append([float(l) for l in line.strip().split("\t")])

		txtData1 = np.array(txtFile1)
		txtData2 = np.array(txtFile2)

		fig1, [ax11, ax12] = plt.subplots(2, 1, figsize=(12, 10))

		for ii in range(12):
			ax11.plot(txtData1[:, 0], txtData1[:, ii+1], '-o',
				label=headers1[ii+1])

			ax12.plot(txtData1[:, 0], txtData1[:, ii+13], '-o',
				label=headers1[ii+13])

		ax11.set_title(txtFileName1)
		ax11.legend(fontsize=10)
		ax11.set_ylabel("UNKNOWN... (Current?)")

		ax12.set_xlabel("Time (ms)")
		ax12.legend(fontsize=10)
		ax12.set_ylabel("UNKNOWN... (Voltage?)")

		fig1.tight_layout()

		figName1 = os.path.join("Figures/FlourakisDataReview",
			"2011_11_11_0003_txt.pdf")
		fig1.savefig(figName1, format='pdf', dpi=600)


		fig2, [ax21, ax22] = plt.subplots(2, 1, figsize=(12, 10))

		ax21.plot(txtData2[:, 0], txtData2[:, 1], '-o', label=headers2[1])

		ax22.plot(txtData2[:, 0], txtData2[:, 2], '-o', label=headers2[2])

		ax21.set_title(txtFileName2)
		ax21.legend(fontsize=10)
		ax21.set_ylabel("UNKNOWN... (Current?)")

		ax22.set_xlabel("Time (ms)")
		ax22.legend(fontsize=10)
		ax22.set_ylabel("UNKNOWN... (Voltage?)")

		fig2.tight_layout()

		figName2 = os.path.join("Figures/FlourakisDataReview",
			"2011_11_11_0007_txt.pdf")
		fig2.savefig(figName2, format='pdf', dpi=600)

################################################################################
## Load abfs one at a time...
################################################################################
	
	if view_one_at_a_time:

		year = '2011'
		month = 'April 2011'
		day = '04012011'

		fileDir = os.path.join(dataDir, year, month, day)
		fileList = sorted(os.listdir(fileDir))
		for file in fileList:
			if 'abf' not in file:
				continue

			if "15.abf" not in file:
				continue

			print(file)

			[d, h] = abf.ABF_read(file, datadir=fileDir)

			hour = int(h['lFileStartTime']/3600)
			mins = int((h['lFileStartTime'] - hour*3600)/60)
			secs = int(h['lFileStartTime'] % 60)
			msec = int((h['lFileStartTime'] - int(h['lFileStartTime']))*1000)

			print(f"Recorded at {hour:02d}:{mins:02d}:{secs:02d}:{msec:03d}")
			print(f"File has {h['nADCNumChannels']} channels")
			print(f"There are {h['dataPtsPerChan']} data points per channel")
			print(f"The channels are:")
			for name, unit in zip(h['recChNames'], h['recChUnits']):
				print(f"\t{name}\t({unit})")
			print(f"There are {h['lActualEpisodes']} episodes")

			timebase = abf.GetTimebase(h, 0)

			fig, axes = plt.subplots(h['nADCNumChannels']+1, 1, 
				figsize=(12,10), sharex=True)

			for chan in range(h['nADCNumChannels']):
				ax = axes[chan]

				name = h['recChNames'][chan]
				unit = h['recChUnits'][chan]

				ax.plot(timebase, d[:, chan], '-o', ms=2)
				ax.set_ylabel(f"{name} ({unit})")

			ax = axes[-1]
			for chan in range(h['lActualEpisodes']):
				waveform = abf.GetWaveform(h, chan+1)
				ax.plot(timebase, waveform, '-o', ms=2)

			ax.set_xlabel("Time (ms)")
			ax.set_ylabel("Waveform")

			axes[0].set_title(file)

			fig.tight_layout()

			# saveName = file[:-3] + "pdf"
			# fig.savefig(os.path.join(h['sDirectory'], 'Figures', saveName),
			# 	format='pdf', dpi=600)

			foo = input("")

			if foo:
				break

			plt.close(fig)

################################################################################
## Make CSV for Recordings
################################################################################
	if make_recording_list:

		appendToFile = True
		csvName = "Misc/AllRecordings.tsv"

		years = ['2010', '2011', '2012', '2013', '2014']
		months = [f'{d:02d}' for d in range(1, 13)]
		days = [f"{d:02d}" for d in range(1, 32)]

		dateStrs = [m + d + y for y in years for m in months for d in days]

		header = ["Date", "DateStr", "DateNo", "RecTime", "ZT", "RecNo",
			"NrnNo", "Geno", "Prot", "IApp", "NEps", "FR", "Amp", "PSD", "RI",
			"tau", "C", "Notes", "Notebook", "Page", "RedCheck"]
		topLine = "\t".join(header) + "\n"

		if appendToFile:
			try:
				with open(csvName, 'r') as f:
					_ = f.readline()
					lines = []
					for line in f:
						lines.append(line.strip().split("\t"))
				
				dateRecNos = [(int(line[2]), int(line[5])) for line in lines]

			except:
				with open(csvName, 'w') as f:
					f.write(topLine)
				dateRecNos = []

		lines = []
		dateNo = 0
		for dateStr in dateStrs:

			try:
				dateTpl = datetime.datetime(int(dateStr[4:]), int(dateStr[:2]),
					int(dateStr[2:4]))

				datePath = DIO.find_date_folder(dateTpl)

			except ValueError:
				continue

			# if dateNo != 7:
			# 	dateNo += 1
			# 	continue

			fullDate = datetime.datetime.ctime(dateTpl)

			print(f"\n\nLoading info from {fullDate}")

			breakEverything = False

			recFiles = os.listdir(datePath)
			recFiles = sorted([f for f in recFiles if f[-3:] == 'abf'])

			print(f"There are {len(recFiles)} abf files!\n")

			for file in recFiles:

				if "Concatenate" in file:
					continue

				recNo = int(file[-8:-4])

				if appendToFile:
					if (dateNo, recNo) in dateRecNos:
						continue

				[data, hdr] = abf.ABF_read(file, datadir=datePath)

				if hdr['fFileSignature'] != "ABF ":
					print("SKIPPING BECAUSE FILE VERSION NUMBER")
					continue

				recTime = hdr['lFileStartTime']

				ZT = NrnNo = Geno = Notebook = Page = RedCheck = ""

				protocol = epu.getExpProtocol(hdr)

				if protocol == -1:
					protStr = "Unknown"
					if hdr['nADCNumChannels'] == 1:
						if 'pA' == hdr['recChUnits'][0]:

							print(f"{dateNo}:{recNo}:\tVoltage Clamp Protocol")

							protStr = "VClamp"

					IApp = FR = Amp = PSD = RI = Tau = C = Notes = ""

				elif protocol == EPHYS_PROT_REST:
					protStr = "Rest"

					print(f"{dateNo}:{recNo}:\tRest Protocol")

					if data.shape[2] > 1:
						subData = data[:, :, 0]
					else:
						subData = deepcopy(data)

					dataFeat = epu.getRestFeatures(subData, hdr, infoDict, {},
						key=recNo, verbose=3)

					IApp = "0.0"
					FR = 1./dataFeat['ISI']['rest'][recNo]
					Amp = dataFeat['Amp']['rest'][recNo]
					PSD = dataFeat['PSD']['rest'][recNo]

					FR = f"{FR:5.2f}"
					Amp = f"{Amp:5.2f}"
					PSD = f"{PSD:5.2f}"
					RI = Tau = C = Notes = ""

				elif protocol == EPHYS_PROT_CONSTHOLD:
					protStr = "Const"

					print(f"{dateNo}:{recNo}:\tConstant Hold Protocol")

					print(f"Len of data: {data.shape}")

					# continue

					if len(data) < 10000:
						dataFeat = epu.getConstHoldFeatures(data[:30000], hdr, 
							infoDict, {}, key=recNo, verbose=2)

						waveform = abf.GetWaveform(hdr, hdr['lActualEpisodes'])

						IApp = waveform[0, hdr['nActiveDACChannel']]*1000
						FR = 1./dataFeat['ISI']['hold'][recNo]
						Amp = dataFeat['Amp']['hold'][recNo]
						PSD = dataFeat['PSD']['hold'][recNo]

						IApp = f"{IApp:5.2f}"
						FR = f"{FR:5.2f}"
						Amp = f"{Amp:5.2f}"
						PSD = f"{PSD:5.2f}"
						RI = Tau = C = Notes = ""

					else:
						IApp = PSD = FR = Amp = RI = Tau = C = Notes = ""
						Notes = "Const hold, data too long."

				elif protocol == EPHYS_PROT_DEPOLSTEP:
					protStr = "DplStep"

					print(f"{dateNo}:{recNo}:\tDepolarization Step Protocol")

					dataFeat = epu.getDepolFeatures(data, hdr, infoDict, {},
						key=recNo, verbose=3)

					waveform = abf.GetWaveform(hdr, hdr['lActualEpisodes'])

					IApp = [waveform[0, hdr['nActiveDACChannel']]*1000,
						waveform[:, hdr['nActiveDACChannel']].max()*1000]
					FR = [1./ISI for ISI in dataFeat['ISI']['depol'][recNo]]
					Amp = dataFeat['Amp']['depol'][recNo]
					PSD = dataFeat['PSD']['depol'][recNo]

					IApp = ", ".join([f"{ia:5.2f}" for ia in IApp])
					FR = ", ".join([f"{fr:5.2f}" for fr in FR])
					Amp = f"{Amp:5.2f}"
					PSD = f"{PSD:5.2f}"

					RI = Tau = C = Notes = ""

				elif protocol == EPHYS_PROT_HYPERPOLSTEP:
					protStr = "HplStep"

					print(f"{dateNo}:{recNo}:\tHyperpolarization Step Protocol")

					dataFeat = epu.getHyperpolFeatures(data, hdr, infoDict,
						{}, key=recNo, verbose=2)

					waveform = abf.GetWaveform(hdr, hdr['lActualEpisodes'])

					IApp = [waveform[0, hdr['nActiveDACChannel']]*1000,
						waveform[:, hdr['nActiveDACChannel']].min()*1000]

					IApp = ", ".join([f"{ia:5.2f}" for ia in IApp])

					PSD = dataFeat['PSD']['hyperpol'][recNo]
					PSD = f"{PSD:5.2f}"

					FR = Amp = RI = Tau = C = Notes = ""


				elif protocol == EPHYS_PROT_DEPOLSTEPS:
					protStr = "DplSteps"

					print(f"{dateNo}:{recNo}:\tDepolarization Steps Protocol")

					IApp = PSD = FR = Amp = RI = Tau = C = Notes = ""

				elif protocol == EPHYS_PROT_HYPERPOLSTEPS:
					protStr = "HplSteps"

					print(f"{dateNo}:{recNo}:\tHyperpol Steps Protocol")

					if hdr['lActualEpisodes'] <= 3:
						continue

					dataFeat = epu.getHyperpolStepsFeatures(data, hdr, infoDict,
						{}, key=recNo, verbose=2)

					IApp = [hdr['fEpochInitLevel'][hdr['nActiveDACChannel']][0]]
					IApp = IApp +list(dataFeat['PSD']['hyperpol'][recNo].keys())

					IApp = ", ".join([f"{ia*1000:5.2f}" for ia in IApp])

					PSD = list(dataFeat['PSD']['hyperpol'][recNo].values())

					PSD = ", ".join([f"{psd:5.2f}" for psd in PSD])

					RI = dataFeat['RI'][recNo]
					Tau = dataFeat['tau'][recNo]
					C = dataFeat['C'][recNo]

					RI = f"{RI:5.2f}"
					Tau = f"{Tau:5.2f}"
					C = f"{C:5.2f}"

					FR = Amp = Notes = ""

				else:
					raise ValueError(f"What protocol is {protocol}???")


				line = ("\t".join([fullDate, dateStr, f"{dateNo:d}", 
					f"{recTime:6.5f}", ZT, f"{recNo:d}", NrnNo, Geno, protStr,
					IApp, f"{hdr['lActualEpisodes']:d}", FR, Amp, PSD, RI, 
					Tau, C, Notes, Notebook, Page, RedCheck]) + "\n")
				lines.append(line )

				if appendToFile:
					with open(csvName, "a") as f:
						f.write(line)


				recNo += 1

			if breakEverything:
				break

			dateNo += 1

			# if dateNo >= 7:
			# 	break

		readFmt = "w"
		if appendToFile:
			readFmt = "a"

		if not appendToFile:
			with open(csvName, "w") as f:
				f.write(topLine)
				for line in lines:
					f.write(line)


################################################################################
## SHOW PLOTS
################################################################################
	plt.show()

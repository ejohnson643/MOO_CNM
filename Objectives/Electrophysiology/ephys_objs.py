"""
================================================================================
	Electrophysiology Feature Extraction Functions
================================================================================

	Author: Eric Johnson
	Date Created: Thursday, March 7, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains a module of functions that will extract various features
	from electrophysiology data.

	In particular, this function will contain methods for extracting:
	 - Spike Location
	 - ISI
	 	- Entire Recording
	 	- First third
	 	- Last third
	 	- First and last third
	 - Spike Amplitude
	 	- Average
	 	- Exponential Fit
	 - Post-Spike Depth (Membrane Potential) 
	 	- Average
	 	- Exponential Fit
	 - F-I slope
	 - Input Resistance
	 	- Linear Fit to Slope, with Cov
	 	- RI, tau, C with err

================================================================================
================================================================================
"""
from collections import Counter
from copy import deepcopy
import numpy as np
import scipy.interpolate as intrp
from scipy.optimize import curve_fit
import scipy.signal as sig
import scipy.stats as st
from skimage import filters
import warnings

import Objectives.Electrophysiology.ephys_util as epu

import Utility.ABF_util as abf
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
################################################################################
##
##		Electrophysiology Feature Extraction Methods
##
################################################################################
################################################################################

################################################################################
##	Get Index Locations of Action Potentials
################################################################################
def getSpikeIdx(data, dt=0.001, maxRate=100., exact=True, pad=1, 
	minSlope=20., window=100, thresh=None, pThresh=90., minProm=None, 
	pProm=(1, 90), maxMedDiff=5., verbose=0, **kwds):

	############################################################################
	##	Check Inputs, Keyword Arguments, Setup Median Subtraction
	############################################################################
	if True:
		## Check the verbosity keyword
		verbose = utl.force_pos_int(verbose, name='spike.verbose', 
			zero_ok=True, verbose=verbose)

		## Check the type and shape of the data
		data = utl.force_float_arr(data, name='spike.data',
			verbose=verbose).squeeze()

		err_str = "Input argument 'data' must be 1D array."
		err_str += f" (data.ndim = {data.ndim})"
		assert data.ndim == 1, err_str

		## Check maxRate, set minimum ISI
		maxRate = utl.force_pos_float(maxRate, name='spike.maxRate',
			verbose=verbose)
		minISI = int(1./dt/maxRate)

		## Check that we're looking for exact or non-exact spike locations
		err_str = "Keyword argument 'exact' must be a boolean."
		assert isinstance(exact, bool), err_str

		if verbose >= 2:
			if exact:
				print(f"Getting Exact (non-integer) Spike Locations!")
			else:
				print(f"Getting Spike Locations")

		## Set the minimum allowable spike slope
		minSlope = utl.force_pos_float(minSlope, name='epo.getSpikeIdx.wlen',
			zero_ok=True, verbose=verbose)

		## Set spline fitting pad parameter
		pad = utl.force_pos_int(pad, name='spike.pad', verbose=verbose)

		## Check that 'window' is an integer and is odd
		window = utl.force_pos_int(window, name='epo.getSpikeIdx.window',
			verbose=verbose)
		window = window + 1 if ((window % 2) == 0) else window

		pThresh = utl.force_pos_float(pThresh,
			name='epo.getSpikeIdx.pThresh', verbose=verbose)
		errStr = "Keyword argument 'pThresh' must be a percent in (0, 100)"
		assert pThresh < 100, errStr

		maxMedDiff = utl.force_pos_float(maxMedDiff,
			name='epo.getSpikeIdx.maxMedDiff', verbose=verbose)

	############################################################################
	##	Get median subtracted data.
	############################################################################

		noMedData = data - epu.getRollPerc(data, window=window, perc=50,
			verbose=verbose)

		## Check threshold, if none, set based on data
		if thresh is not None:
			thresh = utl.force_float(thresh, name='epo.getSpikeIdx.thresh',
				verbose=verbose)
			thresh -= np.median(data)

			dCounts = Counter(noMedData.ravel())
			dVals = np.msort(list(dCounts.keys()))
			dCDF = np.cumsum(np.asarray([dCounts[ii] for ii in dVals]))
			dCDF = dCDF/dCDF[-1]
			pThresh = dCDF[(dVals <= thresh).nonzero()]
			if len(pThresh) > 0:
				pThresh = pThresh[-1]*100.
			else:
				pThresh = 100.

		else:
			thresh = np.percentile(noMedData, pThresh)

		## Check minimum prominence, if none, set based on data
		if minProm is not None:
			minProm = utl.force_pos_float(minProm, 
				name='epo.getSpikeIdx.minProm', zero_ok=True, verbose=verbose)
		else:
			errStr = "pProm must be (pMin, pMax) tuple!"
			assert (len(pProm) == 2) and isinstance(pProm, tuple), errStr
			assert np.all([utl.force_pos_float(p, name='epo.getSpikeIdx.pProm',
				verbose=verbose) for p in pProm]), errStr
			assert np.all([p < 100 for p in pProm]), errStr
			pMin, pMax = pProm
			minProm = np.diff(np.percentile(noMedData, [pProm[0], pProm[1]]))[0]

	############################################################################
	##	Find Minimal Allowed wlen
	############################################################################
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")

		if verbose >= 3:
			print("Finding minimum allowable wlen")
			print(f"Starting with T =\t{thresh:.4g}mV\nP =\t{minProm:.4g}mV")


		## Find the maximum possible number of spikes with the given threshold
		## and prominence
		wLen_Max = len(noMedData)

		tLevel = pThresh
		while True:

			peakIdx, _ = sig.find_peaks(noMedData, height=thresh,
				distance=minISI, prominence=minProm, wlen=wLen_Max)

			peakVals = noMedData[peakIdx]

			if len(peakVals) < 2:
				break

			if len(np.unique(peakVals)) < 2:
				break

			oThr = filters.threshold_otsu(peakVals)

			lowMed = np.median(peakVals[peakVals <= oThr])
			highMed = np.median(peakVals[peakVals > oThr])

			if verbose >= 4:
				print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals<=oThr)} on left "+
					f"{sum(peakVals>oThr)} on right)")
				print(f"Low Median {lowMed:.4g}, High Median {highMed:.4g}")

			if (highMed - lowMed) < maxMedDiff:
				break

			tLevel += (100 - tLevel)/2.
			thresh = np.percentile(noMedData, tLevel)

			if verbose >= 4:
				print(f"The threshold level is {tLevel} ({thresh:.4g})")

		NSp_Max = len(peakIdx)
		if verbose >= 3:
			print(f"\n{NSp_Max} Spikes Found!")

		if NSp_Max == 0:
			return [], []

		## Find the number of spikes found when wlen corresponds to the minimum
		## allowed slope of an AP
		wLen_MinSlope = int(np.ceil(minProm/minSlope/dt))
		if wLen_MinSlope % 2 == 0:
			wLen_MinSlope += 1

		if verbose >= 3:
			print(f"wlen for min slope = {wLen_MinSlope}")

		tLevel = pThresh
		thresh = np.percentile(noMedData, pThresh)
		while True:

			peakIdx, _ = sig.find_peaks(noMedData, height=thresh,
				distance=minISI, prominence=minProm, wlen=wLen_MinSlope)

			peakVals = noMedData[peakIdx]

			if len(peakVals) < 2:
				break

			if len(np.unique(peakVals)) < 2:
				break

			oThr = filters.threshold_otsu(peakVals)

			lowMed = np.median(peakVals[peakVals <= oThr])
			highMed = np.median(peakVals[peakVals > oThr])

			if verbose >= 4:
				print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals<=oThr)} on left "+
					f"{sum(peakVals>oThr)} on right)")
				print(f"Low Median {lowMed:.4g}, High Median {highMed:.4g}")

			if (highMed - lowMed) < maxMedDiff:
				break

			tLevel += (100 - tLevel)/2.
			thresh = np.percentile(noMedData, tLevel)

			if verbose >= 4:
				print(f"The threshold level is {tLevel} ({thresh:.4g})")

		NSp_MinSlope = len(peakIdx)
		if verbose >= 3:
			print(f"\n{NSp_MinSlope} Spikes Found!")


		## If the min slope and max wlen yield the same number of spikes,
		## search downwards to find the minimum wlen that gives the same number.
		if NSp_Max == NSp_MinSlope:
			if verbose >= 3:
				print(f"wLen_MinSlope yields same no. of spikes as wLen_Max!")

			wLen = wLen_MinSlope
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore")
				peakWids = sig.peak_widths(noMedData, peakIdx,
					wlen=wLen_MinSlope, rel_height=1.)

			leftWids, rightWids = peakIdx-peakWids[2], peakWids[3]-peakIdx

			NSpLeft = np.sum(leftWids <= (wLen-1)/2.)
			NSpRight = np.sum(rightWids <= (wLen-1)/2.)

			while np.minimum(NSpLeft, NSpRight) == NSp_Max:

				wLen -= 2

				NSpLeft = np.sum(leftWids <= (wLen-1)/2.)
				NSpRight = np.sum(rightWids <= (wLen-1)/2.)

			wLen += 2

			if verbose >= 3:
				print(f"Determined that the minimal spike width is {wLen}")

		## Otherwise, look at all smaller wlens and find the "most stable" 
		## number of spikes, i.e. the number that occurs most often
		else:
			if verbose >= 3:
				print(f"wLen_MinSlope yields a different number of spikes...")

			wLenArr = np.arange(3, wLen_MinSlope+1, 2)
			NSpArr = np.zeros_like(wLenArr)

			for ii, wLen in enumerate(wLenArr):

				if ((ii % 10) == 0) and (verbose >= 4):
					print(f"{ii}: wLen = {wLen}")

				tLevel = pThresh
				thresh = np.percentile(noMedData, pThresh)
				while True:

					peakIdx, _ = sig.find_peaks(noMedData, height=thresh,
						distance=minISI, prominence=minProm, wlen=wLen)

					peakVals = noMedData[peakIdx]

					if len(peakVals) < 2:
						break

					if len(np.unique(peakVals)) < 2:
						break
						
					oThr = filters.threshold_otsu(peakVals)

					lowMed = np.median(peakVals[peakVals <= oThr])
					highMed = np.median(peakVals[peakVals > oThr])

					if verbose >= 5:
						print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals<=oThr)} " +
							f"on left {sum(peakVals>oThr)} on right)")
						print(f"Low Median {lowMed:.4g}, High Median " +
							f"{highMed:.4g}")

					if (highMed - lowMed) < maxMedDiff:
						break

					tLevel += (100 - tLevel)/2.
					thresh = np.percentile(noMedData, tLevel)

					if verbose >= 5:
						print(f"The threshold level is {tLevel} ({thresh:.4g})")

				NSpArr[ii] = len(peakIdx)

			NSp_Mode = st.mode(NSpArr).mode[0]

			wLen = wLenArr[(NSpArr == NSp_Mode).nonzero()[0][0]]

			if verbose >= 3:
				print(f"NSp_Mode = {NSp_Mode} giving wLen = {wLen}")

		## Using the optimized parameters now, find the peakIdx
		tLevel = pThresh
		thresh = np.percentile(noMedData, pThresh)
		while True:

			peakIdx, _ = sig.find_peaks(noMedData, height=thresh,
				distance=minISI, prominence=minProm, wlen=wLen)

			peakVals = noMedData[peakIdx]

			if len(peakVals) < 2:
				break

			if len(np.unique(peakVals)) < 2:
				break
			oThr = filters.threshold_otsu(peakVals)

			lowMed = np.median(peakVals[peakVals <= oThr])
			highMed = np.median(peakVals[peakVals > oThr])

			if verbose >= 4:
				print(f"Otsu Thr: {oThr:.4g} ({sum(peakVals<=oThr)} on left "+
					f"{sum(peakVals>oThr)} on right)")
				print(f"Low Median {lowMed:.4g}, High Median {highMed:.4g}")

			if (highMed - lowMed) < maxMedDiff:
				break

			tLevel += (100 - tLevel)/2.
			thresh = np.percentile(noMedData, tLevel)

			if verbose >= 4:
				print(f"The threshold level is {tLevel} ({thresh:.4g})")

		NPeaks = len(peakIdx)

		if verbose >= 2:
			print(f"{NPeaks} Peaks Found...")
		if verbose >= 3:
			print(f"Thresh = {thresh:.4g}\tProm = {minProm:.4g}\twLen = {wLen}")

	############################################################################
	##	Get InExact Peak Locations, If Requested
	############################################################################
	if not exact:
		spikeIdx = np.array(peakIdx).astype(float).squeeze()
		spikeVals = np.array(data[peakIdx]).astype(float).squeeze()

		## If spikeIdx is a single number, make it an array
		try:
			_ = len(spikeIdx)
		except:
			spikeIdx = np.array([spikeIdx]).astype(int)

		try:
			_ = len(spikeVals)
		except:
			spikeVals = np.array([spikeVals]).astype(int)

		## Return indices and values
		return spikeIdx, spikeVals

	############################################################################
	##	Get Exact Peak Locations, If Requested
	############################################################################
	else:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			widths, _, wLeft, wRight = sig.peak_widths(data, peakIdx,
				rel_height=0.5)
			wLeft = np.floor(wLeft).astype(int)
			wRight = np.ceil(wRight).astype(int)
		
		spikeIdx, spikeVals = [], []

		if verbose >= 4:
			print(f"Itr:\tWL - PEAK - WR - Width")

		for itr, (wl, wr) in enumerate(zip(wLeft, wRight)):

			rightSize = ((wr-wl) > 3) and ((wr-wl) <= int(minISI/2.))
			
			counter, maxCounter = 0, 10
			while not rightSize:

				if verbose >= 4:
					print(f"{itr}:\tAdjust {wl} - {peakIdx[itr]} - "+
						f"{wr} - {wr-wl}")

				## If the width is too small, pad it to make it larger
				if (wr - wl) <= 3:
					if itr > 0:
						wl = max(wl-pad, min(wRight[itr-1], wRight[itr]))
					else:
						wl = max(wl-pad, 0)

					if itr < NPeaks-1:
						wr = min(wr+pad, max(wLeft[itr+1], wLeft[itr]))
					else:
						wr = min(wr+pad, len(data)-1)

				## If the width is too large, move halfway closer in to the peak
				elif (wr - wl) > int(minISI/2.):
					if (peakIdx[itr] - wl) > int(minISI/4.):
						wl += int((peakIdx[itr] - wl)/2.)
					if (wr - peakIdx[itr]) > int(minISI/4.):
						wr -= int((wr - peakIdx[itr])/2.)

				## Check if right size
				rightSize = ((wr-wl) > 3) and ((wr-wl) <= int(minISI/2.))

				## Increment the counter and only try so hard
				counter += 1
				if counter > maxCounter:
					if verbose >= 3:
						print("[epo.GetSpikeIdx]: WARNING: Could not find "+
							f"optimal spike width in {maxCounter} attempts...")
					break

			## Grid for the data
			grid = np.arange(wl, wr+.1).astype(int)
			## Grid on which to evaluate the spline
			finegrid = np.linspace(wl, wr, 1001)

			## Fit the spline to the data on the coarse grid
			splfit = intrp.splrep(grid, data[grid], k=3) ## CUBIC
			## Calculate the derivative
			dsplfit = intrp.splder(splfit)
			## Fit the derivative to the fine grid
			derfit = intrp.splrep(finegrid, intrp.splev(finegrid, dsplfit), k=3)
			## Find the location of the zeros of the derivative
			peakLoc = intrp.sproot(derfit, mest=len(finegrid))


			## If there are no peaks, skip this AP
			if len(peakLoc) == 0:
				continue

			## Get the peak height
			peakVal = intrp.splev(peakLoc, splfit)

			## Assume the AP is at the locaiton of the largest root
			spikeIdx.append(peakLoc[np.argmax(peakVal)])
			spikeVals.append(np.max(peakVal))

		spikeIdx = np.array(spikeIdx).astype(float).squeeze()
		spikeVals = np.array(spikeVals).astype(float).squeeze()

		## If spikeIdx is a single number, make it an array
		try:
			_ = len(spikeIdx)
		except:
			spikeIdx = np.array([spikeIdx]).astype(int)

		try:
			_ = len(spikeVals)
		except:
			spikeVals = np.array([spikeVals]).astype(int)

		## Return indices and values
		return spikeIdx, spikeVals


################################################################################
##	Get Inter-Spike Interval
################################################################################
def getISI(spikeIdx, dt=0.001, minRate=0., NSpikes=1, **kwds):

	## Check keywords
	dt = utl.force_pos_float(dt, name='getISI.dt')

	minRate = utl.force_pos_float(minRate, name='getISI.minRate', zero_ok=True)

	NSpikes = utl.force_pos_int(NSpikes, name='getISI.NSpikes')

	## If the minimum number of spikes are present
	if len(spikeIdx) > NSpikes:
		ISI = np.median(np.diff(spikeIdx))*dt
		if 1./ISI >= minRate:
			return ISI

	return np.inf


################################################################################
##	Get AP Amplitude
################################################################################
def getSpikeAmp(spikeIdx, spikeVals, dt=0.001, NSpikes=1, fit='exp', covTol=0.4,
	returnAll=False, verbose=0, **kwds):

	############################################################################
	##	Check Inputs, Keyword Arguments
	############################################################################
	if True:

		verbose = utl.force_pos_int(verbose, name="getSpikeAmp.verbose",
			zero_ok=True)

		## Check type, shape of spikeIdx, spikeVals
		spikeIdx = utl.force_float_arr(spikeIdx, name='getSpikeAmp.spikeIdx',
			verbose=verbose).squeeze()
		spikeVals = utl.force_float_arr(spikeVals, name='getSpikeAmp.spikeVals',
			verbose=verbose).squeeze()

		if spikeIdx.shape == ():
			spikeIdx = spikeIdx.reshape(-1)

		if spikeVals.shape == ():
			spikeVals = spikeVals.reshape(-1)

		if len(spikeIdx) == 0:
			return np.NaN

		## Check that spikeIdx and spikeVals are 1D arrays
		err_str = "(ephys_objs.getSpikeAmp): Expected input argument 'spikeIdx'"
		err_str += f"to have 1 dimension, got shape={spikeIdx.shape}"
		assert spikeIdx.ndim == 1, err_str

		err_str ="(ephys_objs.getSpikeAmp): Expected input argument 'spikeVals'"
		err_str += f"to have 1 dimension, got shape={spikeVals.shape}"
		assert spikeVals.ndim == 1, err_str

		## Check that spikeIdx and spikeVals match
		err_str = "(ephys_obj.getSpikeAmp): Dimension mismatch between "
		err_str += f"'spikeIdx' and 'spikeVals' ({spikeIdx.shape} != " 
		err_str += f"{spikeVals.shape})."
		assert len(spikeIdx) == len(spikeVals), err_str

		## Check dt
		dt = utl.force_pos_float(dt, name='getSpikeAmp.dt', verbose=verbose)

		## Check NSpikes
		NSpikes = utl.force_pos_int(NSpikes, name='getSpikeAmp.NSpikes',
			verbose=verbose)

		## Check that there *are* spikes
		if len(spikeIdx) <= NSpikes:
			return np.NaN

		err_str = "(ephys_objs.getSpikeAmp): Keyword argument 'fit' must be a "
		err_str += "string!" 
		assert isinstance(fit, str), err_str

		allowed_fits = ['exp', 'mean']
		err_str = "(ephys_objs.getSpikeAmp): Invalid value for keyword "
		err_str += "argument 'fit':{fit}.  Allowed values: "
		err_str += ", ".join([f"'{f}'" for f in allowed_fits])
		assert fit in allowed_fits, err_str

		## Check covTol, which is the minimum tolerance for the coeff of var.
		## in the exponential fit.
		covTol = utl.force_pos_float(covTol, name='getSpikeAmp.covTol',
			verbose=verbose)

		## Check returnAll is bool
		err_str = "(ephys_objs.getSpikeAmp): Keyword argument 'returnAll' must "
		err_str += "be a boolean!"
		assert isinstance(returnAll, bool), err_str

	############################################################################
	##	Compute AP Amplitude
	############################################################################
	if fit == 'exp':
		spikeT = spikeIdx*dt

		AmpP, AmpCov = epu.fitExp(spikeVals, times=spikeT, returnAll=True)
		AmpP[AmpP == 0] = 1.e-16

		CoV = np.sqrt(AmpCov)/np.abs(AmpP)

		if CoV[1] < covTol:
			if verbose > 1:
				print(f"(epu.getSpikeAmp): Exponential fit was good! "+
					f"(CoV <= {covTol:.3g})")

			if returnAll:
				return AmpP, AmpCov
			else:
				return AmpP

	if verbose > 1:
		print("(epu.getSpikeAmp): Using mean fit!")

	return np.mean(spikeVals)


################################################################################
##	Get Post-Spike Depth
################################################################################
def getPSD(data, spikeIdx, dt=0.001, window=None, perc=None, covTol=1.,
	fit='exp', verbose=0, returnAll=False, **kwds):

	############################################################################
	##	Check Inputs, Keyword Arguments
	############################################################################
	if True:

		verbose = utl.force_pos_int(verbose, name="getPSD.verbose",
			zero_ok=True)

		## Check type of data
		data = utl.force_float_arr(data, name='getPSD.data',
			verbose=verbose).squeeze()

		## Check that data is 1D
		err_str = "(ephys_objs.getPSD): Expected input argument 'data'"
		err_str += f"to have 1 dimension, got shape={data.shape}"
		assert data.ndim == 1, err_str

		## Check type of spikeIdx
		spikeIdx = utl.force_float_arr(spikeIdx, name='getPSD.spikeIdx',
			verbose=verbose).squeeze()

		if spikeIdx.shape == ():
			spikeIdx = spikeIdx.reshape(-1)

		## Check that spikeIdx is 1D
		if len(spikeIdx) > 0:
			err_str = "(ephys_objs.getPSD): Expected input argument 'spikeIdx'"
			err_str += f"to have 1 dimension, got shape={spikeIdx.shape}"
			assert spikeIdx.ndim == 1, err_str

		## Check dt
		dt = utl.force_pos_float(dt, name='getPSD.dt', verbose=verbose)

		## Check window... 
		## If window is given, check.
		if window is not None:
			window = utl.force_pos_int(window, name='getPSD.window',
				verbose=verbose)
		else: ## Else, use data to infer window size.
			if len(spikeIdx) > 1: ## If there are spikes, use avg dist bt APs
				window = int(1.5*np.mean(np.diff(spikeIdx)))
			else: ## Else use 100th of data or 100 time points (pick larger)
				window = max(100, int(len(data)/100.))

		## Force window to be odd.
		if (window % 2) != 0:
			window += 1

		## Check perc...
		if perc is None:
			perc = 10.
		else:
			perc = utl.force_pos_float(perc, label='getPSD.perc',
				verbose=verbose)
			if perc < 1:
				perc *= 100

			err_str = "(ephys_objs.getPSD): Keyword argument 'perc' must be"
			err_str += "less than 100 (correspond to a percentage)."
			assert perc < 100, err_str

		## Check covTol, which is the minimum tolerance for the coeff of var.
		## in the exponential fit.
		covTol = utl.force_pos_float(covTol, name='getSpikeAmp.covTol',
			verbose=verbose)

		## Check fit
		err_str = "(ephys_objs.getPSD): Keyword argument 'fit' must be a "
		err_str += "string!" 
		assert isinstance(fit, str), err_str

		## Check that fit is allowed
		allowed_fits = ['exp', 'mean']
		err_str = "(ephys_objs.getPSD): Invalid value for keyword "
		err_str += "argument 'fit':{fit}.  Allowed values: "
		err_str += ", ".join([f"'{f}'" for f in allowed_fits])
		assert fit in allowed_fits, err_str

		## Check returnAll is bool
		err_str = "(ephys_objs.getSpikeAmp): Keyword argument 'returnAll' must "
		err_str += "be a boolean!"
		assert isinstance(returnAll, bool), err_str

	############################################################################
	##	Compute Post-Spike Depth
	############################################################################
	
	## Compute order...
	order = int(window*perc/100.)

	## Compute windowed percentile (usually 10th percentile)
	windArr = np.ones(window+1)
	PSDArr = sig.order_filter(data, windArr, order)

	## Get windowed time array (taking half of window from each end)
	pad = int(window/2.)
	times = np.arange(len(data))[pad:-pad]*dt

	if fit == 'exp':
		P, cov = epu.fitExp(PSDArr, times=times, returnAll=True)
		P[P == 0] = 1.e-16

		CoV = np.sqrt(cov)/np.abs(P)

		if np.all(CoV <= covTol):

			if verbose > 1:
				print(f"(epo.getPSD): Exponential fit was good! " +
					f"(CoV <= {covTol:.3g})")

			if returnAll:
				return P, cov
			else:
				return P

		else:
			if verbose > 2:
				print(f"(epo.getPSD): Exponential fit was not good!" +
					f"(CoV = {CoV})")

	if verbose > 1:
		print("(epu.getPSD): Using mean fit!")

	return np.mean(PSDArr)


################################################################################
##	Get F-I Slope
################################################################################
def getFISlope(data, objDict, IArr, featDict=None, dt=0.001, **kwds):

	## Check that the objDict is the 'objectives' field from an infoDict
	err_str = "Input argument 'epo.getFISlope.objDict' must be a dictionary!"
	assert isinstance(objDict, dict), err_str

	try:
		verbose = objDict['FI']['verbose']
	except:
		verbose = 0
	objDict['FI']['verbose'] = verbose

	verbose = utl.force_pos_int(verbose, name='epo.getFISlope.verbose',
		zero_ok=True)

	## Check dt
	dt = utl.force_pos_float(dt, name='epo.getFISlope.dt', verbose=verbose)

	## If a feature dictionary is provided, extract the F-I slope from that.
	## (This method assumes that protocol==DEPOLSTEPS!)
	if (featDict is not None) and (isinstance(featDict, dict)):
		if verbose > 1:
			print("Extracting F-I Slope from Feature Dictionary!")
			print("WARNING: Will not use weighted least-squares because not "+
				"recomputing spike indices!")
		return getFISlope_from_Dict(featDict, dt=dt, **objDict['FI'])

	if verbose > 1:
		print("Calculating F-I Slope from Data!")

	## Check that data is the right type and shape
	err_str = "Input argument 'data' must be 2D array of data."
	data = utl.force_float_arr(data, name='epo.getFISlope.data',
		verbose=verbose).squeeze()
	assert data.ndim == 2, err_str

	## Check that IArr is the right type
	IArr = utl.force_float_arr(IArr, name='epo.getFISlope.IArr',
		verbose=verbose).squeeze()

	## Iterate through the episodes... Assumes dpData is N x NEps shape
	FArr, IQRArr = [], []
	for ii, D in enumerate(data.T):
		if verbose > 2:
			print(f"Extracting ISI from episode {ii}!")

		ISIInfo = objDict["ISI"]

		spikeIdx, spikeVals = getSpikeIdx(D, dt=dt, **objDict['Spikes'])

		if ISIInfo['depol'] in ['thirds', 'lastthird']:
			bounds = np.linspace(0, len(D), 4).astype(int)

			err = []
			iqr = []

			first = spikeIdx[spikeIdx < bounds[1]]
			err.append(getISI(first, dt=dt, **ISIInfo))
			if len(first) > 5:
				iqr.append(list(np.percentile(1./(np.diff(first)*dt), [25,75])))
			else:
				iqr.append([np.min(1./(np.diff(first)*dt)),
					np.max(1./(np.diff(first)*dt))])


			last = spikeIdx[spikeIdx >= bounds[2]]
			err.append(getISI(last, dt=dt, **ISIInfo))
			if len(first) > 5:
				iqr.append(list(np.percentile(1./(np.diff(last)*dt), [25, 75])))
			else:
				iqr.append([np.min(1./(np.diff(last)*dt)),
					np.max(1./(np.diff(last)*dt))])

			if ISIInfo['depol'] == 'lastthird':
				err = err[-1]
				iqr = iqr[-1]

				if verbose > 3:
					print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

			else:
				if verbose > 3:
					for e in err:
						print(f"ISI = {e:.4g}ms (FR = {1/e:.4g}Hz)")

		else:
			err = getISI(spikeIdx, dt=dt, **ISIInfo)
			if verbose > 2:
				print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

		FArr.append(err)
		IQRArr.append(iqr)

	IArr = np.array(IArr)*1000.
	FArr = 1./np.array(FArr)
	IQRArr = np.diff(np.array(IQRArr), axis=2).squeeze()/2.
	IQRArr[IQRArr == 0] = np.inf
	# print(FArr)
	# print(IQRArr)
	# print(np.diff(IQRArr, axis=2).squeeze())

	if FArr.shape[1] == 2:
		p1, cov1 = np.polyfit(IArr, FArr[:, 0], deg=1, w=1./IQRArr[:, 0],
			full=False, cov=True)
		p2, cov2 = np.polyfit(IArr, FArr[:, 1], deg=1, w=1./IQRArr[:, 1],
			full=False, cov=True)

		P = [p1, p2]
		print(P)
		Cov = [cov1, cov2]

		err = [p[0] for p in P]

	else:
		P, Cov = np.polyfit(IArr, FArr, deg=1, w=1./IQRArr,
			full=False, cov=True)

		err = P[0]

	return err


################################################################################
##	Get F-I Slope (from dataFeat dictionary!)
################################################################################
def getFISlope_from_Dict(featDict, dt=0.001, returnAll=False, **kwds):

	verbose = utl.force_pos_int(verbose, name='epo.getFISlopeDict.verbose',
		zero_ok=True)

	dt = utl.force_pos_float(dt, name='epo.getFISlopeDict.dt', verbose=verbose)

	err_str = "Keyword argument 'returnAll' must be a boolean!"
	assert isinstance(returnAll, bool), err_str

	err_str = "Input argument 'featDict' must be a *dictionary* of ephys "
	err_str += "features containing 'ISI' fields indexed by current."
	assert isinstance(featDict, dict), err_str
	assert "ISI" in featDict.keys(), err_str
	assert "depol" in featDict['ISI'].keys(), err_str

	ISIDict = featDict['ISI']['depol']

	PList, CovList = [], []
	for key in ISIDict:
		if isinstance(ISIDict[key], dict):
			IArr, FArr = [], []
			for item in ISIDict[key].items():
				IArr.append(item[0])
				FArr.append(item[1])

			IArr = np.array(IArr)*1000.
			FArr = 1./np.array(FArr)

			if FArr.shape[1] == 2:
				p1, cov1 = np.polyfit(IArr, FArr[:, 0], deg=1, full=False,
					cov=True)
				p2, cov2 = np.polyfit(IArr, FArr[:, 1], deg=1, full=False,
					cov=True)

				P = [p1, p2]
				Cov = [cov1, cov2]

				err = [p[0] for p in P]

			else:
				P, Cov = np.polyfit(IArr, FArr, deg=1, full=False, cov=True)

				err = P[0]

			PList.append(P)
			CovList.append(Cov)

	if returnAll:
		return PList, CovList

	return err


################################################################################
##	Get Input Resistance
################################################################################
def getInputResistance(data, objDict, IArr, featDict=None, dt=0.001, covTol=1.,
	verbose=0, estC=False, estTau=False, returnAll=False, **kwds):

	## Check that the objDict is the 'objectives' field from an infoDict
	err_str = "Input argument 'epo.getInputRes.objDict' must be a dictionary!"
	assert isinstance(objDict, dict), err_str

	## Check verbose
	verbose = utl.force_pos_int(verbose, name='epo.getInputRes.verbose',
		zero_ok=True)

	## Check whether to estimate Capacitance, tau
	err_str = "Keyword argument 'epo.getRI.estC' must be a boolean!"
	assert isinstance(estC, bool)
	err_str = "Keyword argument 'epo.getRI.estTau' must be a boolean!"
	assert isinstance(estTau, bool)

	## Check returnAll keyword
	err_str = "Keyword argument 'epo.getRI.returnAll' must be a boolean!"
	assert isinstance(returnAll, bool)

	## Check dt
	dt = utl.force_pos_float(dt, name='epo.getInputRes.dt', verbose=verbose)

	## Check covTol
	covTol = utl.force_pos_float(covTol, name='epo.getInputRes.covTol',
		verbose=verbose)

	## If a feature dictionary is provided, extract the F-I slope from that.
	## (This method assumes that protocol==DEPOLSTEPS!)
	if (featDict is not None) and (isinstance(featDict, dict)):
		if verbose > 1:
			print("Extracting Input Resistance from Feature Dictionary!")
		return getRI_from_Dict(featDict, dt=dt, **objDict['RI'])

	if verbose > 1:
		print("Calculating Input Resistance from Data!")

	## Check that data is the right type and shape
	err_str = "Input argument 'data' must be 2D array of data."
	data = utl.force_float_arr(data, name='epo.getInputRes.data',
		verbose=verbose).squeeze()
	assert data.ndim == 2, err_str

	## Check that IArr is the right type
	IArr = utl.force_float_arr(IArr, name='epo.getInputRes.IArr',
		verbose=verbose).squeeze()*1000.

	## Iterate through the episodes... Assumes dpData is N x NEps shape
	PList, CovList = [], []
	for ii, D in enumerate(data.T):
		if verbose > 2:
			print(f"Extracting PSD from episode {ii}!")

		PSDInfo = objDict["PSD"]
		PSDInfo['fit'] = 'exp'

		tGrid = np.arange(len(D))*dt

		P, cov = epu.fitExp(D, times=tGrid, returnAll=True)
		cov = np.sqrt(cov)

		CoV = cov/np.abs(P)

		if np.all(CoV <= covTol):

			if verbose > 3:
				print("(epo.getInputRes): Exponential fit was good! " +
					f"(CoV <= {covTol:.3g})")

		else:
			if verbose > 3:
				print("(epo.getInputRes): Exponential fit was not good! " +
					f"(CoV > {covTol:.3g})")

			P = [D[0], np.mean(D), np.inf]
			cov = [np.inf, np.std(D), np.inf]

		PList.append(P)
		CovList.append(cov)

	PSDArr = np.array([P[1] for P in PList])
	PSDStdArr = 1./np.array([cov[1] for cov in CovList])

	linP, linCov = np.polyfit(IArr, PSDArr, deg=1, w=PSDStdArr, cov=True)
	linCov = np.sqrt(np.diag(linCov))

	if verbose > 3:
		print("(epo.getInputRes): Linear fit to V-I curve yields " +
			f"V = {linP[0]:.4g}I + {linP[1]:.4g}\n" +
			f"That is,\n\t\tR_I = {linP[0]:.4g} +/- {linCov[0]:.4g} GOhm")

	tauArr = np.array([P[2] if not np.isinf(P[2]) else tGrid[-1] 
		for P in PList])
	tauStdArr = np.array([cov[2]**2. if not np.isinf(cov[2]) else 10000. 
		for cov in CovList])

	tau = np.average(tauArr, weights=1./tauStdArr)*1000.
	tauStd = np.sqrt(1./np.sum(1./tauStdArr))*1000.

	if (verbose > 3) and estTau:
		print("(epo.getInputRes): Estimating time-constant as\n\t\t"+
			f"tau = {tau:.4g} +/- {tauStd:.4g} ms")

	C = tau/linP[0]

	CStd = np.sqrt((tauStd/linP[0])**2. + (tau*linCov[0]/linP[0]**2.)**2.)

	if (verbose > 3) and estC:
		print("(epo.getInputRes): Estimating cell capacitance as C = RI/tau" +
			f"\n\t\tC = {C:.4g} +/- {CStd:.4g} pF")

	if not returnAll:
		return linP[0]

	out = {
		"expFitPs":PList,
		"expFitCovs":CovList,
		"I":IArr,
		"linFitP":linP,
		"linFitCov":linCov
	}

	if estC:
		out['C'] = C
		out['CStd'] = CStd

	if estTau:
		out['tau'] = tau
		out['tauStd'] = tauStd

	return out


################################################################################
##	Get F-I Slope (from dataFeat dictionary!)
################################################################################
def getRI_from_Dict(featDict, dt=0.001, returnAll=False, verbose=0, **kwds):

	verbose = utl.force_pos_int(verbose, name='epo.getRIDict.verbose',
		zero_ok=True)

	dt = utl.force_pos_float(dt, name='epo.getRIDict.dt', verbose=verbose)

	err_str = "Keyword argument 'epo.getRIDict.returnAll' must be a boolean!"
	assert isinstance(returnAll, bool), err_str

	err_str = "Input argument 'epo.getRIDict.featDict' must be a *dictionary* "
	err_str += "of ephys features containing 'PSD' fields indexed by current."
	assert isinstance(featDict, dict), err_str
	assert "PSD" in featDict.keys(), err_str
	assert "hyperpol" in featDict['PSD'].keys(), err_str

	PSDDict = featDict['PSD']['hyperpol']

	PList, CovList = [], []
	for key in PSDDict:
		if isinstance(PSDDict[key], dict):
			IArr, PSDArr = [], []
			for item in PSDDict[key].items():
				IArr.append(item[0])
				PSDArr.append(item[1])

			IArr = np.array(IArr)*1000.
			PSDArr = np.array(PSDArr)

			P, Cov = np.polyfit(IArr, PSDArr, deg=1, full=False, cov=True)

			err = P[0]

			PList.append(P)
			CovList.append(Cov)

	if returnAll:
		return PList, CovList

	return err





	
























"""
================================================================================
	Electrophysiology Feature Extraction Functions (Version 2)
================================================================================

	Author: Eric Johnson
	Date Created: Thursday, March 7, 2019
	Date Modified: Monday, November 18, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains a module of functions that will extract various features
	from electrophysiology data.

	In particular, this function will contain methods for extracting:
	 - Spike Location and Height
	 - ISI
	 - Spike Amplitude
	 	- Average
	 	- Exponential Fit
	 - Inter-Spike Voltage
	 	- Average
	 	- Exponential Fit
	 - F-I slope and intercept
	 - Input Resistance
	 	- Linear Fit to Slope, with Cov
	 	- RI, tau, (C) with err

================================================================================
================================================================================
"""
from collections import Counter
from copy import deepcopy
import numpy as np
import os
import pyabf as abf
import scipy.interpolate as intrp
from scipy.optimize import curve_fit
import scipy.signal as sig
import scipy.stats as st
from skimage import filters
import warnings

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
def getSpikeIdx(data, dt=0.001, maxRate=100, minSlope=20., thresh=None,
	minProm=None, pPromAdj=0.1, window=100, pad=1, exact=True, minWidth=3,
	verbose=0):

	errHdr = "[getSpikeIdx]: "

	############################################################################
	##	Check Inputs, Keyword Arguments; Set-Up Median Subtraction
	############################################################################
	if True:
		
		verbose = forceFloat(verbose, 'verbose')

		if isinstance(data, abf.ABF):
			if verbose >= 1:
				print("\nWARNING: You have input a pyabf.ABF object, assuming "+
					"that you want spike finding on the *default* sweep!\n")
			X = data.sweepY.copy().astype(float).squeeze()
		elif isinstance(data, np.ndarray):
			X = data.copy().astype(float).squeeze()
			if X.ndim != 1:
				errStr = errHdr + "Input argument 'data' must be a 1D array."
				raise ValueError(errStr)
		else:
			errStr = errHdr + "Input argument 'data' must be a pyabf.ABF "
			errStr += "object or numpy.ndarray."
			raise TypeError(errStr)

		X = X.astype(float)

		dt = forcePosFloat(dt, 'dt', zeroOK=False, verbose=verbose)

		maxRate = forcePosFloat(maxRate, 'maxRate', zeroOK=False,
			verbose=verbose)

		minISI = int(1./dt/maxRate)

		minSlope = forcePosFloat(minSlope, 'minSlope', zeroOK=False,
			verbose=verbose)

		if thresh is not None:
			if isinstance(thresh, np.ndarray):
				errStr = errdr + "If supplying array of thresholds, must match "
				errStr = "number of sweeps in data!"
				assert len(thresh) == len(X), errStr
				thresh = thresh.astype(float).squeeze()
			else:
				thresh = forceFloat(thresh, 'thresh', verbose=verbose)
				thresh = (np.ones((len(X)))*thresh).astype(float)

		if minProm is not None:
			minProm = forcePosFloat(minProm, 'minProm', zeroOK=True,
				verbose=verbose)

		pPromAdj = forcePosFloat(pPromAdj, 'pPromAdj', zeroOK=True,
			verbose=verbose)

		window = forcePosInt(window, 'window', zeroOK=False, verbose=verbose)
		## window must be an odd number
		window = window + 1 if ((window % 2) == 0) else window

		pad = forcePosInt(pad, 'pad', zeroOK=False, verbose=verbose)

		minWidth = forcePosInt(minWidth, 'minWidth', zeroOK=False,
			verbose=verbose)
		if minWidth < 3:
			minWidth = 3

		errStr = "Invalid values for keywords 'minWidth' and 'maxRate', must "
		errStr += f"have minWidth < minISI=1/dt/maxRate; {minWidth} >= "
		errStr += f"{minISI}!"
		assert minWidth < minISI, errHdr + errStr

		exact = bool(exact)

		if verbose >= 2:
			if exact:
				print(f"Getting *exact* (non-integer) spike locations!")
			else:
				print(f"Getting array-located spike locations!")

	############################################################################
	##	Get Median-Subtracted Data
	############################################################################
	if True:

		noMedX = np.zeros_like(X)

		if verbose >= 1:
			print(f"Subtracting median filter!")

		## Subtract median from data
		noMedX = X - _getRollPerc(X, window=window, perc=50., verbose=verbose)

		## Adjust or set threshold
		if thresh is not None:
			thresh -= np.median(X)
		else:
			thresh = np.median(noMedX)

		promAdj = np.percentile(noMedX, pPromAdj)

	############################################################################
	##	Get Percentile Array from CDF of no-median data
	############################################################################
	if True:

		## Create CDF
		counts = Counter(noMedX.ravel())
		vals = np.msort(list(counts.keys()))
		CDF = np.cumsum(np.asarray([counts[ii] for ii in vals]))
		CDF = CDF / CDF[-1]

		## Assemble the list of possible prominences
		if minProm is not None:
			promArr = vals[vals >= minProm] - promAdj
		else:
			promArr = vals[CDF >= 0.02] - promAdj

		if len(promArr) == 0:
			promArr = np.array([minProm])
			pPromArr = np.array([1.])

		if verbose >= 3:
			print(f"There are {len(promArr)} prominences to check. " +
				f"({promArr[0]:.2f} to {promArr[-1]:.2f})")

	############################################################################
	##	Iterate through percentile array
	############################################################################
		NPeaksArr = np.zeros((len(promArr))).astype(int)
		for ii, prom in enumerate(promArr):

			## For each prominence, calculate the corresponding wlen
			wLenMin = int(np.ceil(prom/minSlope/dt))
			wLenMin = max(wLenMin, int(minWidth))
			wLenMin = wLenMin + 1 if ((wLenMin % 2) == 0) else wLenMin

			## Find the peaks
			peakIdx, _ = sig.find_peaks(noMedX, height=thresh, #width=minWidth,
				prominence=prom, wlen=wLenMin, distance=minISI,)
			NPeaksArr[ii] = len(peakIdx)
			# print(ii, prom, prom-promAdj, len(peakIdx))

		if np.all(NPeaksArr == 0):
			if verbose >= 1:
				print(f"\nNo peaks detected for any prominences!")
			return np.array([]), np.array([])

	############################################################################
	##	Find the location and number of peaks that persist over the largest
	##	range (in mV)
	############################################################################
		
		## Difference in the number of peaks from one prominence to the next.
		diffNPeaks = np.diff(NPeaksArr)

		## Find at which prominences the peak counts change
		promWhereNDiff = promArr[:-1][diffNPeaks.nonzero()]# - promAdj

		## Find peak numbers corresponding to these changes
		nPeaksWhereNDiff = NPeaksArr[:-1][diffNPeaks.nonzero()]

		## If the number of peaks doesn't change, this probably isn't a real
		## signal... Just use the most stringent prominence.
		if np.all(diffNPeaks == 0):

			promAtMode = promArr[-1]# - promAdj

			if verbose >= 1:
				print(f"\nNo differences in number of peaks detected... ")

		elif np.all(NPeaksArr == 0):

			promAtMode = promArr[-1]# - promAdj

			if verbose >= 1:
				printf("\nNo peaks detected for any prominence!")

		else:
			## Differences between prominences at which peak numbers change,
			## including end points
			paddedPromWhereNDiff = np.array([promArr[0]]+#-promAdj] + 
				list(promWhereNDiff) + [promArr[-1]])#-promAdj])
			diffPromWhereNDiff = np.diff(paddedPromWhereNDiff)

			## Maximum prominence difference between change points.
			## Note that we use np.where to find all the maxima and then we
			## select the last one.  We subtract 1 to account for the way that
			## differencing is performed.
			maxPromDiffIdx = np.where(
				diffPromWhereNDiff == diffPromWhereNDiff.max())[0][-1] + 1

			## Create a padded NPeaks array similar to paddedPromWhereNDiff
			paddedNWhereNDiff = np.array([NPeaksArr[0]] + 
				list(nPeaksWhereNDiff) + [NPeaksArr[-1]])

			if verbose >= 1:
				print(f"\nThere are {paddedNWhereNDiff[maxPromDiffIdx]} peaks!")
				print(f"\nThe best prominence is " +
					f"{paddedPromWhereNDiff[maxPromDiffIdx]:.2f} mV!")

			if verbose >= 3:
				print(f"\nMaximum prominence difference between peak-number "+
				f"changes was\n\t{paddedPromWhereNDiff[maxPromDiffIdx-1]:.2f} "+
				f"({paddedNWhereNDiff[maxPromDiffIdx-1]} peaks) to "+
				f"{paddedPromWhereNDiff[maxPromDiffIdx]:.2f} "+
				f"({paddedNWhereNDiff[maxPromDiffIdx]} peaks)")

			promAtMode = paddedPromWhereNDiff[maxPromDiffIdx]

		wLenAtMode = max(int(np.ceil(promAtMode/minSlope/dt)), int(minWidth*2))
		wLenAtMode = wLenAtMode + 1 if ((wLenAtMode % 2) == 0) else wLenAtMode

		peakIdx, peakVals = sig.find_peaks(noMedX, height=thresh, #width=minWidth,
			prominence=promAtMode, wlen=wLenAtMode, distance=minISI)

		NPeaks = len(peakIdx)

		if verbose >= 3:
			print(f"{NPeaks} peaks were found! (wlen = {wLenAtMode}," +
				f" prom = {promAtMode:.2f})")

	############################################################################
	##	Get Inexact Peak Locations, If Requested.
	############################################################################
		if not exact:

			if verbose >= 2:
				print("Returning integer-indexed spike locations.")

			spikeIdx = np.array(peakIdx).astype(int).squeeze()
			spikeVals = np.array(X[peakIdx]).astype(float).squeeze()

			## If spikeIdx is a single number, make it an array
			if spikeIdx.ndim == 0:
				spikeIdx = np.array([spikeIdx]).astype(int)
				spikeVals = np.array([spikeVals]).astype(float)

			return spikeIdx, spikeVals

	############################################################################
	##	Get Interpolated "Exact" Peak Locations, If Requested.
	############################################################################
		else:

			if verbose >= 2:
				print("Returning spline-interpolated spike locations.")

			with warnings.catch_warnings():
				warnings.filterwarnings("ignore")

				## Get the left and right edges of the peaks!
				widths, _, wLeft, wRight = sig.peak_widths(X, peakIdx,
					rel_height=0.5)
				wLeft = np.floor(wLeft).astype(int)
				wRight = np.ceil(wRight).astype(int)

			## Initialize *spike* arrays
			spikeIdx = np.zeros_like(peakIdx.squeeze()).astype(float)
			if spikeIdx.ndim == 0:
				spikeIdx = np.array([spikeIdx]).astype(float)
			spikeVals = np.zeros_like(spikeIdx).astype(float)

			if verbose >= 4:
				print(f"Itr:\t\tWL - PEAK - WR - Width")

			## Iterate through the peak edges
			for itr, (wl, wr) in enumerate(zip(wLeft, wRight)):

				## Check that the peak seems to be the right size (for fitting!)
				rightSize = ((wr-wl) > minWidth) and ((wr-wl) <= minISI)

				## While not the right size, move wl and wr around.
				counter, maxCounter = 0, 30
				while not rightSize:
					if verbose >= 4:
						print(f"{itr}:\tAdjust\t{wl} - {peakIdx[itr]} - "+
							f"{wr} - {wr-wl}")

					## If the width is too small, pad it to make it larger
					if (wr - wl) <= minWidth:
						if itr > 0:
							rightBound = min(wRight[itr-1], wLeft[itr],
								peakIdx[itr-1]+pad)
							wl = max(wl - pad, rightBound)

						else:
							wl = max(wl - pad, 0)

						if itr < NPeaks - 1:
							leftBound = max(wLeft[itr+1], wRight[itr],
								peakIdx[itr+1]-pad)
							wr = min(wr+pad, leftBound)
						else:
							wr = min(wr+pad, len(X)-1)

					## If the width is too large, move halfway closer to peak
					elif (wr - wl) > int(minISI/2.):
						if (peakIdx[itr] - wl) > int(minISI/4.):
							wl += int((peakIdx[itr] - wl)/2.)
						if (wr - peakIdx[itr]) > int(minISI/4.):
							wr -= int((wr - peakIdx[itr])/2.)

					## Check if right size now
					rightSize = ((wr - wl) > minWidth) and ((wr - wl) <= minISI)

					## Increment the counter... we're only trying so hard here
					counter += 1
					if counter > maxCounter:
						if verbose >= 0:
							print("\n")
							print(errHdr + "WARNING: Could not find optimal "+
								f"spike width for spike {itr} in {maxCounter}" +
								" attempts.  Moving on...")

						if verbose >= 1:
							print("\nDiagnostics:\n" + 40*"=" + "\n")
							print(f"itr:\t{itr}\nwl:\t{wl}")
							print(f"peakIdx:\t{peakIdx[itr]}\nwr:\t{wr}")
							print(f"NPeaks:\t{NPeaks}\nwlen:\t{wLenAtMode}")
							print("peakIdx:\n\t", peakIdx[itr-3:itr+4])

						break

				if verbose >= 5:
					print(f"{itr}:\tKeep\t{wl} - {peakIdx[itr]} - "+
						f"{wr} - {wr-wl}")

				## Grid for the data
				grid = np.arange(wl, wr+0.1).astype(int)
				## Grid for the spline
				finegrid = np.linspace(wl, wr, 1001)

				## Fit the spline to the data on the coarse grid
				splfit = intrp.splrep(grid, X[grid], k=3)
				## Calculate the derivative
				dsplfit = intrp.splder(splfit)
				## Fit the derivative on the fine grid... We need to do this
				## so we can bump up the order of the derivative spline and 
				## find the roots.
				derfit = intrp.splrep(finegrid,
					intrp.splev(finegrid, dsplfit), k=3)
				## Find the zeros of the derivative
				peakLoc = intrp.sproot(derfit, mest=len(finegrid))

				## If there are no peaks, skip this AP
				if len(peakLoc) == 0:
					if verbose >= 1:
						print(errHdr + "WARNING: peak detected but no maxima "+
							f"found... (itr = {itr})")
					continue

				## Get the peak height
				peakVal = intrp.splev(peakLoc, splfit)

				## Assume the AP is at the location of the largest maxima
				spikeIdx[itr] = peakLoc[np.argmax(peakVal)]
				spikeVals[itr] = np.max(peakVal)

			spikeIdx = spikeIdx.astype(float).squeeze()
			spikeVals = spikeVals.astype(float).squeeze()

	############################################################################
	##	Return spikeIdx, spikeVals
	############################################################################
		## If spikeIdx is a single number, make it an array
		if spikeIdx.ndim == 0:
			spikeIdx = np.array([spikeIdx]).astype(float)
			spikeVals = np.array([spikeVals]).astype(float)

		return spikeIdx, spikeVals


################################################################################
##	Get Inter-Spike Interval
################################################################################
def getISI(spikeIdx, dt=0.001, minRate=0., NSpikes=2):

	errHdr = "[getISI]: "

	############################################################################
	##	Check Inputs, Keyword Arguments; Set-Up Median Subtraction
	############################################################################
	if True:
		errStr = errHdr + "Input argument 'spikeIdx' must be a 1D array of "
		errStr += "floats."
		try:
			spikeIdx = np.array(spikeIdx, dtype=float, copy=True).squeeze()
			if spikeIdx.ndim == 0:
				spikeIdx = spikeIdx.reshape(1)
			else:
				assert spikeIdx.ndim == 1
		except:
			raise ValueError(errStr)

		## Conversion between units of spikeIdx and seconds
		dt = forcePosFloat(dt, 'dt', zeroOK=False)

		## Minimum detectable rate
		minRate = forcePosFloat(minRate, 'minRate', zeroOK=True)

		## Minimum number of spikes to use in calculation
		NSpikes = forcePosInt(NSpikes, 'NSpikes', zeroOK=False)

	############################################################################
	##	Calculate mean ISI
	############################################################################
	## If we have enough spikes
	if len(spikeIdx) > NSpikes:
		## Calculate the median difference
		ISI = np.median(np.diff(spikeIdx))*dt
		## If the firing rate is above the minimum we want to detect
		if 1./ISI >= minRate:
			return ISI ## Return the ISI

	## Otherwise return np.inf
	return np.inf


################################################################################
##	Get Firing Rate
################################################################################
def getFR(ISI):
	if isFloat(ISI):
		return 1./ISI
	return np.NaN


################################################################################
##	Get Spike Amplitude
################################################################################
def getSpikeAmp(spikeIdx, spikeVals, dt=0.001, NSpikes=2, fit='exp',
	covTol=0.4, returnAll=False, verbose=0):
	
	errHdr = "[getSpikeAmp]: "

	############################################################################
	##	Check Inputs, Keyword Arguments; Set-Up Median Subtraction
	############################################################################
	if True:
		
		verbose = forceFloat(verbose, 'verbose')

		errStr = errHdr + "Input argument 'spikeIdx' must be a 1D array of "
		errStr += "floats."
		try:
			spikeIdx = np.array(spikeIdx, dtype=float, copy=True).squeeze()
			assert spikeIdx.ndim == 1
		except:
			raise ValueError(errStr)

		errStr = errHdr + "Input argument 'spikeVals' must be a 1D array of "
		errStr += "floats."
		try:
			spikeVals = np.array(spikeVals, dtype=float, copy=True).squeeze()
			assert spikeVals.ndim == 1
		except:
			raise ValueError(errStr)

		## Conversion between units of spikeIdx and seconds
		dt = forcePosFloat(dt, 'dt', zeroOK=False, verbose=verbose)

		## Minimum number of spikes to use in calculation
		NSpikes = forcePosInt(NSpikes, 'NSpikes', zeroOK=False, verbose=verbose)

		errStr = "Keyword argument 'fit' must be either 'exp' or 'mean'."
		assert isinstance(fit, str), errHdr + errStr
		fit = fit.lower()
		assert fit in ['exp', 'mean'], errHdr + errStr

		covTol = forcePosFloat(covTol, 'covTol', zeroOK=True, verbose=verbose)

		try:
			returnAll = bool(returnAll)
		except:
			raise TypeError(errHdr + "Keyword argument 'returnAll' must be a "+
				"boolean.")

		if len(spikeIdx) <= NSpikes:
			if verbose >= 1:
				print(f"\nToo few spikes ({len(spikeIdx)} <= {NSpikes}), "+
					"returning NaN.")
			if returnAll:
				return np.array([np.NaN]), np.array([np.NaN])
			return np.array([np.NaN])

	############################################################################
	##	Fit Exponential, If Requested
	############################################################################
	if fit == 'exp':
		if verbose >= 2:
			print(f"Attempting to fit an exponential to spike amplitudes!")

		spikeT = spikeIdx*dt

		AmpP, AmpCov = fitExp(spikeVals, times=spikeT, returnAll=True)
		AmpP[AmpP == 0] = 1.e-16

		CoV = np.sqrt(AmpCov)/np.abs(AmpP)

		if CoV[1] < covTol:
			if verbose >= 2:
				print(f"Exponetial fit was good! (CoV <= {covTol:.3g})")

			if returnAll:
				return AmpP, AmpCov
			else:
				return AmpP

	if verbose >= 2:
		print(f"Returning mean as best fit for spike amplitudes!")

	if returnAll:
		return np.array([np.mean(spikeVals)]), np.array([np.std(spikeVals)])
	else:
		return np.array([np.mean(spikeVals)])


################################################################################
##	Get Inter-Spike Voltage
################################################################################
def getInterSpikeVoltage(data, dt=0.001, window=None, perc=None, fit='exp',
	covTol=0.1, returnAll=False, verbose=0):
	
	errHdr = "[getInterSpikeVoltage]: "

	############################################################################
	##	Check Inputs, Keyword Arguments; Set-Up Median Subtraction
	############################################################################
	if True:
		
		verbose = forceFloat(verbose, 'verbose')
	
		if isinstance(data, abf.ABF):
			if verbose >= 1:
				print("\nWARNING: You have input a pyabf.ABF object, assuming "+
					"that you want spike finding on the *default* sweep!\n")
			X = data.sweepY.copy().astype(float).squeeze()
		elif isinstance(data, np.ndarray):
			X = data.copy().astype(float).squeeze()
			if X.ndim != 1:
				errStr = errHdr + "Input argument 'data' must be a 1D array."
				raise ValueError(errStr)
		else:
			errStr = errHdr + "Input argument 'data' must be a pyabf.ABF "
			errStr += "object or numpy.ndarray."
			raise TypeError(errStr)

		X = X.astype(float)

		## Conversion between units of spikeIdx and seconds
		dt = forcePosFloat(dt, 'dt', zeroOK=False, verbose=verbose)

		if window is not None:
			window = forcePosInt(window, 'window', zeroOK=False,
				verbose=verbose)
		else:
			window = 100
		## window must be an odd number
		window = window + 1 if ((window % 2) == 0) else window

		## Percentile to assume is inter-spike voltage
		if perc is None:
			perc = 10.
		else:
			perc = forceFloat(perc, 'perc', zeroOk=True, verbose=verbose)

		covTol = forcePosFloat(covTol, 'covTol', zeroOK=True, verbose=verbose)

		errStr = "Keyword argument 'fit' must be either 'exp' or 'mean'."
		assert isinstance(fit, str), errHdr + errStr
		fit = fit.lower()
		assert fit in ['exp', 'mean'], errHdr + errStr

		try:
			returnAll = bool(returnAll)
		except:
			raise TypeError(errHdr + "Keyword argument 'returnAll' must be a "+
				"boolean.")

	############################################################################
	##	Use Windowed Order Filter to Compute Background
	############################################################################
	## Calculate the order
	order = int(window*perc/100.)

	## Compute windowed percentile
	windArr = np.ones(window)
	ISVArr = sig.order_filter(X, windArr, order)

	## Get windowed time array (taking half of window from each end)
	pad = int(window/2.)
	times = np.arange(len(X))[pad:-pad]*dt

	############################################################################
	##	Fit Exponential, If Requested
	############################################################################
	if fit == 'exp':
		if verbose >= 2:
			print(f"Attempting to fit an exponential to inter-spike voltages!")

		ISVP, ISVCov = fitExp(ISVArr, times=times, returnAll=True)
		ISVP[ISVP == 0] = 1.e-16

		if verbose >= 4:
			print(f"Fitted exponential parameters: {ISVP} (Cov = {ISVCov})")

		## Calculate the coefficient of variation
		CoV = np.sqrt(ISVCov)/np.abs(ISVP)

		if np.all(CoV <= covTol):
			if verbose >= 2:
				print(f"Exponetial fit was good! (all CoV <= {covTol:.3g})")

			if returnAll:
				return ISVP, ISVCov
			else:
				return ISVP

	if verbose >= 2:
		print(f"Returning mean as best fit for spike amplitudes!")

	if returnAll:
		return np.array([np.mean(ISVArr)]), np.array([np.std(ISVArr)])
	else:
		return np.array([np.mean(ISVArr)])


################################################################################
################################################################################
##
##		Utility Methods  (move to Utility folder some day)
##
################################################################################
################################################################################

################################################################################
##	Type coersion and checking methods.
################################################################################
def isFloat(value):
	try:
		float(value)
		return True
	except:
		return False


def forceFloat(value, name=None, verbose=5):

	if name is None:
		name = 'var'
	errStr = f"{name} = {value} is not floatable!"

	assert isFloat(value), errStr

	if verbose >= 5:
		print(f"Setting name = {value} to float.")

	return float(value)


def forcePosFloat(value, name=None, zeroOK=False, verbose=5):

	if name is None:
		name = 'var'

	value = forceFloat(value, name, verbose)

	errStr = f"{name} = {value} is not positive!"

	if zeroOK:
		assert value >= 0, errStr
	else:
		assert value > 0, errStr

	return value


def forceInt(value, name=None, verbose=5):

	if name is None:
		name = 'var'
	errStr = f"{name} = {value} is not floatable!"

	assert isFloat(value), errStr

	if verbose >= 5:
		print(f"Setting name = {value} to integer.")

	return int(value)


def forcePosInt(value, name=None, zeroOK=False, verbose=5):

	if name is None:
		name = 'var'

	value = forceInt(value, name, verbose)

	errStr = f"{name} = {value} is not positive!"

	if zeroOK:
		assert value >= 0, errStr
	else:
		assert value > 0, errStr

	return value

################################################################################
##	Get Windowed Percentile of Data
################################################################################
def _getRollPerc(data, window=101, perc=50., edgeCorrect=True, verbose=0):

	if verbose >= 3:
		print("Calculating windowed percentile of data.")
		print("WARNING: private method is not type-checking inputs!")

	order = int(window*perc/100.)

	medData = sig.order_filter(data, np.ones(window), order)

	if edgeCorrect:
		
		windArr = np.arange(window).astype(int)
		oddArr = (windArr + windArr%2. + 1).astype(int)

		leftEnd, rightEnd = [], []
		for printItr, (ii, wd) in enumerate(zip(windArr, oddArr)):
			printItr += 1

			if verbose >= 3:
				if (printItr) % 20. == 0.:
					print(f"{printItr}/{len(windArr)}: {ii}, {wd}")

			leftEnd.append(sig.order_filter(data[:window*2], np.ones(wd),
				int((wd-1)/2))[ii])

			wd = oddArr[-1] - wd + 1
			rightEnd.append(sig.order_filter(data[-window*2-1:],
				np.ones(wd), int((wd - 1)/2))[-(window-ii)-1])

		medData[:window] = np.array(leftEnd)
		medData[-window:] = np.array(rightEnd)

	return medData


################################################################################
##	Fitting Routines
################################################################################
def fitExp(data, times=None, returnAll=False):

	## Check that data is 1D float array
	try:
		data = np.array(data, dtype=float, copy=True).squeeze()
		assert data.ndim == 1
	except:
		errStr = "Input data must be 1D float array."
		raise ValueError(errStr)

	## Check that 'times' is 1D float array (if not None)
	if times is None:
		times = np.arange(len(data)).astype(float)
	else:
		try:
			times = np.array(times, dtype=float, copy=True).squeeze()
			assert times.ndim == 1
		except:
			errStr = "Input 'times' must be 1D float array."
			raise ValueError(errStr)

	## Try and fit the curve
	try:
		## Set initial guess based on data
		p0 = [
			max(-150, min(data[0], 100)),		## V0 (intial voltage)
			max(-150, min(np.mean(data), 100)), ## VInf (plateau voltage)
			max(0, min(len(data)/10., np.inf))	## tau (time constant)
		]

		## Set lower bounds
		lb = [
			min(-150, data[0]-20),
			min(-150, data.min()-30),
			0
		]

		## Set upper bounds
		ub = [
			max(100, data[0] + 20),
			max(100, data.max() + 30.),
			np.inf
		]

		params, cov = curve_fit(offsetExp, times, data, p0=p0, bounds=(lb, ub))
		cov = np.diag(cov)

	## If something has gone wrong, return a really bad result
	except:
		params = [-150, -150, 0]
		cov = [100, 100, 100]

	if returnAll:
		return np.array(params).astype(float), np.array(cov).astype(float)
	else:
		return np.array(params).astype(float)


def offsetExp(t, V0, VInf, tau):
	return VInf + (V0 - VInf)*np.exp(-t/tau)


################################################################################
################################################################################
##
##		TEST CODE
##
################################################################################
################################################################################
if __name__ == "__main__":

	import matplotlib.pyplot as plt
	import seaborn as sns

	import Utility.DataIO_util as DIO

	plt.close('all')
	sns.set(color_codes=True)

	fileName = "06/04/2011"
	dataNum = 7
	sweepNum = 0

	data = DIO.loadABF(fileName, dataNum=dataNum)

	print(data)

	data.setSweep(sweepNum)

	window = 101
	dt = 0.001
	maxRate = 100
	minISI = int(1./dt/maxRate)
	minSlope = 20.
	pad = 1
	minWidth = 3
	pPromAdj = 0.1

	verbose = 4

	spikeIdx, spikeVals = getSpikeIdx(data, dt=dt, maxRate=maxRate,
		minSlope=minSlope, minWidth=minWidth, pad=pad, window=window,
		verbose=verbose)

	ISI = getISI(spikeIdx, dt=dt)

	FR = getFR(ISI)

	AmpP, AmpCov = getSpikeAmp(spikeIdx, spikeVals, dt=dt, returnAll=True,
		verbose=verbose)

	if np.all([t == 'Step' for t in data.sweepEpochs.types]):
		if np.all([l == 0 for l in data.sweepEpochs.levels]):
			ISVP, ISVCov = getInterSpikeVoltage(data, dt=dt, returnAll=True,
				verbose=verbose)
		else:
			try:
				stepIdx = np.where([l > data.sweepEpochs.levels[0]
					for l in data.sweepEpochs.levels])[0][0]
				print(stepIdx)
				beg = data.sweepEpochs.p1s[stepIdx]
				end = data.sweepEpochs.p2s[stepIdx]
				stepTimes = data.sweepX[beg:end]
				stepVals = data.sweepY[beg:end]
				ISVP, ISVCov = getInterSpikeVoltage(stepVals, dt=dt,
					returnAll=True, verbose=verbose)
			except:
				ISVP = np.array([None])
				ISVCov = np.array([None])
	else:
		ISVP = np.array([None])
		ISVCov = np.array([None])

	print(f"\nThe Firing Rate is {FR:.2f}Hz (ISI = {ISI:.2f}s)\n")

	X = data.sweepY
	noMedX = X - _getRollPerc(X, window=window, perc=50)

	thresh = np.median(noMedX)
	promAdj = np.percentile(noMedX, pPromAdj)

	## Create CDF
	counts = Counter(noMedX.ravel())
	vals = np.msort(list(counts.keys()))
	CDF = np.cumsum(np.asarray([counts[ii] for ii in vals]))
	CDF = CDF / CDF[-1]

	promArr = vals[CDF >= 0.02]
	pPromArr = CDF[CDF >= 0.02]

	NPeaksArr = np.zeros((len(promArr))).astype(int)
	for ii, prom in enumerate(promArr):

		## For each prominence, calculate the corresponding wlen
		wLenMin = int(np.ceil((prom - promAdj)/minSlope/dt))
		wLenMin = max(wLenMin, int(minWidth))
		wLenMin = wLenMin + 1 if ((wLenMin % 2) == 0) else wLenMin

		## Find the peaks
		peakIdx, _ = sig.find_peaks(noMedX, height=thresh, #width=minWidth,
			prominence=prom-promAdj, wlen=wLenMin, distance=minISI,)
		NPeaksArr[ii] = len(peakIdx)

	## Difference in the number of peaks from one prominence to the next.
	diffNPeaks = np.diff(NPeaksArr)

	## Find at which prominences the peak counts change
	promWhereNDiff = promArr[:-1][diffNPeaks.nonzero()] - promAdj

	## Find peak numbers corresponding to these changes
	nPeaksWhereNDiff = NPeaksArr[:-1][diffNPeaks.nonzero()]
	
	if np.all(NPeaksArr == 0):
		promAtMode = promArr[-1] - promAdj

	else:
		## Differences between prominences at which peak numbers change,
		## including end points
		paddedPromWhereNDiff = np.array([promArr[0]-promAdj] + 
			list(promWhereNDiff) + [promArr[-1]-promAdj])
		diffPromWhereNDiff = np.diff(paddedPromWhereNDiff)

		## Maximum prominence difference between change points.
		## Note that we use np.where to find all the maxima and then we
		## select the last one.  We subtract 1 to account for the way that
		## differencing is performed.
		maxPromDiffIdx = np.where(
			diffPromWhereNDiff == diffPromWhereNDiff.max())[0][-1]+1

		## Create a padded NPeaks array similar to paddedPromWhereNDiff
		paddedNWhereNDiff = np.array([NPeaksArr[0]] + 
			list(nPeaksWhereNDiff) + [NPeaksArr[-1]])

		print(f"\nMaximum prominence difference between peak-number "+
			f"changes was\n\t{paddedPromWhereNDiff[maxPromDiffIdx-1]:.2f} "+
			f"({paddedNWhereNDiff[maxPromDiffIdx-1]} peaks) to "+
			f"{paddedPromWhereNDiff[maxPromDiffIdx]:.2f} "+
			f"({paddedNWhereNDiff[maxPromDiffIdx]} peaks)")

		print(f"\nThere are {paddedNWhereNDiff[maxPromDiffIdx]} peaks!")
		print(f"\nThe best prominence is " +
			f"{paddedPromWhereNDiff[maxPromDiffIdx]:.2f} mV!")

		promAtMode = paddedPromWhereNDiff[maxPromDiffIdx]
	
	wLenAtMode = max(int(np.ceil(promAtMode/minSlope/dt)), int(minWidth*2))
	wLenAtMode = wLenAtMode + 1 if ((wLenAtMode % 2) == 0) else wLenAtMode

	peakIdx, peakVals = sig.find_peaks(noMedX, height=thresh, #width=minWidth,
		prominence=promAtMode, wlen=wLenAtMode, distance=minISI)

	NPeaks = len(peakIdx)

	print(f"{NPeaks} peaks were found! (wlen = {wLenAtMode}," +
		f" prom = {promAtMode:.2f})")

	fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

	ax1.plot(data.sweepX, data.sweepY, lw=1)
	ax1.scatter(spikeIdx/data.dataRate, spikeVals, c='r',
		label=f'AP Locs (FR = {FR:.2f}Hz)')

	if len(AmpP) == 3:

		label = f"Amp " + r"$= " + f"{AmpP[1]:.2f}" + r" + \left("
		label += f"{AmpP[0]:.2f}" + r" - " + f"{AmpP[1]:.2f}"
		label += r"\right) e^{-t/" + f"{AmpP[2]:.2f}" + r"}$" + f"\n"
		label += "$\left(V_{\infty} = " + f"{AmpP[1]:.2f}" + r"mV\right)$"

		ax1.plot(spikeIdx*dt, offsetExp(spikeIdx*dt, *AmpP), '-g',
			label=label)
	else:
		try:
			ax1.hlines(xmin=spikeIdx[0]*dt, xmax=spikeIdx[-1]*dt, y=AmpP,
				color='g', label=f'Amp = {AmpP[0]:.2f}mV')
		except:
			pass

	if len(ISVP) == 3:

		label = f"ISV " + r"$= " + f"{ISVP[1]:.2f}" + r" + \left("
		label += f"{ISVP[0]:.2f}" + r" - " + f"{ISVP[1]:.2f}"
		label += r"\right) e^{-t/" + f"{ISVP[2]:.2f}" + r"}$" + f"\n"
		label += "$\left(V_{\infty} = " + f"{ISVP[1]:.2f}" + r"mV\right)$"

		ax1.plot(spikeIdx*dt, offsetExp(spikeIdx*dt, *ISVP), '-',
			color='purple', label=label)
	else:
		try:
			ax1.hlines(xmin=spikeIdx[0]*dt, xmax=spikeIdx[-1]*dt, y=ISVP,
				color='purple', label=f'ISV = {ISVP[0]:.2f}mV')
		except:
			pass

	ax2.plot(data.sweepX, noMedX, lw=1)

	ax2.scatter(data.sweepX[peakIdx], noMedX[peakIdx], c='r')

	if "(pA)" in data.sweepLabelC:
		ax3.plot(data.sweepX, data.sweepC*1000, lw=1)
	else:
		ax3.plot(data.sweepX, data.sweepC, lw=1)

	ax3.set_xlabel(data.sweepLabelX)
	ax3.set_ylabel(data.sweepLabelC)

	ax1.set_ylabel(data.sweepLabelY)
	ax1.legend()

	ax2.set_ylabel(data.sweepLabelY + "\n(Median Subtracted)")

	fig.tight_layout()

	figName = f"FeatureExtraction_{''.join(fileName.split('/'))}_{dataNum}_{sweepNum}.pdf"
	fig.savefig(os.path.join("Figures", figName), format='pdf')

	plt.show()
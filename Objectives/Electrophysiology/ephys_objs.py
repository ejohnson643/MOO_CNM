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
from copy import deepcopy
import numpy as np
import scipy.interpolate as intrp
from scipy.optimize import curve_fit
import scipy.signal as sig
import scipy.stats as st
from skimage import filters

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
def getSpikeIdx(data, dt=0.001, minDiff=7., thresh=None, maxRate=50., 
	exact=False, minProm=8., pad=1, verbose=0, **kwds):

	############################################################################
	##	Check Inputs, Keyword Arguments
	############################################################################
	if True:
		data = utl.force_float_arr(data, name='spike.data',
			verbose=verbose).squeeze()

		verbose = utl.force_pos_int(verbose, name='spike.verbose', 
			zero_ok=True, verbose=verbose)

		err_str = "Input argument 'data' must be 1D array."
		err_str += f" (data.ndim = {data.ndim})"
		assert data.ndim == 1, err_str

		## Set the minimum range of ephys data for spike detection
		## By default, if the range < 7mV, then it's assumed there are no
		## spikes...
		minDiff = utl.force_pos_float(minDiff, name='spike.minDiff',
			zero_ok=True, verbose=verbose)

		maxmin = data.max() - data.min()
		if (data.max() - np.median(data)) < minDiff:
			return [], []

		if thresh is not None:
			thresh = utl.force_float(thresh, name='spike.thresh',
				verbose=verbose)

		maxRate = utl.force_pos_float(maxRate, name='spike.maxRate',
			verbose=verbose)
		minISI = int(1./dt/maxRate)

		err_str = "Keyword argument 'exact' must be a boolean."
		assert isinstance(exact, bool), err_str

		minProm = utl.force_pos_float(minProm, name='spike.minProm',
			zero_ok=True, verbose=verbose)

		pad = utl.force_pos_int(pad, name='spike.pad', verbose=verbose)

		if verbose > 1:
			if exact:
				print(f"Getting Exact (non-integer) Spike Locations!")
			else:
				print(f"Getting Spike Locations")

	############################################################################
	##	Get Index Locations of Action Potentials
	############################################################################
	peakIdx, _ = sig.find_peaks(data, threshold=thresh, distance=minISI,
		prominence=minProm)
	NPeaks = len(peakIdx)

	############################################################################
	##	Get half-widths, left-, and right-edges of APs
	############################################################################
	widths, _, wLeft, wRight = sig.peak_widths(data, peakIdx, rel_height=0.5)
	wLeft = np.floor(wLeft).astype(int)
	wRight = np.ceil(wRight).astype(int)

	############################################################################
	##	Get Index/Time Locations of AP Peaks from Spline Fits
	############################################################################
	spikeIdx, spikeVals = [], []

	if verbose > 2:
		print("Itr:\tWL - Peak - WR - Width")

	## Iterate through peaks
	for itr, (wl, wr) in enumerate(zip(wLeft, wRight)):

		## Peak width must be > 3 indices apart and less than maxRate half-width
		rightSize = ((wr - wl) > 3) and ((wr - wl) < int(minISI/2.))
		counter, maxCounter = 0, 10
		while not rightSize:

			if verbose > 2:
				print(f"{itr}:\t Adjust {wl} - {peakIdx[itr]} - {wr} - {wr-wl}")

			## If the width is too small, pad it to make it larger
			if (wr - wl) <= 3:
				if itr > 0:
					wl = max(wl-pad, wRight[itr-1])
				else:
					wl = max(wl-pad, 0)

				if itr < NPeaks-1:
					wr = min(wr+pad, wLeft[itr+1])
				else:
					wr = min(wr+pad, len(data)-1)

			## If the width is too large, move halfway closer to the peak
			elif (wr - wl) >= int(minISI/2.):
				if (peakIdx[itr] - wl) >= int(minISI/4.):
					wl += int((peakIdx[itr] - wl)/2.)
				if (wr - peakIdx[itr]) >= int(minISI/4.):
					wr -= int((wr - peakIdx[itr])/2.)

			## Right size yet?
			rightSize = ((wr - wl) > 3) and ((wr - wl) < int(minISI/2.))

			## Increment the counter and only try so hard.
			counter += 1
			if counter > maxCounter:
				if verbose:
					print("(getSpikeIdx) WARNING: Could not find optimal "+
						f"spike width in {maxCounter} attempts... Moving on...")
				break

		if verbose > 2:
			print(f"{itr}:\t{wl} - {peakIdx[itr]} - {wr} - {wr-wl}")

		## Grid for data
		grid = np.arange(wl, wr+.1).astype(int)
		## Grid on which to evaluate the spline
		finegrid = np.linspace(wl, wr, 1001)

		## Fir the spline to the data on the coarse grid
		splfit = intrp.splrep(grid, data[grid], k=3) ## CUBIC
		## Calculate the derivative
		dsplfit = intrp.splder(splfit)
		## Fit the derivative to the fine grid
		derfit = intrp.splrep(finegrid, intrp.splev(finegrid, dsplfit), k=3)
		## Find the location of the zeros of the derivative
		peakLoc = intrp.sproot(derfit, mest=len(finegrid))

		## If we don't want non-integer spike locations
		if not exact:
			peakLoc = np.round(peakLoc).astype(int)

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

	## I DON'T UNDERSTAND THIS - CLEARLY SOME OLD PATCH...
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
		ISI = np.mean(np.diff(spikeIdx))*dt
		if 1./ISI >= minRate:
			return ISI

	return np.inf


################################################################################
##	Get AP Amplitude
################################################################################
def getSpikeAmp(spikeIdx, spikeVals, dt=0.001, NSpikes=1, fit='exp',
	returnAll=True, verbose=0, **kwds):

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

		dt = utl.force_pos_float(dt, name='getSpikeAmp.dt', verbose=verbose)

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

		err_str = "(ephys_objs.getSpikeAmp): Keyword argument 'returnAll' must "
		err_str += "be a boolean!"
		assert isinstance(returnAll, bool), err_str

	############################################################################
	##	Compute AP Amplitude
	############################################################################
	if fit == 'exp':
		spikeT = spikeIdx*dt

		AmpP, AmpCov = epu.fitExp(spikeVals, times=spikeT, returnAll=True)



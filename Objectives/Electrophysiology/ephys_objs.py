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
def getSpikeIdx(data, minDiff=7., thresh=None, maxRate=50., exact=False,
	minProm=8., pad=1, dt=0.001):

	############################################################################
	##	Check Inputs, Keyword Arguments
	############################################################################
	if True:
		data = utl.force_float_arr(data, name='data').squeeze()

		err_str = "Input argument 'data' must be 1D array."
		err_str += f" (data.ndim = {data.ndim})"
		assert data.ndim == 1, err_str

		## Set the minimum range of ephys data for spike detection
		## By default, if the range < 7mV, then it's assumed there are no
		## spikes...
		minDiff = utl.force_pos_float(minDiff, name='spike.minDiff',
			zero_ok=True)

		maxmin = data.max() - data.min()
		if (data.max() - np.median(data)) < minDiff:
			return [], []

		if thresh is not None:
			thresh = utl.force_float(thresh, name='spike.thresh')

		maxRate = utl.force_pos_float(maxRate, name='spike.maxRate')
		minISI = int(1./dt/maxRate)

		err_str = "Keyword argument 'exact' must be a boolean."
		assert isinstance(exact, bool), err_str

		minProm = utl.force_pos_float(minProm, name='spike.minProm',
			zero_ok=True)

		pad = utl.force_pos_int(pad, name='spike.pad')

	############################################################################
	##	Get Index Locations of Action Potentials
	############################################################################
	peakIdx, _ = sig.find_peaks(data, threshold=thresh, distance=minISI,
		prominence=minProm)

	############################################################################
	##	Get half-widths, left-, and right-edges of APs
	############################################################################
	widths, _, wLeft, wRight = sig.peak_widths(data, peakIdx, rel_height=0.5)
	wLeft = np.floor(wLeft).astype(int)
	wRight = np.ceil(wRight).astype(int)

	############################################################################
	##	Get Index/Time Locations of AP Peaks from Spline Fits
	############################################################################
	peakIdx, peakVals = [], []

	## Iterate through peaks
	for itr, (wl, wr) in enumerate(zip(wLeft, wRight)):
		## Peak width must be > 3 indices apart and less than maxRate half-width
		if ((wr - wl) > 3) and ((wr - wl) < int(minISI/2.)):
			right_size = True
		else:
			right_size = False

		###### FIX THIS.  Make it move in interval, bisect between min/max

		# ## If the spline interval is not right
		# while not right_size:
		# 	## If it's too small, pad
		# 	if (wr - wl) <= 3:
		# 		wl = max(wl - pad, 0)
		# 		wr = min(wr + pad, len(wRight)-1)

		# 		if ((wr - wl) > 3) and ((wr - wl) < int(minISI/2.)):
		# 			break

		# 	## If it's too big, shrink
		# 	elif itr > 0:
		# 		## As long as we're still right of the previous peak
		# 		if wl < wRight[itr-1]:
		# 			wl += int(minISI/4.)
		# 		else:
		# 			right_size = True

		# 	## 




	return None, None
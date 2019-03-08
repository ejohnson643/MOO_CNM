"""
================================================================================
	Electrophysiology Feature Extraction Utility Functions
================================================================================

	Author: Eric Johnson
	Date Created: Thursday, March 7, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains a module of functions that will be useful in the 
	extraction of various features from electrophysiology data.

	In particular, this function will contain methods for protocol detection and
	for the creation of protocol dictionaries.

================================================================================
================================================================================
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import curve_fit

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
##		Electrophysiology Protocol Methods
##
################################################################################
################################################################################

################################################################################
## Get Experimental Protocol
################################################################################
def getExpProtocol(hdr, useList=True):
	"""getExpProtocol(hdr, uselist=True)

	This function will attemp to automatically detect the type of protocol that
	was implemented in a given data recording by parsing the header of the abf
	file.

	At the moment, we can reliably detect 6 different protocols:

	EPHYS_PROT_REST				= 0
		This is detected by noting that the entire waveform is 0

	EPHYS_PROT_DEPOLSTEP		= 1

	EPHYS_PROT_HYPERPOLSTEP		= 2

	EPHYS_PROT_DEPOLSTEPS		= 3

	EPHYS_PROT_HYPERPOLSTEPS	= 4

	EPHYS_PROT_CONSTHOLD		= 5
		This is detected by noting that the entire waveform is being held at one
		value.

	TO DO:
	======
	 - Implement use of useList
	 - Allow for other channels to be used, not just mV channel.
	 - Allow for ramped epochs?

	Inputs:
	=======
	hdr 		(dict)		Header structure as extracted from an abf file.
							(Use abf.ABF_read(abf file))

	useList 	(bool)		(Optional, not implemented) Eventually we would like
							to have an annotated database of the data that would
							be cross-referenced to indicate the type of
							protocol, for now we just try to autodetect.

	Outputs:
	========
	protNo 		(int)		Number indicating which protocol has been detected.
							-1 indicates that an unknown protocol has been
							detected.
	"""

	## Check that our default channel has mV units
	if (hdr['recChUnits'][hdr['nADCSamplingSeq'][0]] != 'mV'):
		return -1

	## Check that the epoch types are either disabled or steps
	## (Ramps are not allowed at this time)
	if np.any(hdr['nEpochType'] > 1):
		return -1

	## Get the indices of the relevant epochs
	epochIdx = getEpochIdx(hdr)

	## Get the applied waveform for this observation
	waveform = abf.GetWaveform(hdr, hdr['lActualEpisodes'])

	## If the waveform is all 0, then assume the protocol is rest
	if np.all(waveform == 0):
		return EPHYS_PROT_REST

	## If the waveform has one value and there is only one epoch (after 
	## holding), then assume the protocol is a constant hold.
	if np.sum(hdr['nEpochType'][0] == 1) == 1:
		if len(np.unique(waveform)) == 1:
			return EPHYS_PROT_CONSTHOLD

	## If the waveform has two values, and there are more than 3 epochs (after
	## holding), then assume that the protocol is a (set of) depol or hyperpol
	## step(s).
	holdCurr = abf.GetHoldingLevel(hdr, hdr['nActiveDACChannel'], 1)
	if np.sum(hdr['nEpochType'][0] == 1) > 2:
		## Check the unique epoch current levels
		epochLevs = hdr['fEpochInitLevel'][0][(hdr['nEpochType'][0]).nonzero()]
		epochLevs = np.unique(epochLevs)
		## If there are two unique levels...
		if len(epochLevs) == 2:
			## But they are both different from the holduing current, breal
			if np.sum(epochLevs != holdCurr) > 1:
				return -1

			## Get non-holding current level
			nonZeroLev = epochLevs[(epochLevs != holdCurr).nonzero()]

			## If it's positive, it's a depol step(s)
			if nonZeroLev > 0:
				if hdr['lActualEpisodes'] > 1:
					return EPHYS_PROT_DEPOLSTEPS
				else:
					return EPHYS_PROT_DEPOLSTEP

			## If it's negative, it's a hyperpol step(s)
			else:
				if hdr['lActualEpisodes'] > 1:
					return EPHYS_PROT_HYPERPOLSTEPS
				else:
					return EPHYS_PROT_HYPERPOLSTEP

		## If there's only one voltage...
		elif len(epochLevs) == 1:
			## Get any increments
			epochIncs = hdr['fEpochLevelInc'][0]
			epochIncs = np.unique(epochIncs[(hdr['nEpochType'][0]).nonzero()])

			## Check whether there's a single non-zero voltage increment
			if np.sum(epochIncs != 0) == 1:

				## Get the value of the increment
				nonZeroInc = epochIncs[(epochIncs).nonzero()]

				## If it's positive, it's a depol step(s)
				if nonZeroInc > 0:
					if hdr['lActualEpisodes'] > 1:
						return EPHYS_PROT_DEPOLSTEPS
					else:
						return EPHYS_PROT_DEPOLSTEP

				## If it's negative, it's a hyperpol step(s)
				else:
					if hdr['lActualEpisodes'] > 1:
						return EPHYS_PROT_HYPERPOLSTEPS
					else:
						return EPHYS_PROT_HYPERPOLSTEP

	## It its none of these return unknown
	return -1


################################################################################
## Get Epoch Indices for the Relevant Channel
################################################################################
def getEpochIdx(hdr):
	"""getEpochIdx(hdr)

	This is a shortcut to get the epoch indices (in the waveform array) of the
	voltage channel (which is usually what we want).  This is basically a 
	wrapper for abf.GetEpochIdx, which outputs the epoch indices for all
	channels.
	"""

	return abf.GetEpochIdx(hdr)[hdr['nActiveDACChannel']]

################################################################################
################################################################################
##
##		Fitting Routines
##
################################################################################
################################################################################

################################################################################
## Fit an Exponential
################################################################################
def fitExp(data, times=None, returnAll=False):

	## Check that data is floats and 1D
	data = utl.force_float_arr(data).squeeze()

	err_str = "(ephys_utl.fitExp): Invalid shape for input argument 'data'; "
	err_str += f"expected 1D, got {data.shape}"
	assert data.ndim == 1, err_str

	## Set time grid on which to fit Exp
	if times is None:
		times = np.arange(len(data)).astype(float)
	else:
		times = utl.force_float_arr(times)

	## Try and fit the curve!
	try:
		## Set initial guess based on data
		p0 = [
			max(-150, min(data[0], 100)),
			max(-150, min(np.mean(data), 100)),
			max(0, min(len(data)/10., np.inf))
		]

		## Fit the curve with some reasonable bounds
		params, cov = curve_fit(offsetExp, times, data, p0=p0,
			bounds=([-150, -150, 0], [100, 100, np.inf]))
		cov = np.diag(cov)

	## If something went wrong, return a really bad result
	except:
		params = [-150, -150, 0]
		cov = [100, 100, 100]

	## If needed, return everything
	if returnAll:
		return params, cov
	else:
		return params

################################################################################
## Offset Exponenital Function
################################################################################
def offsetExp(t, V0, VInf, tau):
	return VInf + (V0-VInf)*np.exp(-t/tau)
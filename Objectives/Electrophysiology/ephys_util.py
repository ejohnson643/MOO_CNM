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
import scipy.signal as sig

import Objectives.Electrophysiology.ephys_objs as epo

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
			if nonZeroLev > holdCurr:
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

			elif len(epochIncs) == 1:
				return EPHYS_PROT_CONSTHOLD

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
##		Electrophysiology Feature Extraction Methods
##
################################################################################
################################################################################

################################################################################
## Get Electrophysiology Features
################################################################################
def getEphysFeatures(dataDict, infoDict, verbose=None):

	if verbose is not None:
		try:
			verbose = utl.force_pos_int(verbose,
				name='epu.getEphysFeatures.verbose', zero_ok=True)
		except:
			verbose = 0
	else:
		try:
			verbose = infoDict['objectives']['verbose']
		except:
			verbose = 0

	if verbose:
		print_str = "\nExtracting Electrophysiology Features from Data\n"
		print(print_str + (len(print_str)+1)*"=")

	dataFeat = {}

	keys = sorted(list(dataDict.keys()))

	for key in keys:
		data = dataDict[key]['data']
		hdr = dataDict[key]['header']

		protocol = getExpProtocol(hdr)

		if protocol == EPHYS_PROT_REST:
			dataFeat = getRestFeatures(data, hdr, infoDict, dataFeat, key=key,
				verbose=verbose)

		elif protocol == EPHYS_PROT_DEPOLSTEP:
			dataFeat = getDepolFeatures(data, hdr, infoDict, dataFeat, key=key,
				verbose=verbose)

		elif protocol == EPHYS_PROT_HYPERPOLSTEP:
			dataFeat = getHyperpolFeatures(data, hdr, infoDict, dataFeat,
				key=key, verbose=verbose)

		elif protocol == EPHYS_PROT_DEPOLSTEPS:
			dataFeat = getDepolStepsFeatures(data, hdr, infoDict, dataFeat,
				key=key, verbose=verbose)

		elif protocol == EPHYS_PROT_HYPERPOLSTEPS:
			dataFeat = getHyperpolStepsFeatures(data, hdr, infoDict, dataFeat,
				key=key, verbose=verbose)

		elif protocol == EPHYS_PROT_CONSTHOLD:
			dataFeat = getConstHoldFeatures(data, hdr, infoDict, dataFeat,
				key=key, verbose=verbose)

		else:
			if verbose > 1:
				print_str = f"Unknown protocol = {protocol}, cannot extract "
				print(print_str + "features!")
			continue

	return dataFeat


################################################################################
## Get Electrophysiology Features from Rest Protocol
################################################################################
def getRestFeatures(data, hdr, infoDict, dataFeat, key=None, verbose=0):

	verbose = utl.force_pos_int(verbose, name='epu.getRestFeatures.verbose',
		zero_ok=True)

	if verbose > 1:
		print("Getting features from REST PROTOCOL")

	data = data[:, 0].squeeze()

	spikeIdx, spikeVals = epo.getSpikeIdx(data, dt=infoDict['data']['dt'],
		**infoDict['objectives']['Spikes'])

	for obj in infoDict['objectives']:

		subInfo = infoDict['objectives'][obj]

		## For rest protocols, only want mean fit.
		subInfo['fit'] = 'mean'

		if obj == 'ISI':
			err = epo.getISI(spikeIdx, dt=infoDict['data']['dt'],
				**subInfo)

			if verbose > 2:
				print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

		elif obj == 'Amp':
			err = epo.getSpikeAmp(spikeIdx, spikeVals,
				dt=infoDict['data']['dt'], **subInfo)

			if verbose > 2:
				print(f"Amp = {err:.4g}mV")

		elif obj == 'PSD':
			err = epo.getPSD(data, spikeIdx,
				dt=infoDict['data']['dt'], **subInfo)

			if verbose > 2:
				print(f"PSD = {err:.4g}mV")

		else:
			continue

		try:
			_ = dataFeat[obj]
		except KeyError:
			dataFeat[obj] = {}

		try:
			_ = dataFeat[obj]['rest']
		except KeyError:
			dataFeat[obj]['rest'] = {}

		if key is not None:
			key = utl.force_pos_int(key, name='epu.getRestFeatures.key',
				zero_ok=True, verbose=verbose)
		else:
			key = list(out[obj]['rest'].keys()).sort().pop() + 1

		dataFeat[obj]['rest'][key] = deepcopy(err)

	return dataFeat.copy()


################################################################################
## Get Electrophysiology Features from Depolarization Step Protocol
################################################################################
def getDepolFeatures(data, hdr, infoDict, dataFeat, key=None, verbose=0):

	verbose = utl.force_pos_int(verbose, name='epu.getDepolFeatures.verbose',
		zero_ok=True)

	if verbose > 1:
		print("Getting features from DEPOL STEP PROTOCOL")

	dt = deepcopy(infoDict['data']['dt'])

	data = data[:, 0].squeeze()

	dpData, dpIdx, dpI = getDepolIdx(data, hdr, protocol=EPHYS_PROT_DEPOLSTEP)

	tGrid = abf.GetTimebase(hdr, 0)[dpIdx[0]:dpIdx[1]]*dt

	spikeIdx, spikeVals = epo.getSpikeIdx(dpData, dt=dt,
		**infoDict['objectives']['Spikes'])

	for obj in infoDict['objectives']:

		subInfo = infoDict['objectives'][obj]

		if obj == "ISI":
			if len(spikeIdx) == 0:
				continue

			if subInfo['depol'] in ['thirds', 'lastthird']:
				bounds = np.linspace(0, len(tGrid), 4).astype(int)

				err = []

				first = spikeIdx[spikeIdx < bounds[1]]
				err.append(epo.getISI(first, dt=dt, **subInfo))

				last = spikeIdx[spikeIdx >= bounds[2]]
				err.append(epo.getISI(last, dt=dt, **subInfo))

				if subInfo['depol'] == 'lastthird':
					err = err[-1]

					if verbose > 2:
						print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

				else:
					if verbose > 2:
						for e in err:
							print(f"ISI = {e:.4g}ms (FR = {1/e:.4g}Hz)")

			else:
				err = epo.getISI(spikeIdx, dt=dt, **subInfo)

		elif obj == 'Amp':
			if len(spikeIdx) == 0:
				continue

			err = epo.getSpikeAmp(spikeIdx, spikeVals, **subInfo)

			if not isinstance(err, float):
				err = err[1]

			if verbose > 2:
				print(f"Amp = {err:.4g}mV")

		elif obj == 'PSD':
			err = epo.getPSD(dpData, spikeIdx, dt=dt, **subInfo)

			if not isinstance(err, float):
				err = err[1]

			if verbose > 2:
				print(f"PSD = {err:.4g}mV")

		else:
			continue

		try:
			_ = dataFeat[obj]
		except KeyError:
			dataFeat[obj] = {}

		try:
			_ = dataFeat[obj]['depol']
		except KeyError:
			dataFeat[obj]['depol'] = {}

		if key is not None:
			key = utl.force_pos_int(key, name='epu.getDepolFeatures.key',
				zero_ok=True, verbose=verbose)
		else:
			key = list(out[obj]['depol'].keys()).sort().pop() + 1

		dataFeat[obj]['depol'][key] = deepcopy(err)

	return dataFeat.copy()


################################################################################
## Get Electrophysiology Features from Hyperpolarization Step Protocol
################################################################################
def getHyperpolFeatures(data, hdr, infoDict, dataFeat, key=None, verbose=0):

	verbose = utl.force_pos_int(verbose, name='epu.getHyperpolFeatures.verbose',
		zero_ok=True)

	if verbose > 1:
		print("Getting features from HYPERPOL STEP PROTOCOL")

	dt = deepcopy(infoDict['data']['dt'])

	data = data[:, 0].squeeze()

	hpData, hpIdx, hpI = getHyperpolIdx(data, hdr,
		protocol=EPHYS_PROT_HYPERPOLSTEP)

	spikeIdx, spikeVals = epo.getSpikeIdx(hpData, dt=dt,
		**infoDict['objectives']['Spikes'])

	for obj in infoDict['objectives']:

		subInfo = infoDict['objectives'][obj]

		if obj == 'PSD':
			err = epo.getPSD(data, spikeIdx, dt=dt, **subInfo)

			if not isinstance(err, float):
				err = err[1]

			if verbose > 2:
				print(f"PSD = {err:.4g}mV")

		else:
			continue

		try:
			_ = dataFeat[obj]
		except KeyError:
			dataFeat[obj] = {}

		try:
			_ = dataFeat[obj]['hyperpol']
		except KeyError:
			dataFeat[obj]['hyperpol'] = {}

		if key is not None:
			key = utl.force_pos_int(key, name='epu.getHyperpolFeatures.key',
				zero_ok=True, verbose=verbose)
		else:
			key = list(out[obj]['hyperpol'].keys()).sort().pop() + 1

		dataFeat[obj]['hyperpol'][key] = deepcopy(err)

	return dataFeat.copy()


################################################################################
## Get Electrophysiology Features from Depolarization Steps Protocol
################################################################################
def getDepolStepsFeatures(data, hdr, infoDict, dataFeat, key=None, verbose=0):

	verbose = utl.force_pos_int(verbose,
		name='epu.getDepolStepsFeatures.verbose', zero_ok=True)

	if verbose > 1:
		print("Getting features from DEPOL STEPS PROTOCOL")

	dt = deepcopy(infoDict['data']['dt'])

	data = data[:, 0].squeeze()

	dpData, dpIdx, dpI = getDepolIdx(data, hdr, protocol=EPHYS_PROT_DEPOLSTEPS)

	if verbose > 2:
		print(f"There are {hdr['lActualEpisodes']} episodes!")

	for ii, dpD in enumerate(dpData.T):

		if verbose > 2:
			print(f"Extracting features from episode {ii}!")

		spikeIdx, spikeVals = epo.getSpikeIdx(dpD, dt=dt,
			**infoDict['objectives']['Spikes'])

		for obj in infoDict['objectives']:

			if verbose > 3:
				print(f"Considering objective {obj}")

			subInfo = infoDict['objectives'][obj]

			if obj == "ISI":

				if subInfo['depol'] in ['thirds', 'lastthird']:
					bounds = np.linspace(0, len(dpD), 4).astype(int)

					err = []

					first = spikeIdx[spikeIdx < bounds[1]]
					err.append(epo.getISI(first, dt=dt, **subInfo))

					last = spikeIdx[spikeIdx >= bounds[2]]
					err.append(epo.getISI(last, dt=dt, **subInfo))

					if subInfo['depol'] == 'lastthird':
						err = err[-1]

						if verbose > 2:
							print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

					else:
						if verbose > 2:
							for e in err:
								print(f"ISI = {e:.4g}ms (FR = {1/e:.4g}Hz)")

				else:
					err = epo.getISI(spikeIdx, dt=dt, **subInfo)
					if verbose > 2:
						print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

			elif obj == 'Amp':
				err = epo.getSpikeAmp(spikeIdx, spikeVals, **subInfo)

				if not isinstance(err, float):
					err = err[1]

				if verbose > 2:
					print(f"Amp = {err:.4g}mV")

			elif obj == 'PSD':
				err = epo.getPSD(dpD, spikeIdx, dt=dt, **subInfo)

				if not isinstance(err, float):
					err = err[1]

				if verbose > 2:
					print(f"PSD = {err:.4g}mV")

			else:
				continue

			try:
				_ = dataFeat[obj]
			except KeyError:
				dataFeat[obj] = {}

			try:
				_ = dataFeat[obj]['depol']
			except KeyError:
				dataFeat[obj]['depol'] = {}

			if key is not None:
				key = utl.force_pos_int(key, name='epu.getDepolStepsFeats.key',
					zero_ok=True, verbose=verbose)
			else:
				key = list(out[obj]['depol'].keys()).sort().pop() + 1

			try:
				_ = dataFeat[obj]['depol'][key]
			except KeyError:
				dataFeat[obj]['depol'][key] = {}

			dataFeat[obj]['depol'][key][dpI[ii]] = deepcopy(err)

	if "FI" in infoDict['objectives'].keys():

		err = epo.getFISlope(dpData, infoDict['objectives'], dpI, dt=dt,
			returnAll=False)

		if verbose > 2:
			if not isinstance(err, float):
				for e in err:
					print(f"F-I Slope = {e:.4g}Hz/pA")
			else:
				print(f"F-I Slope = {err:.4g}Hz/pA")

		try:
			_ = dataFeat['FI']
		except KeyError:
			dataFeat['FI'] = {}

		if key is not None:
			key = utl.force_pos_int(key, name='epu.getDepolStepsFeats.key',
				zero_ok=True, verbose=verbose)
		else:
			key = list(out['FI'].keys()).sort().pop() + 1

		try:
			_ = dataFeat['FI'][key]
		except KeyError:
			dataFeat['FI'][key] = {}

		dataFeat['FI'][key] = deepcopy(err)

	return dataFeat


################################################################################
## Get Electrophysiology Features from Hyperpolarization Steps Protocol
################################################################################
def getHyperpolStepsFeatures(data, hdr, infoDict, dataFeat, key=None,
	verbose=0):

	verbose = utl.force_pos_int(verbose,
		name='epu.getHyperpolStepsFeatures.verbose', zero_ok=True)

	if verbose > 1:
		print("Getting features from HYPERPOL STEPS PROTOCOL")

	dt = deepcopy(infoDict['data']['dt'])

	data = data[:, 0].squeeze()

	hpData, hpIdx, hpI = getHyperpolIdx(data, hdr,
		protocol=EPHYS_PROT_HYPERPOLSTEPS)

	if verbose > 2:
		print(f"There are {hdr['lActualEpisodes']} episodes!")


	for ii, hpD in enumerate(hpData.T):

		if verbose > 2:
			print(f"Extracting features from episode {ii}!")

		spikeIdx, spikeVals = epo.getSpikeIdx(hpD, dt=dt,
			**infoDict['objectives']['Spikes'])

		for obj in infoDict['objectives']:

			if verbose > 3:
				print(f"Considering objective {obj}")

			subInfo = infoDict['objectives'][obj]

			if obj == "PSD":
				err = epo.getPSD(hpD, spikeIdx, dt=dt, **subInfo)

				if not isinstance(err, float):
					err = err[1]

				if verbose > 2:
					print(f"PSD = {err:.4g}mV")

			else:
				continue

			try:
				_ = dataFeat[obj]
			except KeyError:
				dataFeat[obj] = {}

			try:
				_ = dataFeat[obj]['hyperpol']
			except:
				dataFeat[obj]['hyperpol'] = {}

			if key is not None:
				key = utl.force_pos_int(key, name='epu.getHyperpolFeatures.key',
					zero_ok=True, verbose=verbose)
			else:
				key = list(out[obj]['hyperpol'].keys()).sort().pop() + 1

			try:
				_ = dataFeat[obj]['hyperpol'][key]
			except KeyError:
				dataFeat[obj]['hyperpol'][key] = {}

			dataFeat[obj]['hyperpol'][key][hpI[ii]] = deepcopy(err)

	if "RI" in infoDict['objectives'].keys():

		err = epo.getInputResistance(hpData, infoDict['objectives'], hpI,
			dt=dt, **infoDict['objectives']['RI'])

		if isinstance(err, dict):
			err = err['linFitP'][0]

		if verbose > 2:
			print(f"Input Resistance = {err:.4g} GOhms")

		try:
			_ = dataFeat['RI']
		except KeyError:
			dataFeat['RI'] = {}

		if key is not None:
			key = utl.force_pos_int(key, name='epu.getHyperpolStepsFeats.key',
				zero_ok=True, verbose=verbose)
		else:
			key = list(out['RI'].keys()).sort().pop() + 1

		try:
			_ = dataFeat['RI'][key]
		except KeyError:
			dataFeat['RI'][key] = {}

		print(dataFeat['RI'], key, dataFeat['RI'][key])

		dataFeat['RI'][key] = deepcopy(err)

	return dataFeat


################################################################################
## Get Electrophysiology Features from Constant Holding Protocol
################################################################################
def getConstHoldFeatures(data, hdr, infoDict, dataFeat, key=None, verbose=0):

	verbose = utl.force_pos_int(verbose,
		name='epu.getConstHoldFeatures.verbose', zero_ok=True)

	if verbose > 1:
		print("Getting features from CONSTANT HOLD PROTOCOL")

	if hdr['lActualEpisodes'] > 1:
		if verbose:
			print("WARNING: At this time, we cannot handle multi-episode hold"
				"ing current protocols (non-rest + no current-injection).")
		return dataFeat

	data = data[:, 0].squeeze()

	## For const hold protocols, want to make spikes are larger than noise
	holdData = data[:int(abf.GetHoldingDuration(hdr)/hdr['nADCNumChannels'])]
	spikeDict = infoDict['objectives']['Spikes'].copy()

	spikeIdx, spikeVals = epo.getSpikeIdx(data, dt=infoDict['data']['dt'],
		**spikeDict)

	for obj in infoDict['objectives']:

		subInfo = infoDict['objectives'][obj]

		## For const hold protocols, only want mean fit.
		subInfo['fit'] = 'mean'

		if obj == 'ISI':
			err = epo.getISI(spikeIdx, dt=infoDict['data']['dt'],
				**subInfo)

			if verbose > 2:
				print(f"ISI = {err:.4g}ms (FR = {1/err:.4g}Hz)")

		elif obj == 'Amp':
			err = epo.getSpikeAmp(spikeIdx, spikeVals,
				dt=infoDict['data']['dt'], **subInfo)

			if verbose > 2:
				print(f"Amp = {err:.4g}mV")

		elif obj == 'PSD':
			err = epo.getPSD(data, spikeIdx,
				dt=infoDict['data']['dt'], **subInfo)

			if verbose > 2:
				print(f"PSD = {err:.4g}mV")

		else:
			continue

		try:
			_ = dataFeat[obj]
		except KeyError:
			dataFeat[obj] = {}

		try:
			_ = dataFeat[obj]['hold']
		except KeyError:
			dataFeat[obj]['hold'] = {}

		if key is not None:
			key = utl.force_pos_int(key, name='epu.getConstHoldFeatures.key',
				zero_ok=True, verbose=verbose)
		else:
			key = list(out[obj]['hold'].keys()).sort().pop() + 1

		dataFeat[obj]['hold'][key] = deepcopy(err)

	return dataFeat.copy()


################################################################################
################################################################################
##
##		Electrophysiology Data Processing Methods
##
################################################################################
################################################################################


################################################################################
## Get Indices, Current of Depolarization Step
################################################################################
def getDepolIdx(data, hdr, protocol=None, verbose=0):
	"""getDepolIdx(data, hdr, protocol=None):

	Get the section of data, indices, and input current corresponding to a
	depolarization step(s) protocol.

	INPUTS:
	=======
	data 		(ndarray)		Array containing ephys data to be parsed.

	hdr 		(dict)			Header dictionary corresponding to data.

	protocol 	(int)			(Default: None) integer key indicating the 
								experimental protocol used to generate data.
								Will be inferred from header if not indicated.

	verbose 	(int)			(Default: 0) Flag for verbosity of method.
								Default is lowest verbosity.

	OUTPUTS:
	=========
	data 		(ndarray)		Truncated data array corresponding to depol 
								step(s) region.

	dpIdx 		(tuple)			Tuple of indices indicating where in the data
								array the depol step(s) begin and end.

	dpI 		(float, list)	Input current of depol step(s).  If multiple
								steps, gives list of currents.
	"""

	verbose = utl.force_pos_int(verbose, name='epu.getDepolIdx.verbose',
		zero_ok=True)

	if protocol is None:
		protocol = getExpProtocol(hdr)
	else:
		protocol = utl.force_int(protocol, name='epu.getHyperpolIdx.verbose',
			verbose=verbose)

	uDACChan = hdr['nActiveDACChannel']

	epochIdx = getEpochIdx(hdr)

	## If not known protocol, raise error
	if protocol < 0:
		err_str = f"Invalid value for keyword 'protocol'... {protocol} "
		err_str += "is not a known ephys protocol."
		raise ValueError(err_str)

	elif protocol == EPHYS_PROT_DEPOLSTEP:

		err_str = f"Input argument 'data' must have 1 dims, got {data.ndim}."
		assert data.ndim == 1, err_str

		if len(np.unique(epochIdx, axis=0)) == 1:
			epochIdx = epochIdx[0]
		else:
			err_str = "ERROR: Multiple epoch protocols for DEPOL STEP PROTOCOL!"
			raise ValueError(err_str)

		holdCurr = abf.GetHoldingLevel(hdr, uDACChan, 1)

		epochLevs = hdr['fEpochInitLevel'][uDACChan]
		epochLevs = epochLevs[(hdr['nEpochType'][uDACChan]).nonzero()]

		if len(np.unique(epochLevs)) == 2:
			dpEpchIdx = (epochLevs != holdCurr).nonzero()[0]

			dpI = hdr['fEpochInitLevel'][uDACChan][dpEpchIdx]

		else:
			epochIncs = hdr['fEpochLevelInc'][uDACChan]
			epochIncs =epochIncs[(hdr['nEpochType'][uDACChan]).nonzero()]

			dpEpchIdx = (epochIdx).nonzero()[0]

			dpI = hdr['fEpochInitLevel'][uDACChan][dpEpchIdx]
			dpI += hdr['fEpochLevelInc'][uDACChan][dpEpchIdx]

		startIdx = int(epochIdx[dpEpchIdx+1])
		endIdx = int(epochIdx[dpEpchIdx+2])
		dpIdx = (startIdx, endIdx)

		data = data[startIdx:endIdx]

		return data, dpIdx, dpI

	elif protocol == EPHYS_PROT_DEPOLSTEPS:

		err_str = f"Input argument 'data' must have 2 dims, got {data.ndim}."
		assert data.ndim == 2, err_str

		epochLevs = hdr['fEpochInitLevel'][uDACChan]
		epochLevs = epochLevs[(hdr['nEpochType'][uDACChan]).nonzero()]

		if len(np.unique(epochLevs)) == 2:
			dpEpchIdx = (epochLevs != holdCurr).nonzero()[0]

		else:
			epochIncs = hdr['fEpochLevelInc'][uDACChan]
			epochIncs =epochIncs[(hdr['nEpochType'][uDACChan]).nonzero()]
			dpEpchIdx = (epochIncs).nonzero()[0]


		if len(np.unique(epochIdx, axis=0)) == 1:
			epochIdx = epochIdx[0]

			startIdx = int(epochIdx[dpEpchIdx+1])
			endIdx = int(epochIdx[dpEpchIdx+2])
			dpIdx = (startIdx, endIdx)

			data = data[startIdx:endIdx]

			dpI = []
			for epNo in range(hdr['lActualEpisodes']):

				tmpI = hdr['fEpochInitLevel'][uDACChan][dpEpchIdx]
				tmpI += epNo*hdr['fEpochLevelInc'][uDACChan][dpEpchIdx]

				dpI.append(tmpI)

		else:
			dpData, dpIdx, dpI = [], [], []

			for epNo in range(hdr['lActualEpisodes']):

				startIdx = int(epochIdx[epNo, dpEpchIdx+1])
				endIdx = int(epochIdx[epNo, dpEpchIdx+2])
				dpIdx.append((startIdx, endIdx))

				dpData.append(data[startIdx:endIdx, epNo])

				tmpI = hdr['fEpochInitLevel'][uDACChan][dpEpchIdx]
				tmpI += epNo*hdr['fEpochLevelInc'][uDACChan][dpEpchIdx]

				dpI.append(tmpI)

			try:
				data = np.array(dpData).astype(float)
			except:
				err_str = f"Error coercing dpData into np.ndarray... Probably "
				err_str += "should not allow episodes of different lengths!"
				raise ValueError(err_str)

		return data, dpIdx, np.array(dpI).squeeze()

	else:
		err_str = f"Invalid value for keyword 'protocol'... Protocol={protocol}"
		err_str += " is not allowed (only expect depol step(s))."
		raise ValueError(err_str)


################################################################################
## Get Indices, Current of Hyperpolarization Step
################################################################################
def getHyperpolIdx(data, hdr, protocol=None, verbose=0):
	"""getHyperpolIdx(data, hdr, protocol=None):

	Get the section of data, indices, and input current corresponding to a
	hyperpolarization step(s) protocol.

	INPUTS:
	=======
	data 		(ndarray)		Array containing ephys data to be parsed.

	hdr 		(dict)			Header dictionary corresponding to data.

	protocol 	(int)			(Default: None) integer key indicating the 
								experimental protocol used to generate data.
								Will be inferred from header if not indicated.

	verbose 	(int)			(Default: 0) Flag for verbosity of method.
								Default is lowest verbosity.

	OUTPUTS:
	=========
	data 		(ndarray)		Truncated data array corresponding to hyperpol 
								step(s) region.

	hpIdx 		(tuple)			Tuple of indices indicating where in the data
								array the hyperpol step(s) begin and end.

	hpI 		(float, list)	Input current of hyperpol step(s).  If multiple
								steps, gives list of currents.
	"""

	verbose = utl.force_pos_int(verbose, name='epu.getHyperpolIdx.verbose',
		zero_ok=True)

	if protocol is None:
		protocol = getExpProtocol(hdr)
	else:
		protocol = utl.force_int(protocol, name='epu.getHyperpolIdx.verbose',
			verbose=verbose)

	epochIdx = getEpochIdx(hdr)

	uDACChan = hdr['nActiveDACChannel']

	## If not known protocol, raise error
	if protocol < 0:
		err_str = f"Invalid value for keyword 'protocol'... {protocol} "
		err_str += "is not a known ephys protocol."
		raise ValueError(err_str)

	elif protocol == EPHYS_PROT_HYPERPOLSTEP:

		if len(np.unique(epochIdx, axis=0)) == 1:
			epochIdx = epochIdx[0]
		else:
			err_str = "ERROR: Multiple epoch protocols for HYPERPOL STEP "
			err_str += "PROTOCOL!"
			raise ValueError(err_str)


		holdCurr = abf.GetHoldingLevel(hdr, uDACChan, 1)

		epochLevs = hdr['fEpochInitLevel'][uDACChan]
		epochLevs = epochLevs[(hdr['nEpochType'][uDACChan]).nonzero()]

		if len(np.unique(epochLevs)) == 2:
			hpEpchIdx = (epochLevs != holdCurr).nonzero()[0]

			hpI = hdr['fEpochInitLevel'][uDACChan][hpEpchIdx]

		else:
			epochIncs = hdr['fEpochLevelInc'][uDACChan]
			epochIncs =epochIncs[(hdr['nEpochType'][uDACChan]).nonzero()]

			hpEpchIdx = (epochIdx).nonzero()[0]

			hpI = hdr['fEpochInitLevel'][uDACChan][hpEpchIdx]
			hpI += hdr['fEpochLevelInc'][uDACChan][hpEpchIdx]

		startIdx = int(epochIdx[hpEpchIdx+1])
		endIdx = int(epochIdx[hpEpchIdx+2])
		hpIdx = (startIdx, endIdx)

		data = data[startIdx:endIdx]

		return data, hpIdx, hpI


	elif protocol == EPHYS_PROT_HYPERPOLSTEPS:

		err_str = f"Input argument 'data' must have 2 dims, got {data.ndim}."
		assert data.ndim == 2, err_str

		epochLevs = hdr['fEpochInitLevel'][uDACChan]
		epochLevs = epochLevs[(hdr['nEpochType'][uDACChan]).nonzero()]

		if len(np.unique(epochLevs)) == 2:
			hpEpchIdx = (epochLevs != holdCurr).nonzero()[0]

		else:
			epochIncs = hdr['fEpochLevelInc'][uDACChan]
			epochIncs =epochIncs[(hdr['nEpochType'][uDACChan]).nonzero()]

			hpEpchIdx = (epochIncs).nonzero()[0]

		if len(np.unique(epochIdx, axis=0)) == 1:
			epochIdx = epochIdx[0]

			startIdx = int(epochIdx[hpEpchIdx+1])
			endIdx = int(epochIdx[hpEpchIdx+2])
			hpIdx = (startIdx, endIdx)

			data = data[startIdx:endIdx]

			hpI = []
			for epNo in range(hdr['lActualEpisodes']):

				tmpI = hdr['fEpochInitLevel'][uDACChan][hpEpchIdx]
				tmpI += epNo*hdr['fEpochLevelInc'][uDACChan][hpEpchIdx]

				hpI.append(tmpI)

		else:
			hpData, hpIdx, hpI = [], [], []

			for epNo in range(hdr['lActualEpisodes']):

				startIdx = int(epochIdx[epNo, hpEpchIdx+1])
				endIdx = int(epochIdx[epNo, hpEpchIdx+2])
				hpIdx.append((startIdx, endIdx))

				hpData.append(data[startIdx:endIdx, epNo])

				tmpI = hdr['fEpochInitLevel'][uDACChan][hpEpchIdx]
				tmpI += epNo*hdr['fEpochLevelInc'][uDACChan][hpEpchIdx]

				hpI.append(tmpI)

			try:
				data = np.array(hpData).astype(float)
			except:
				err_str = f"Error coercing hpData into np.ndarray... Probably "
				err_str += "should not allow episodes of different lengths!"
				raise ValueError(err_str)

		return data, hpIdx, np.array(hpI).squeeze()

	else:
		err_str = f"Invalid value for keyword 'protocol'... Protocol={protocol}"
		err_str += " is not allowed (only expect depol step(s))."
		raise ValueError(err_str)


################################################################################
## Get Rolling Percentile of Data
################################################################################
def getRollPerc(data, window=100, perc=50., verbose=0, edgeCorrect=True):

	############################################################################
	##	Check Inputs, Keyword Arguments
	############################################################################
	if True:

		## Check the verbosity keyword
		verbose = utl.force_pos_int(verbose, name='epu.getRollPerc.verbose', 
			zero_ok=True, verbose=verbose)

		## Check the type and shape of the data
		data = utl.force_float_arr(data, name='epu.getRollPerc.data',
			verbose=verbose).squeeze()

		err_str = "Input argument 'data' must be 1D array."
		err_str += f" (data.ndim = {data.ndim})"
		assert data.ndim == 1, err_str

		## Check that 'window' is an integer and is odd
		window = utl.force_pos_int(window, name='epu.getRollPerc.window',
			verbose=verbose)
		window = window + 1 if ((window % 2) == 0) else window

		perc = utl.force_pos_float(perc, name="epu.getRollPerc.perc",
			verbose=verbose)
		errStr = "Keyword argument 'perc' must be a percentage (0, 100)."
		assert perc < 100, errStr

		errStr = "Keyword argument 'edgeCorrect' must be a boolean."
		assert isinstance(edgeCorrect, bool), errStr

	############################################################################
	##	Calculate Percentile
	############################################################################
		order = int(window*perc/100.)

		## Get the rolling median
		medData = sig.order_filter(data, np.ones(window), order)

		if edgeCorrect:
			## Edge correct the median
			windArr = np.arange(window).astype(int)
			oddArr = (windArr + windArr%2. + 1).astype(int)
			leftEnd, rightEnd = [], []
			counter = 0
			for (ii, wd) in zip(windArr, oddArr):

				if verbose >= 3:
					if (counter + 1) % 20 == 0.:
						print(f"{counter+1}/{len(windArr)}: {ii}, {wd}")
				
				leftEnd.append(sig.order_filter(data[:window*2], np.ones(wd),
					int((wd-1)/2))[ii])

				wd = oddArr[-1]-wd+1
				rightEnd.append(sig.order_filter(data[-window*2-1:],
					np.ones(wd), int((wd-1)/2))[-(window-ii)-1])

				counter += 1

			medData[:window] = np.array(leftEnd)
			medData[-window:] = np.array(rightEnd)

		## Get rolling percentile array
		return medData


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

		lb = [
			min(-150, data[0]-20),
			min(-150, data.min() - 30),
			0
		]

		ub = [
			max(100, data[0]+20),
			max(100, data.max() + 30),
			np.inf
		]

		## Fit the curve with some reasonable bounds
		params, cov = curve_fit(offsetExp, times, data, p0=p0, bounds=(lb, ub))
		cov = np.diag(cov)

	## If something went wrong, return a really bad result
	except:
		params = [-150, -150, 0]
		cov = [100, 100, 100]

	## If needed, return everything
	if returnAll:
		return np.array(params).astype(float), np.array(cov).astype(float)
	else:
		return np.array(params).astype(float)


################################################################################
## Offset Exponenital Function
################################################################################
def offsetExp(t, V0, VInf, tau):
	return VInf + (V0-VInf)*np.exp(-t/tau)
"""
================================================================================
	Evolutionary Algorithm Runfile Processing Utility
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, February 20, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains utility functions for reading and checking EA runfiles.

	For example, we will want to have a function that verifies that all the
	entries in a runfile are legal and that any folders/files/functions exist 
	and are functional.  We will also have a function that loads the runfile
	from a json to a dictionary.

================================================================================
================================================================================
"""
from copy import deepcopy
import json
import numpy as np
import os
import pickle as pkl

import Utility.utility as utl

################################################################################
#	Get Info Dict from File Path
################################################################################
def getInfo(infoPath, verbose=0):

	## Check verbose keyword
	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	## Check that infoPath is a valid path
	err_str = f"Invalid argument 'infoPath': {infoPath} is not a directory."
	assert os.path.isdir(infoPath), err_str

	## Load info.json
	if verbose:
		print_str = f"Loading info from {infoPath}"
		print(print_str)
	with open(os.path.join(infoPath, "info.json"), "r") as f:
		infoDict = json.load(f)

	## Check the entries of infoDict
	infoDict = checkInfo(infoDict, verbose=verbose)

	## Set infoDir field
	infoDict['infoDir'] = deepcopy(infoPath)

	return infoDict


################################################################################
#	Check Info Dict Fields
################################################################################
def checkInfo(infoDict, verbose=0, isdefault=False):

	## Check verbose keyword
	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	if verbose > 1:
		print_str = "\n...Checking entries of 'infoDict'!"
		print(print_str)

	## Make sure that infoDict is a *dictionary*
	err_str = "Argument 'info' must be a dictionary.  (Use runfile_"
	err_str += "util.getInfo(infoPath) to load.)"
	assert isinstance(infoDict, dict), err_str

	## Check directory fields of infoDict
	infoDict = _checkInfoDirs(infoDict, verbose=verbose)

	## Check runtime parameter fields of info Dict
	infoDict = _checkRuntimeParams(infoDict, verbose=verbose)

	## Check individual parameter fields of info dict
	infoDict = _checkIndParams(infoDict, verbose=verbose)

	## Check population parameter fields of info dict
	infoDict = _checkPopParams(infoDict, verbose=verbose)

	## Check mutation parameter fields of dict
	infoDict = _checkMutParams(infoDict, verbose=verbose)

	## Check crossover parameter fields of dict
	infoDict = _checkXOverParams(infoDict, verbose=verbose)

	## Check simulation protocol parameter fields of dict
	infoDict = _checkSimParams(infoDict, verbose=verbose)

	## Check SimProt, Objs, Model

	## Fill in all missing fields with defaults from modelDir/info.json
	if not isdefault:
		if verbose > 1:
			print("\nChecking and loading defaults for missing entries!")
		infoDict = getDefaults(infoDict, verbose=0)

	return infoDict


################################################################################
#	Check Directory Fields
################################################################################
def _checkInfoDirs(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking basic 'infoDict' directories..."
		print(print_str)

	## Iterate through the required directories
	basicDirs = ['modelDir', 'logDir', 'cpDir']
	for bDir in basicDirs:
		if verbose > 1:
			print(f"Checking info['{bDir}'] =\t{infoDict[bDir]}...")
		err_str =f"info['{bDir}'] = {infoDict[bDir]} is not a directory!"
		assert os.path.isdir(infoDict[bDir]), err_str

	## Check modelDir to make sure it has the required files
	if verbose > 1:
		print(f"\nChecking that 'modelDir' has correct files!")
	reqFiles = ['model_Ind.py', ## Ind subclass with model-specific methods
				'model.py',		## Actual ODE functions to implement model
				'info.json', 	## Default info dict for model
				'model.json'] 	## Model parameters

	modelFiles = os.listdir(infoDict['modelDir'])

	for reqFile in reqFiles:
		if verbose > 1:
			print(f"Checking that {reqFile} is in modelDir...")
		err_str = f"{reqFile} is not in modelDir = {infoDict['modelDir']}!"
		assert reqFile in modelFiles, err_str

	return deepcopy(infoDict)


################################################################################
#	Check Runtime Parameter Fields
################################################################################
def _checkRuntimeParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Runtime Parameters in 'infoDict'!"
		print(print_str)

	## Check that runtime parameters, 'EA', exists in infoDict
	try:
		infoDict['ind']
	except KeyError:
		## If it doesn't, just return the infoDict
		infoDict['ind'] = {}
		return deepcopy(infoDict)

	rtInts = ['NGen', 	## Number of generations to run EA
			  'cpFreq']	## Frequency (in generations) to save checkpoints

	## Check integer keys
	for rtP in rtInts:
		if verbose > 1:
			print(f"Checking infoDict[{rtP}]...")

		try:
			infoDict['EA'][rtP] = utl.force_pos_int(infoDict['EA'][rtP],
				name=f"infoDict['EA']['{rtP}']",
				verbose=verbose, zero_ok=True)
		except:
			err_str = f"Invalid entry for 'infoDict['EA']['{rtP}']'!"
			raise ValueError(err_str)

	rtBools = ["archive_logs",		## Flag whether to archive logs
			   "remove_old_logs",	## Flag whether to remove old logs
			   "archive_cps",		## Flag whether to archive cps
			   "remove_old_cps"]	## Flag whether to remove old cps

	## Check that these entries are boolean
	for rtP in rtBools:
		if verbose > 1:
			print(f"Checking infoDict['EA']['{rtP}']...")

		err_str = f"Invalid entry for 'infoDict['EA']['{rtP}']'; must be bool"
		assert isinstance(infoDict['EA'][rtP], bool), err_str

	return deepcopy(infoDict)


################################################################################
#	Check Individual Parameter Fields
################################################################################
def _checkIndParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Individual Parameters in 'infoDict'!"
		print(print_str)

	## Make sure that the 'Indvidual' field in infoDict exists
	try:
		infoDict['ind']
	except KeyError:
		infoDict['ind'] = {}
		return deepcopy(infoDict)

	## Check that 'ind.verbose' is a positive integer.
	try:
		tmp = infoDict['ind']['verbose']
		tmp = utl.force_pos_int(tmp, name="infoDict['ind']['verbose']",
			zero_ok=True, verbose=verbose)
		infoDict['ind']['verbose'] = tmp
	except KeyError:
		pass
	except AssertionError:
		del infoDict['ind']['verbose']

	try:
		tmp = infoDict['ind']['randomInit']

		err_str = "infoDict['ind']['randomInit'] must be boolean!"
		assert isinstance(tmp, bool), err_str
	except KeyError:
		pass
	except AssertionError as err:
		if verbose > 1:
			print(err.args[0])
			print(" ... deleting")
		del infoDict['ind']['randomInit']


	return deepcopy(infoDict)


################################################################################
#	Check Population Parameter Fields
################################################################################
def _checkPopParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Population Parameters in 'infoDict'!"
		print(print_str)

	## Make sure that the 'Population' field in infoDict exists
	try:
		infoDict['pop']
	except KeyError:
		infoDict['pop'] = {}
		return deepcopy(infoDict)

	ppInts = ['NInd', 		## Number Individuals with which to init Pop
			  'maxInd',		## Maximum number of Individuals allowed in Pop
			  'verbose']	## Verbosity of Population obj

	## Check integer keys
	for ppP in ppInts:
		if verbose > 1:
			print(f"Checking infoDict[{ppP}]...")

		try:
			infoDict['pop'][ppP] = utl.force_pos_int(infoDict['pop'][ppP],
				name=f"infoDict['pop']['{ppP}']",
				verbose=verbose, zero_ok=True)
		except:
			err_str = f"Invalid entry for 'infoDict['pop']['{ppP}']'!"
			raise ValueError(err_str)

	ppFloats = ["Mut_Prob",		## Probability of Individual Mutating
			    "Cross_Prob"]	## Probability of Individual Crossing

	## Check that these entries are float
	for ppP in ppFloats:
		if verbose > 1:
			print(f"Checking infoDict['pop']['{ppP}']...")
		
		err_str = f"Invalid entry for 'infoDict['pop']['{ppP}']'!"
		
		try:
			utl.force_pos_float(infoDict['pop'][ppP],
				name=f"infoDict['pop']['{ppP}']",
				verbose=verbose, zero_ok=True)
		except:
			raise ValueError(err_str)

		assert infoDict['pop'][ppP] <= 1, err_str

	return deepcopy(infoDict)


################################################################################
#	Check Mutation Parameter Fields
################################################################################
def _checkMutParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Mutation Parameters in 'infoDict'!"
		print(print_str)

	## Make sure that the 'Mutation' field in infoDict exists
	try:
		infoDict['mutation']
	except KeyError:
		infoDict['mutation'] = {}
		return deepcopy(infoDict)

	## Check method keyword
	try:
		method = infoDict['mutation']['method']
	except:
		if verbose > 0:
			warn_str = "WARNING: no mutation method specified... using default."
			print(warn_str)
		infoDict['mutation'] = {}

		return deepcopy(infoDict)

	if method == 'normal':
		try:
			NMut = infoDict['mutation']['NMut']
			NMut = utl.force_pos_float(NMut,
				name="infoDict['mutation']['NMut']", verbose=verbose)
		except KeyError:
			pass
		except AssertionError:
			del infoDict['mutation']['NMut']

	elif method == 'polynomial':
		try:
			eta = infoDict['mutation']['eta']
			eta = utl.force_pos_float(eta, name="infoDict['mutation']['eta']",
				verbose=verbose)
		except KeyError:
			pass
		except AssertionError:
			del infoDict['mutation']['eta']

	else:
		err_str = "Unrecognized value for mutation method..."
		raise ValueError(err_str)

	return deepcopy(infoDict)


################################################################################
#	Check Crossover Parameter Fields
################################################################################
def _checkXOverParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Crossover Parameters in 'infoDict'!"
		print(print_str)

	## Make sure that the 'Crossover' field in infoDict exists
	try:
		infoDict['crossover']
	except KeyError:
		infoDict['crossover'] = {}
		return deepcopy(infoDict)

	## Check method keyword
	try:
		method = infoDict['crossover']['method']
	except:
		if verbose > 0:
			warn_str = "WARNING: no crossover method specified... "
			warn_str += "using default."
			print(warn_str)
		infoDict['crossover'] = {}

		return deepcopy(infoDict)

	if method == 'uniform':
		try:
			fracCross = infoDict['mutation']['fracCross']
			fracCross = utl.force_pos_float(fracCross,
				name="infoDict['mutation']['fracCross']", verbose=verbose)
		except KeyError:
			pass
		except AssertionError:
			del infoDict['mutation']['fracCross']

	elif method == 'simBinary':
		try:
			eta = infoDict['mutation']['eta']
			eta = utl.force_pos_float(eta, name="infoDict['mutation']['eta']",
				verbose=verbose)
		except KeyError:
			pass
		except AssertionError:
			del infoDict['mutation']['eta']

	else:
		err_str = "Unrecognized value for mutation method..."
		raise ValueError(err_str)

	return deepcopy(infoDict)


################################################################################
#	Check Simulation Protocol Parameter Fields
################################################################################
def _checkSimParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Simulation Protocol Parameters in 'infoDict'!"
		print(print_str)

	## Make sure that the 'simulation' field in infoDict exists
	try:
		infoDict['simulation']
	except KeyError:
		infoDict['simulation'] = {}
		return deepcopy(infoDict)

	## Check whether to test holding current
	try:
		assert isinstance(infoDict['simulation']['holdCheck'], bool)
	except KeyError:
		pass
	except AssertionError:
		err_str = "infoDict['simulation']['holdCheck'] must be boolean!"
		raise ValueError(err_str)

	## Check whether to reset initial conditions to holding or rest equilibrium
	try:
		equil = infoDict['simulation']['equil']
	except:
		equil = "rest"
		pass

	err_str = "Invalid value for infoDict['simulation']['equil'] (only 'rest'"
	err_str += " and 'holding' are allowed)."
	assert equil in ["rest", "holding"], err_str

	## Check that subprotocol duration is a positive float
	try:
		SPD = infoDict['simulation']['subProtDur']
		infoDict['simulation']['subProtDur'] = utl.force_pos_float(SPD,
			name="infoDict['simulation']['subProtDur']", verbose=verbose)
	except KeyError:
		pass

	## Check that time step 'dt' is a positive float
	try:
		dt = infoDict['simulation']['dt']
		infoDict['simulation']['dt'] = utl.force_pos_float(dt,
			name="infoDict['simulation']['dt']", verbose=verbose)
	except KeyError:
		pass

	## Check that the number of times to print simulations to screen is pos int
	try:
		NP = infoDict['simulation']['NPrint']
		infoDict['simulation']['NPrint'] = utl.force_pos_int(NP,
			name="infoDict['simulation']['NPrint']", verbose=verbose,
			zero_ok=True)
	except KeyError:
		pass

	return deepcopy(infoDict)


################################################################################
#	Check Objectives Parameter Fields
################################################################################
def _checkObjParams(infoDict, verbose=0):

	if verbose > 1:
		print_str = "\nChecking Objectives Parameters in 'infoDict'!"
		print(print_str)

	## Make sure that the 'objectives' field in infoDict exists
	try:
		infoDict['objectives']
	except KeyError:
		infoDict['objectives'] = {}
		return deepcopy(infoDict)

	## Maybe finish this later...

	# "objectives":{
	# 	"Spikes":{
	# 		"exact":true
	# 	},
	# 	"ISI":{
	# 		"depol":"thirds",
	# 		"kwds":{}
	# 	},
	# 	"Amp":{
	# 		"fit":"exp",
	# 		"kwds":{"verbose":2}
	# 	},
	# 	"PSD":{
	# 		"fit":"exp",
	# 		"kwds":{}
	# 	},
	# 	"RI":{
	# 		"type":"RI",
	# 		"kwds":{}
	# 	},
	# 	"kDist":{
	# 		"type":"rest",
	# 		"sigma":[2, 10],
	# 		"kwds":{}

################################################################################
#	Check and Compare Nested Dictionaries (Replace dict1 with dict2 when empty)
################################################################################
def compareKeys(dict1, dict2, verbose=0):

	assert isinstance(dict1, dict), "Argument 'dict1' must be a dictionary!"
	if not isinstance(dict2, dict):
		dict2 = {}

	for key in dict1:
		if isinstance(dict1[key], dict):
			try:
				foo, bar = compareKeys(dict1[key], dict2[key], verbose=verbose)
			except KeyError:
				foo, bar = compareKeys(dict1[key], {}, verbose=verbose)
			dict2[key] = deepcopy(bar)

		else:
			try:
				_ = dict2[key]
			except KeyError:
				dict2[key] = deepcopy(dict1[key])

	return dict1, dict2


################################################################################
#	Get Defaults, Fill in infoDict with defaults.
################################################################################
def getDefaults(infoDict, verbose=0):

	## Check verbose keyword
	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	## Check that modelPath is a valid path
	modelPath = infoDict['modelDir']
	err_str = f"Invalid argument 'modelPath': {modelPath} is not a directory."
	assert os.path.isdir(modelPath), err_str

	## Load ./ModelDir/info.json
	if verbose:
		print_str = f"Loading info from {modelPath}"
		print(print_str)
	with open(os.path.join(modelPath, "info.json"), "r") as f:
		defDict = json.load(f)

	## Check the entries of defDict
	defDict = checkInfo(defDict, verbose=verbose, isdefault=True)

	## Compare and set empty keys in infoDict with defaults
	defDict, infoDict = compareKeys(defDict, infoDict, verbose=verbose)

	return infoDict








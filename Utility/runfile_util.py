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
def checkInfo(infoDict, verbose=0):

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

	## Check Ind, Pop, Mut/Xover, Archive, SimProt, Objs, Model

	## Fill in all missing fields with defaults from modelDir/info.json
	infoDict = getDefaults(infoDict, verbose=verbose)

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

	rtInts = ['NGen', 	## Number of generations to run EA
			  'cpFreq']	## Frequency (in generations) to save checkpoints

	for rtP in rtInts:
		if verbose > 1:
			print(f"Checking infoDict[{rtP}]...")

		try:
			utl.force_pos_int(infoDict['EA'][rtP],
				name=f"infoDict['EA']['{rtP}']",
				verbose=verbose, zero_ok=True)
		except:
			err_str = f"Invalid entry for 'infoDict['EA']['{rtP}']'!"
			raise ValueError(err_str)

	rtBools = ["archive_logs",
			   "remove_old_logs",
			   "archive_cps",
			   "remove_old_cps"]

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

	return deepcopy(infoDict)




def getDefaults(infoDict, verbose=0):

	return infoDict








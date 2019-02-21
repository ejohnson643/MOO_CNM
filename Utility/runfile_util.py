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


def getInfo(infoPath, verbose=0):

	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	err_str = f"Invalid argument 'infoPath': {infoPath} is not a directory."
	assert os.path.isdir(infoPath), err_str

	if verbose:
		print_str = f"Loading info from {infoPath}"
		print(print_str)
	with open(os.path.join(infoPath, "info.json"), "r") as f:
		info = json.load(f)

	try:
		tmp = deepcopy(info['infoDir'])
		assert tmp == infoPath
	except:
		warn_str = f"\n\nWARNING: info['infoDir'] != infoPath, using infoPath\n"
		if verbose:
			print(warn_str)
		info['infoDir'] = deepcopy(infoPath)

	info = checkInfo(info, verbose=verbose)

	return info


def checkInfo(info, verbose=0):

	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	if verbose > 1:
		print_str = "Checking entries of 'info'!"
		print(print_str)

	err_str = "Argument 'info' must be a dictionary.  (Use runfile_"
	err_str += "util(infoPath) to load.)"
	assert isinstance(info, dict), err_str

	info = checkInfoDirs(info, verbose=verbose)

	info = checkRuntimeParams(info, verbose=verbose)

	return info


def checkInfoDirs(info, verbose=0):

	if verbose > 1:
		print_str = "Checking basic 'info' directories..."
		print(print_str)

	basicDirs = ['infoDir', 'modelDir', 'logDir', 'cpDir']
	for bDir in basicDirs:
		if verbose > 1:
			print(f"Checking info['{bDir}'] =\t{info[bDir]}...")
		err_str =f"info['{bDir}'] = {info[bDir]} is not a directory!"
		assert os.path.isdir(info[bDir]), err_str

	if verbose > 1:
		print(f"Checking that 'modelDir' has correct files!")

	reqFiles = ['Individual.py', 'model.py', 'info.json', 'model_util.py']

	modelFiles = os.listdir(info['modelDir'])

	for reqFile in reqFiles:
		if verbose > 1:
			print(f"Checking that {reqFile} is in modelDir...")
		err_str = f"{reqFile} is not in modelDir = {info['modelDir']}!"
		assert reqFile in modelFiles, err_str

	return deepcopy(info)


def checkRuntimeParams(info, verbose=0):

	if verbose > 1:
		print_str = "Checking Runtime Parameters in 'info'!"

	reqParams = ['NGen', 'checkpoint_freq']
	defParams = [1, 1]

	for reqP, defP in zip(reqParams, defParams):
		if verbose > 1:
			print(f"Checking info[{reqP}]...")

		try:
			utl.force_pos_int(info[reqP], name=f"info[{reqP}]", verbose=verbose)
		except:
			warn_str = f"WARNING: invalid entry for info[{reqP}], setting to "
			warn_str += f"default: {defP}"
			if verbose:
				print(warn_str)
			info[reqP] = defP

	return deepcopy(info)












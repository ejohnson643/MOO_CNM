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

	with open(os.path.join(infoPath, "info.json"), "r") as f:
		info = json.load(f)

	return info
"""
================================================================================
	Evolutionary Algorithm Model Utility Functions
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, February 20, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains utility functions for loading and implementing neuron
	models.

	For example, we will want to have a function that verifies that all the
	entries in a modelDict are legal and that model functions and parameters
	exist and are legal.  We will also have a function that loads the modelDict
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


def check_model_dict(modelDict):

	req_entries = ['model', 'name', 'inits', 'coldict',
		"param_defaults", "param_min", "param_max"]

	for req in req_entries:
		err_str = ""

	return
"""
================================================================================
	Evolutionary Algorithm Utility Functions
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, February 20, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains some "utility" functions that will be useful in 
	implementing an Evolutionary Algorithm (EA) for model optimization.

================================================================================
================================================================================
"""

from copy import deepcopy
import numpy as np
import os
import re
import time


def force_string(value, name=None, verbose=0):

	if value is not None:
		if name is not None:
			err_str = f"{name} = {value} is not a string!"
		else:
			err_str = f"var = {value} is not a string!"

		assert isinstance(value, str), err_str

	return value


def is_floatable(value, name=None, verbose=0):

	if (name is not None) and (verbose > 1):
		print(f"Checking if {name} = {value} is floatable...")

	try:
		_ = float(value)
		if verbose:
			if name is not None:
				print(f"{name} = {value} is floatable!")
			else:
				print(f"var = {value} is floatable")
		return True

	except ValueError:
		if verbose:
			if name is not None:
				print(f"{name} = {value} is not floatable!")
			else:
				print(f"var = {value} is not floatable")
		return False


def force_float(value, name=None, verbose=0):

	if name is not None:
		err_str = f"{name} = {value} is not floatable!"
	else:
		err_str = f"var = {value} is not floatable!"
	assert is_floatable(value, name=name, verbose=verbose), err_str

	if verbose > 1:
		if name is not None:
			print_str = f"Setting {name} = {value} to float!"
		else:
			print_str = f"Setting var = {value} to float!"
		print(print_str)

	return float(value)


def force_pos_float(value, name=None, verbose=0, zero_ok=False):

	value = force_float(value, name=name, verbose=verbose)

	if name is not None:
		err_str = f"{name} = {value} is not positive!"
	else:
		err_str = f"var = {value} is not positive!"

	if zero_ok:
		assert value >= 0, err_str
	else:
		assert value > 0, err_str

	return value


def force_int(value, name=None, verbose=0):

	if name is not None:
		err_str = f"{name} = {value} is not floatable!"
	else:
		err_str = f"var = {value} is not floatable!"
	assert is_floatable(value, name=name, verbose=verbose), err_str

	if verbose > 1:
		if name is not None:
			print_str = f"Setting {name} = {value} to int!"
		else:
			print_str = f"Setting var = {value} to int!"
		print(print_str)
		
	return int(value)


def force_pos_int(value, name=None, verbose=0, zero_ok=False):

	value = force_int(value, name=name, verbose=verbose)

	if name is not None:
		err_str = f"{name} = {value} is not positive!"
	else:
		err_str = f"var = {value} is not positive!"

	if zero_ok:
		assert value >= 0, err_str
	else:
		assert value > 0, err_str

	return value


"""
================================================================================
	(Electrophysiology) Data Loading Utility
================================================================================

	Author: Eric Johnson
	Date Created: April 10, 2018
	Date Updated: Wednesday, March 6, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains utility functions for loading (electrophysiology) data.

	At this point, we'll be assuming that the data is ABF electrophys data that
	is (assumed to be) stored in 
	
	HURON: "../../"	<- FIX THIS
	Laptop: "../../RTG_Project/EA_Code/Data/"
	

================================================================================
================================================================================
"""
from copy import deepcopy
import datetime
import numpy as np
import os
import re
import time

import Utility.ABF_util as abf
import Utility.runfile_util as rfu
import Utility.utility as utl


def load_data(dataInfo, dataDir="", laptop=False, verbose=0):
	"""load_data(dataInfo, dataDir="", laptop=False, verbose=0)

	This function takes in the infoDict['data'] dictionary and returns a
	dictionary of headers and data files.

	To do this, it interprets the dates field, which is assumed to be a 
	dictionary keyed by dates in "DD/MM/YYYY" format whose values are either
	a list of integers corresponding to the abf files from that date, or None,
	indicating that all the files should be loaded.

	Inputs:
	=======
	dataInfo:	(dict)		Dictionary containing (req fields):
							 - 'dates': dictionary keyed by dates, valued by
							   fileNos or None
							 - 'dt': float corresponding to the timestep (sec)
							   of the data (usually 0.001 = 1ms)
	
	Keywords:
	=========
	dataDir:	(string) 	Path to data.  Default depends on 'laptop' kwd.

	laptop:		(bool)		Flag indicating whether the code is on my laptop or 
							on Huron.  Default is False, indicating Huron.

	verbose:	(int)		Flag indicating level of verbosity of the method.

	
	Outputs:
	========
	dataDict:	(dict)		Dictionary keyed by (date, neuronNo, fileNo), where 
							each value is a dictionary containing a 'header' and
							'data'.
	"""

	## Checking that verbose is a positive integer
	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)
	
	## Make sure input is a dictionary
	err_str = "Input argument 'dataInfo' must be a dictionary."
	assert isinstance(dataInfo, dict), err_str

	## Check that there actually are 'dates' in dataInfo dictionary.
	try:
		datesDict = dataInfo['dates']
	except KeyError:
		if verbose:
			warn_str = "\nWARNING: No dates given, so no data loaded!"
			print(warn_str)
		return {}

	## Checking that dataDir is a string
	err_str = "Keyword argument 'dataDir' must be a string!"
	assert isinstance(dataDir, str), err_str

	## Checking that laptop is a boolean
	err_str = "Keyword argument 'laptop' must be a boolean!"
	assert isinstance(laptop, bool), err_str




# def _load_data(date, datadir='', verbose=0, fileNos=None):

# 	if isinstance(datadir, str):
# 		if len(datadir) == 0:
# 			datadir = "../Data/FlourakisData/"
# 	else:
# 		err_str = "Keyword argument 'datadir' must be a string."
# 		raise TypeError(err_str)

# 	if fileNos is not None:
# 		if np.any([not utl._check_pos_int(fNo,zero_ok=True) for fNo in fileNos]):
# 			err_str = "Invalid entry for keyword argument 'fileNos'"
# 			raise ValueError(err_str)

# 	datestr = deepcopy(date)
# 	# if date == "01042011":
# 	# 	date = "04012011"
# 	date = utl._parse_date(date)

# 	date_fmt_str = time.strftime("%A, %B %d, %Y", date.timetuple())
# 	if verbose:
# 		print("Loading Data from %s\n" % date_fmt_str)

# 	datadir = utl._find_date_folder(date, datadir=datadir)
# 	filelist = sorted([f for f in os.listdir(datadir) if '.abf' in f])
# 	if len(filelist) == 0:
# 		err_str = "No .abf files in specified directory."
# 		raise ValueError(err_str)

# 	if not os.path.isdir(os.path.join(datadir,"Figures")):
# 		os.mkdir(os.path.join(datadir,"Figures/"))

# 	filefmt = time.strftime("%Y_%m_%d_\d{4}", date.timetuple())
# 	fileref = re.compile(filefmt)
# 	data = {}
# 	for f in filelist:
# 		if not fileref.match(f):
# 			continue

# 		f_No = int(f[11:15])
# 		if fileNos is not None:
# 			if f_No not in fileNos:
# 				continue
# 		data[f_No] = {}
# 		[d, h] = abf.ABF_read(f, datadir=datadir, verbose=verbose)
# 		data[f_No]['data'] = d
# 		data[f_No]['header'] = h

# 	return data
if __name__ == "__main__":

	dataDir = "../RTG_Project/EA_Code/Data/"

	files = os.listdir(dataDir)

	infoPath = "./Runfiles/HH_Test/"

	infoDict = rfu.getInfo(infoPath, verbose=0)



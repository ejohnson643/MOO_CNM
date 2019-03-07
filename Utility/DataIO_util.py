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
	dataDict:	(dict)		Dictionary keyed by date, then fileNo, where each 
							value is a dictionary containing a 'header' and 
							'data'.
	"""

	## Checking that verbose is a positive integer
	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	if verbose:
		print("\nLoading data given in dataInfo.")

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

	## Check dataDir, get defaults if neeeded
	if len(dataDir) == 0:
		if laptop:
			dataDir = "../RTG_Project/EA_Code/Data/"
		else:
			## Eventually we should move this to ./Data/
			dataDir = "../EA_Code/Data/FlourakisData/2P/"

		if verbose > 1:
			print(f"Using default dataDir = {dataDir}")
	elif verbose > 1:
		print(f"Using input dataDir = {dataDir}")
	
	err_str = f"dataDir = {dataDir} is not a valid directory!"
	assert os.path.isdir(dataDir), err_str

	dataDict = {}

	## Check dates in datesDict
	for date in datesDict:
		fileNos = datesDict[date]
		dateDict = load_data_from_date(date, fileNos, dataDir, verbose)

		dataDict[date] = {}
		for key in dateDict:
			dataDict[date][key] = deepcopy(dateDict[key])

	return dataDict


def load_data_from_date(date, fileNos, dataDir, verbose=0):
	"""load_data_from_date(date, fileNos, dataDir, verbose=0)

	Loads data in dataDir corresponding to a specific date and fileNos
	"""

	## Check that date is a string
	err_str = "Input argument 'date' must be a string."
	assert isinstance(date, str), err_str

	if fileNos is not None:

		## Check that fileNos are a list of integers
		err_str = "Input argument 'fileNos' must be a list."
		assert isinstance(fileNos, list), err_str

		for ii, fNo in enumerate(fileNos):
			fileNos[ii] = utl.force_pos_int(fNo, name=f'fileNos[{ii}]',
				zero_ok=True)

	## Check that 'dataDir' is a string and a valid directory
	err_str = "Input argument 'dataDir' must be a valid path (string)!"
	assert isinstance(dataDir, str), err_str
	assert os.path.isdir(dataDir), err_str

	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	## Get datetime object
	dateTpl = parse_date(date)

	## Get path to data for that date
	datePath = find_date_folder(dateTpl, dataDir, verbose=verbose)

	## Check that there *are* files in the datePath folder
	err_str = f"There are no .abf files in the directory {datePath}"
	assert len([f for f in os.listdir(datePath) if ".abf" in f]) > 0, err_str

	## Make a figures folder, this is a good place to put trajectory plots.
	if not os.path.isdir(os.path.join(datePath, "Figures")):
		os.mkdir(os.path.join(datePath, "Figures"))

	filefmt = dateTpl.strftime("%Y_%m_%d_\d{4}.abf")
	fileref = re.compile(filefmt)

	## Check that all fileNos are in this folder... If some files don't exist, 
	## then warn, if no files match, break
	data = {}
	for f in sorted(os.listdir(datePath)):
		if not fileref.match(f):
			continue

		fNo = int(f[11:15])

		if fileNos is None:
			data[fNo] = {}
			[d, h] = abf.ABF_read(f, datadir=datePath, verbose=verbose)
			data[fNo]['data'] = d
			data[fNo]['header'] = h
		elif fNo in fileNos:
			data[fNo] = {}
			[d, h] = abf.ABF_read(f, datadir=datePath, verbose=verbose)
			data[fNo]['data'] = d
			data[fNo]['header'] = h
	
	## Check that all fileNos are in this folder... If some files don't exist, 
	## then warn, if no files match, break
	if np.any([fNo not in data.keys() for fNo in fileNos]):
		if verbose:
			warn_str = "WARNING: Some requested fileNos were not loaded."
			print(warn_str)

	err_str = "No data were loaded; none corresponded to fileNos = {fileNos}"
	assert len(data) > 0, err_str

	return data


def parse_date(date):
	"""parse_date(date, verbose=0)

	Takes in a string containing a date either in
		DD/MM/YYYY
	or
		DDMMYYYY or MMDDYYYY
	formats.

	Outputs datetime.datetime tuple.
	"""

	## Check that date is a string
	err_str = "Input argument 'date' must be a string"
	assert isinstance(date, str), err_str

	## Try the easy thing...
	try:
		dateref = re.compile("(\d{2})[/.-](\d{2})[/.-](\d{4})$", re.IGNORECASE)
		mat = dateref.match(date)
		if mat is not None:
			return datetime.datetime(*map(int, mat.groups()[-1::-1]))
	except ValueError:
		pass

	## Otherwise, assume that date is a 8-digit number DDMMYYYY or MMDDYYYY
	## (or one of a few other combos)
	err_str = "Input argument 'date' must be in either DD/MM/YYYY, DDMMYYYY, "
	err_str += "or MMDDYYYY format."
	assert len(date) == 8, err_str
	assert utl.is_floatable(date), err_str

	## Find the year
	yearref = re.compile("2009|201[0-6]")
	yearidx = yearref.search(date).span()
	year = int(date[yearidx[0]:yearidx[1]])
	newdate = date[:yearidx[0]] + date[yearidx[1]:]

	## Find the day
	dayref = re.compile("[0-3][0-9]")
	daymatch = dayref.findall(newdate)

	## Find the month
	monthref = re.compile("0[1-9]|1[0-2]")
	monthmatch = monthref.findall(newdate)

	## If there was a match for the month in the second half.
	if len(monthmatch) > 1:
		month = int(newdate[:2])
		day = int(newdate[2:])

	## Otherwise, work even harder...
	else:
		monthidx = monthref.finditer(newdate)
		for monthid in monthidx:
			month = int(newdate[monthid.span()[0]:monthid.span()[1]])
			if monthid.span()[0] == 0:
				day = int(newdate[2:])

	return datetime.datetime(year, month, day)


def find_date_folder(dateTpl, dataDir, verbose=0):
	"""find_date_folder(dateTpl, dataDir, verbose=0)

	Finds folder in dataDir corresponding to dateTpl, which is a datetime tuple
	containing a date.
	"""

	## Checking that verbose is a positive integer
	verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

	## Checking that dateTpl is a datetime object
	if isinstance(dateTpl, str):
		dateTpl = parse_date(dateTpl)
	err_str = "Input argument 'dateTpl' must be a datetime object"
	assert isinstance(dateTpl, datetime.datetime), err_str

	## Check that 'dataDir' is a string and a valid directory
	err_str = "Input argument 'dataDir' must be a valid path (string)!"
	assert isinstance(dataDir, str), err_str
	assert os.path.isdir(dataDir), err_str

	if verbose > 1:
		dateStr = dateTpl.strftime("%A, %B %d, %Y")
		print(f"Locating data from {dateStr}")

	## Format folder name, which is of course MMDDYYYY
	folderName = dateTpl.strftime("%m%d%Y")

	## Look recursively...  First check if it is in dataDir
	if folderName in os.listdir(dataDir):
		dataDir = os.path.join(dataDir, folderName)
		if verbose > 1:
			print(f"Date folder: {dataDir}")
		return dataDir

	## Then lets look for month folders...
	monthfmt = dateTpl.strftime("%b %y|%B %y|%b %Y|%B %Y")
	monthref = re.compile(monthfmt, re.IGNORECASE)

	for folder in os.listdir(dataDir):
		if monthref.match(folder):
			monthDir = os.path.join(dataDir, folder)
			if os.path.isdir(monthDir):
				if folderName in os.listdir(monthDir):
					dataDir = os.path.join(monthDir, folderName)
					if verbose > 1:
						print(f"Date folder: {dataDir}")
					return dataDir

	## Try looking through years...
	year = str(dateTpl.year)

	if year in os.listdir(dataDir):
		yearDir = os.path.join(dataDir, year)
		if os.path.isdir(yearDir):
			dataDir = deepcopy(yearDir)

	## Then check again that it is not in yearDir
	if folderName in os.listdir(dataDir):
		dataDir = os.path.join(dataDir, folderName)
		if verbose > 1:
			print(f"Date folder: {dataDir}")
		return dataDir

	## Then lets look for month folders...
	for folder in os.listdir(dataDir):
		if monthref.match(folder):
			monthDir = os.path.join(dataDir, folder)
			if os.path.isdir(monthDir):
				if folderName in os.listdir(monthDir):
					dataDir = os.path.join(monthDir, folderName)
					if verbose > 1:
						print(f"Date folder: {dataDir}")
					return dataDir

	## Finally we can try this format...
	dataDir = dataDir[:-5]
	folderfmt = dateTpl.strftime("%Y[/.-_ ]*%m[/.-_ ]*%d"+
		"|%m[/.-_ ]*%d[/.-_ ]*%Y")
	folderref = re.compile(folderfmt, re.IGNORECASE)

	for folder in os.listdir(dataDir):
		if os.path.isdir(os.path.join(dataDir,folder)):
			if folderref.match(folder):
				dataDir = os.path.join(dataDir, folder)
				if verbose > 1:
					print(f"Date folder: {dataDir}")
				return dataDir

	dateStr = dateTpl.strftime("%A, %B %d, %Y")
	err_str = f"Could not locate data from {dateStr}..."
	raise ValueError(err_str)



if __name__ == "__main__":

	infoPath = "./Runfiles/HH_Test/"

	infoDict = rfu.getInfo(infoPath, verbose=0)

	dataInfo = infoDict['data']

	data = load_data(dataInfo, verbose=2)



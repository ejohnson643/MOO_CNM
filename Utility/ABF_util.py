"""
================================================================================
	READ .ABF FILES
================================================================================

	Author/Translator: Eric Johnson
	Date: Thursday, May 17, 2018
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================

	This file will contain all the functions necessary for loading the data and
	header of an arbitrary ABF (Axon Binary Format) file.

================================================================================
"""
from copy import deepcopy
import datetime
import numpy as np
import os
import re
from struct import unpack
import time
import Utility.utility as utl


#===============================================================================
#	Set Parameters
#===============================================================================

# Constants for nFileType
ABF_ABFFILE = 1
ABF_FETCHEX = 2
ABF_CLAMPEX = 3

# Constants used in Header
ABF_WAVEFORMCOUNT = 2
ABF_EPOCHCOUNT = 10

# Constants for nWaveformSource
ABF_WAVEFORMDISABLED   = 0
ABF_EPOCHTABLEWAVEFORM = 1
ABF_DACFILEWAVEFORM = 2

# Constants for ABFH_GetEpochLimits
ABFH_FIRSTHOLDING = -1
ABFH_LASTHOLDING = 10

# Different possible values for nOperationMode
ABF_VARLENEVENTS = 1
ABF_FIXLENEVENTS = 2
ABF_LOSSFREEOSC  = 2
ABF_GAPFREEFILE  = 3
ABF_HIGHSPEEDOSC = 4
ABF_WAVEFORMFILE = 5

# Constants for nEpochType
ABF_EPOCHDISABLED			= 0
ABF_EPOCHSTEPPED			= 1
ABF_EPOCHRAMPED				= 2
ABF_EPOCH_TYPE_RECTANGLE	= 3
ABF_EPOCH_TYPE_TRIANGLE		= 4
ABF_EPOCH_TYPE_COSINE		= 5
ABF_EPOCH_TYPE_RESISTANCE	= 6
ABF_EPOCH_TYPE_BIPHASIC		= 7

# Constants for nParamToVary
ABF_CONDITNUMPULSES			= 0
ABF_CONDITBASELINEDURATION	= 1
ABF_CONDITBASELINELEVEL		= 2
ABF_CONDITSTEPDURATION		= 3
ABF_CONDITSTEPLEVEL			= 4
ABF_CONDITPOSTTRAINDURATION = 5
ABF_CONDITPOSTTRAINLEVEL	= 6
ABF_EPISODESTARTTOSTART		= 7
ABF_INACTIVEHOLDING			= 8
ABF_DIGITALHOLDING			= 9
ABF_PNNUMPULSES				= 10
ABF_PARALLELVALUE			= 11
ABF_EPOCHINITLEVEL			= ABF_PARALLELVALUE + ABF_EPOCHCOUNT
ABF_EPOCHINITDURATION		= ABF_EPOCHINITLEVEL + ABF_EPOCHCOUNT
ABF_EPOCHTRAINPERIOD		= ABF_EPOCHINITDURATION + ABF_EPOCHCOUNT
ABF_EPOCHTRAINPULSEWIDTH	= ABF_EPOCHTRAINPERIOD + ABF_EPOCHCOUNT

A_VERY_SMALL_NUMBER = 1e-10

ABF_BLOCKSIZE = 512 # Data are stored in chunks of ABF_BLOCKSIZE bites
CHUNK = 0.05

DEFAULT_LEVEL_HYSTERESIS = 64
DEFAULT_TIME_HYSTERESIS = 1

g_uWaveformEpisodeNum = 1

machineF = "<"	# Specifies that binary data are little-endian.

#===============================================================================
#	Check Valid ABF File
#===============================================================================
def check_valid_abf_file(filename):
	"""
		check_valid_abf_file(filename)

		This function checks that a given filename is a nonempty string
		corresponding to a .abf file.
	"""

	if not isinstance(filename, str):
		err_str = "Input argument 'filename' is not a string."
		raise TypeError(err_str)

	if len(filename) < 1:
		err_str = "Input argument 'filename' is empty."
		raise ValueError(err_str)

	if filename[-3:] != 'abf':
		err_str = "Input argument 'filename' must be the name of a .abf file."
		raise ValueError(err_str)

	return


#===============================================================================
#	ABF Date Extract
#===============================================================================
def ABF_date_extract(filename):
	"""
		date = ABF_date_extract(filename, **kwargs)

		This function extracts the date that an abf file was created from its
		name.

		This function assumes that the filename has the format
			"YYYY_MM_DD_####.abf"
		by default.
	"""
	check_valid_abf_file(filename)

	dateref = re.compile("(\d{4})_(\d{2})_(\d{2})", re.IGNORECASE)
	mat = dateref.match(filename)

	if mat is not None:
		return datetime.datetime(*map(int, mat.groups()))
	else:
		err_str = "Could not extract date from filename. (%s)" % filename
		raise ValueError(err_str)


#===============================================================================
#	ABF Read
#===============================================================================
def ABF_read(filename,
			 datadir='',
			 verbose=0,
			 onlyInfo=False):
	"""
		[d, si, h] = readabf(filename, **kwargs)

		This function loads and returns data from ABF (Axon Binary File) files.

		This function automatically interprets the header of these files to
		determine the proper method for data retrieval.  Most importantly, the
		function will determine the ABF version, which will dictate the form of
		the header, as well as the protocol under which the data were collected.
		There are three main ways the data may be collected:

			1. 	Event-Driven Variable-Length (< ABF v.2.0 only)
			2.	Event-Drive Fixed-Length or Waveform Fixed-Length
			3.	Gap-Free

		Information about scaling, the time base, and the number of channels and
		episodes is extracted from the header of the ABF file.

	============================================================================
	============================================================================

		Arguments:
		==========
		filename	String containing the name of the ABF file to be loaded.

		Keyword Arguments:
		==================
		datadir		String containing the directory containing the data (default
					'' indicating that the datadirectory should be inferred from
					the filename).
		verbose		Flag indicating the verbosity with which to indicate that 
					data are being loaded.  (default = 0)
		onlyInfo 	Boolean indicating whether the function should return only
					the header and NOT load any data.  (default = False)

		Output:
		=======
		data 		The format depends on the way the data were recorded:
						1. Gap-Free:
							2D array (nDataPts x nChannels)

						2. Episodic Fixed-Length/Waveform Fixed-Length/High-Spd
						   Oscilloscope:
						 	3D array (nDataPtsPerSweep x nChannels x nSweeps)

						3. Episodic Variable-Length:
							Array of 2D arrays, where each sub-array contains 1
							episode. Each episode is a 2D array
							(nDataPtsInSweep x nChannels)

		hdr 		Header parameters (dictionary).  For a detailed summary of
					each of the header parameters, see the Abfhelp.chm.  

					Some important parameters are:
						dataPts -- Number of sample points in file.
						dataPtsPerChan -- Number of sample points in each
							channel
						fADCProgrammableGain -- array of gains
						fADCRange -- ...
						fADCSampleInterval -- ...
						fFileSignature -- First of two file differentiators. 
							This determines whether the file version is </>v.2.0
						fFileVersionNumber -- File version.  Header is
							completely different for > v.2.0.
						fInstrumentOffset -- Voltage offset from instrument (?)
						fInstrumentScaleFactor -- ...
						fSignalGain -- ...
						fSignalOffset -- Voltage offset from signal (?)
						fSynchTimeUnit -- If 0, then arrays have time in terms
							of "ticks", else synch array section is in units of
							microseconds.
						fTelegraphAdditGain -- ...
						lADCResolution -- ...
						lActualAcqLength -- Number of data points collected (?)
						lActualEpisodes -- Number of episodes.
						lDataSectionPtr -- Pointer to when data starts
						lEpisodesPerRun -- Number of episodes
						lFileStartTime -- Time at which recording began
						lNumSamplesPerEpisode -- Samples in each episode
						lNumTagEntries -- Number of tags
						lPreTriggerSamples -- ...
						lSynchArrayPtr -- Pointer to beginning of synch array
						lSynchArraySize -- Size of synch array
						lTagSectionPtr -- ...
						nADCNumChannels -- Number of channels
						nADCPtoLChannelMap -- ...
						nADCSamplingSeq -- ...
						nDataFormat -- Whether data is stored as integers or
							floats.
						nFileStartMillisecs -- Millisecond level precision of
							start time.
						nNumPointsIgnored -- Number of points to ignore after
							header and before data.
						nOperationMode -- Recording protocol.
							1 = Episodic variable-length,
							2 = Event-driven fixed-length
							3 = Gap-free
							4 = Oscilloscope
							5 = Waveform fixed-length
						nTelegraphEnable -- ...
						recChNames -- Recorded Channel Names
						recChUnits -- Recorded Channel Units
						recTime -- Recording time (?)
						sADCChannelName -- ...
						sADCUnits -- ...
						si -- Sampling Interval
						synchArrTimeBase -- Time units for synch array
						tags -- ...

	============================================================================
	============================================================================

		Other Info:
		===========
		This function works by first intepreting the header of the file by
		correctly locating and converting bytes into different data types.  This
		reading in starts at the very beginning of the file, where the
		'fFileSignature' tells how the header is put together and the 
		'fFileVersion' gives even more details on some operations.  After this,
		the rest of the header parameters are loaded.

		Finally, based on what was deduced from the header, the data is loaded 
		according to the corresponding protocol.

		In the description of the headers above, those with '...' are listed as 
		such because I have not had time to fill in what they do.


		Contributors:
		=============
		This code is partially based on MATLAB code translated by Eric Johnson
		(ericjohnson1.2015@u.northwestern.edu).  The original version is by
		Harald Hentschke (harald.hentschke@uni-tuebingen.de).  The current
		version was extended to ABF v.2.0 by Forrest Collman
		(fcollman@Princeton.edu) and Ulrich Egert (egert@bccn.uni-freiburg.de)

		The code is also based on the Axon's pCLAMP data aquisition and analysis
		programs.  For more info, see Abfhelp.chm, the documenation provided by
		Axon.

	============================================================================
	============================================================================
	"""

#===============================================================================
#	Check Inputs, Find File
#===============================================================================

	# Check that verbose is a nonnegative integer.
	verbose = utl.force_pos_float(verbose, name='verbose', zero_ok=True)

	# Check that the given filename is valid.
	check_valid_abf_file(filename)

	if verbose >= 1:
		print_str = "\nReading in file: %s" % filename
		print(print_str)

	# Check that datadir is a valid string
	if not isinstance(datadir, str):
		err_str = "Keyword argument 'datadir' is not a string."
		raise TypeError(err_str)

	# If no data directory is given, infer one from the file name.
	if len(datadir) == 0:
		if verbose >= 2:
			print_str = "\nAttempting to infer file date from filename."
			print(print_str)
		date = ABF_date_extract(filename)
		if verbose >= 2:
			full_date_str = time.strftime("%A, %B %d %Y", date.timetuple())
			print_str = "Data taken on %s" % full_date_str
			print(print_str)

		datedir = utl.find_date_folder(date)
	else:
		datedir = datadir

	fullpath = os.path.join(datedir, filename)

	# Check that a file exists at the derived full file path location.
	if not os.path.isfile(fullpath):
		err_str = "Could not find requested file %s" % fullpath
		raise ValueError(err_str)
	else:
		if verbose >= 2:
			print_str = "\nConfirmed. %s exists in %s." % (filename, datedir)
			print(print_str)

#===============================================================================
#	Open File, Get ABF Version
#===============================================================================

	# Open file and get binary data.
	if verbose >= 1:
		print_str = "\nOpening %s...\n" % filename
		print(print_str)
	with open(fullpath, 'rb') as f:
		rawdata = f.read()

	filesize = len(rawdata)

	# The ABF version is stored in the first 4 bits as a string
	ABF_version = unpack(machineF + "cccc", rawdata[:4])
	ABF_version = (b"".join(ABF_version)).decode()

	# If 'ABF_version' is 'ABF ' (with space), then ABF version < 2.0,
	# otherwise, ABF version is >= 2.0.
	if ABF_version not in ['ABF ', 'ABF2']:
		raise ValueError("Unknown or Incompatible File Signature!")

#===============================================================================
#	Get Header Fields, Load Header
#===============================================================================
	headPar = define_header(ABF_version)

	# Make sure the data are sorted by file offset
	headPar = sorted(headPar, key=lambda row: row[1])

	# Create the header dictionary and fill it using the byte information in
	# headPar
	hdr = {}
	for ii in range(len(headPar)):
		fieldname = headPar[ii][0]
		nData = headPar[ii][4] # Number of data items to load for param ii.
		structstr = machineF + nData*headPar[ii][2] # Data structure format

		# Load the data.
		hdr[fieldname] = unpack(structstr,
			rawdata[headPar[ii][1]:headPar[ii][1] + nData*headPar[ii][3]])

		# If parameter is a string, concatenate
		if headPar[ii][2] == 'c':
			hdr[fieldname] = (b''.join(hdr[fieldname])).decode()

		# If parameter is a single number, remove from tuple
		elif headPar[ii][4] == 1:
			hdr[fieldname] = hdr[fieldname][0]

		# If parameter is multiple numbers, make into numpy array.
		else:
			hdr[fieldname] = np.array(hdr[fieldname])
			# hdr[fieldname] = hdr[fieldname].reshape(1, len(hdr[fieldname]))

	if ABF_version != 'ABF ':
		return None, hdr

	hdr = promoteHeader(hdr)
	hdr['sFilename'] = filename
	hdr['sDirectory'] = datedir
	hdr['sFullPath'] = fullpath

#===============================================================================
#	Parameter Checks for Header Fields
#===============================================================================
	
	# Check that there are enough data
	if hdr['lActualAcqLength'] < hdr['nADCNumChannels']:
		err_str = "There are fewer data points than channels..."
		raise ValueError(err_str)

	# Get the indices of the recorded channels
	recChIdx = hdr['nADCSamplingSeq'][:hdr['nADCNumChannels']].astype(int)
	recChInd = range(len(recChIdx))

	# Extract the names of the used channels
	hdr['sADCChannelName'] = [hdr['sADCChannelName'][10*i:10*(i+1)].strip(' ')
		for i in range(16)]
	hdr['recChNames'] = [hdr['sADCChannelName'][ind] for ind in recChIdx]

	# Extract the units of the channels that are actually recorded
	hdr['sADCUnits'] = [hdr['sADCUnits'][8*i:8*(i+1)].strip(' ')
		for i in range(16)]
	hdr['recChUnits'] = [hdr['sADCUnits'][ind] for ind in recChIdx]

	# Restructure certain header fields
	hdr['sDACChannelName'] = [hdr['sDACChannelName'][10*i:10*(i+1)].strip(' ')
		for i in range(4)]
	hdr['sDACChannelUnits'] = [hdr['sDACChannelUnits'][8*i:8*(i+1)].strip(' ')
		for i in range(4)]
	hdr['lEpochPulsePeriod'] = hdr['lEpochPulsePeriod'].reshape(2, 10)
	hdr['lEpochPulseWidth'] = hdr['lEpochPulseWidth'].reshape(2, 10)
	hdr['nEpochType'] = hdr['nEpochType'].reshape(2, 10)
	hdr['fEpochInitLevel'] = hdr['fEpochInitLevel'].reshape(2, 10)
	hdr['fEpochLevelInc'] = hdr['fEpochLevelInc'].reshape(2, 10)
	hdr['lEpochInitDuration'] = hdr['lEpochInitDuration'].reshape(2, 10)
	hdr['lEpochDurationInc'] = hdr['lEpochDurationInc'].reshape(2, 10)

	if verbose >= 2:
		print_str = "Available Channels:\n" + 40*"=" + "\n"
		print_str += "".join(["%s (%s)\n"  % (s, u) for (s, u)
			in zip(hdr['recChNames'], hdr['recChUnits'])])
		print(print_str)

	# Determine the additional gain, if any.
	addGain = [x*y for (x, y)
		in zip(hdr['nTelegraphEnable'], hdr['fTelegraphAdditGain'])]

	# Determine the TOTAL SAMPLING INTERVAL
	hdr['sampInt'] = hdr['fADCSampleInterval']*hdr['nADCNumChannels']

	### Synch Array Stuff
	if True:
		if (hdr['lSynchArraySize'] <= 0) or (hdr['lSynchArrayPtr'] <= 0):
			hdr['lSynchArraySize'] = 0
			hdr['lSynchArrayPtr'] = 0

		# Set synch array time units
		hdr['synchArrTimeBase'] = hdr['fSynchTimeUnit']
		if hdr['fSynchTimeUnit'] == 0:
			hdr['synchArrTimeBase'] = 1

	# ### Tag Section
	# if hdr['lNumTagEntries'] > 0:
	# 	hdr['Tags'] = []
	# 	for ii in range(hdr['lNumTagEntries']):

	### OTHER STUFF
	if True:
		if hdr['fADCRange'] <= A_VERY_SMALL_NUMBER:
			hdr['fADCRange'] = 10.

		if ((hdr['nTrialTriggerSource']!=-2) and
			(hdr['nTrialTriggerSource']!=-3)):
			hdr['nTriggerSource'] = -1

		if hdr['fAverageWeighting'] < 0.001:
			hdr['fAverageWeighting'] = 0.1

		for ii in range(ABF_WAVEFORMCOUNT):
			if hdr['nPNPolarity'][ii] == 0:
				hdr['nPNPolarity'][ii] = 1

			if hdr['lDACFileEpisodeNum'][ii] == -1:
				hdr['lDACFileEpisodeNum'][ii] = 0

			if ((hdr['nWaveformEnable'][ii] == False) and
				(hdr['nWaveformSource'][ii] == 0)):
				hdr['nWaveformSource'][ii] = 1

		if hdr['nStatsSmoothing'] < 1:
			hdr['nStatsSmoothing'] = 1

		if (hdr['nLevelHysteresis'] == 0) and (hdr['lTimeHysteresis'] == 0):
			hdr['nLevelHysteresis'] = DEFAULT_LEVEL_HYSTERESIS
			hdr['lTimeHysteresis'] = DEFAULT_TIME_HYSTERESIS

	# Round file version to nearest thousandth
	hdr['fFileVersionNumber'] = 0.001*round(hdr['fFileVersionNumber']*1000)
	# Make file start time precise to milliseconds.
	hdr['lFileStartTime'] = (hdr['lFileStartTime'] +
		hdr['nFileStartMillisecs']*0.001)

	# Fix the date.
	lDate = hdr['lFileStartDate']
	lStartDay = int(lDate % 100)
	lStartMonth = int((lDate % 10000)/100)
	lStartYear = int(lDate/10000)
	if lStartYear < 1000:
		if lStartYear < 80:
			lStartYear += 2000
		else:
			lStartYear += 1900

	hdr['lFileStartDate'] = int(lStartYear*1e4 + lStartMonth*100 + lStartDay)

	# If we only want the header, then return here.
	if onlyInfo:
		return hdr

#===============================================================================
#	Checks for Raw Data, Loads Data
#===============================================================================

	# Check that there actually is data to load.
	if (hdr['lActualAcqLength'] <= 0) or (hdr['nADCNumChannels'] <= 0):
		err_str = "No data to read!"
		raise ValueError(err_str)

	# Disable stimulus file output if data file does not include any stimulus
	# file sweeps.  This is to prevent problems later when looking for non-
	# existent DACFile section
	for uDAC in range(ABF_WAVEFORMCOUNT):
		if ((hdr['lDACFileNumEpisodes'][uDAC] <= 0) or
			(hdr['lDACFilePtr'][uDAC]<=0)):
			hdr['lDACFileNumEpisodes'][uDAC] = 0
			hdr['lDACFilePtr'][uDAC] = 0
			if hdr['nWaveformSource'][uDAC] == ABF_DACFILEWAVEFORM:
				hdr['nWaveformSource'][uDAC] = 0

	if hdr['nOperationMode'] == 1:
		if verbose >= 1:
			print_str = "Data were acquired in event-driven variable-length"
			print(print_str + " mode.")
		data = readEDVarLenMode(rawdata, hdr)

	elif hdr['nOperationMode'] == 3:
		if verbose >= 1:
			print_str = "Data were acquired in gap-free mode."
			print(print_str)
		data = readGapFree(rawdata, hdr)

	else:
		if verbose >= 1:
			if hdr['nOperationMode'] == 2:
				print("Data were acquired in event-drive fixed-length mode")
			elif hdr['nOperationMode'] == 4:
				print("Data were acquired in high-speed oscilloscope mode")
			else:
				print("Data were acquired in waveform fixed-length mode")

		data = readEDFixLenMode(rawdata, hdr)



	return data, hdr


#===============================================================================
#	Define Header
#===============================================================================
def define_header(fileVersion):
	"""
		headPar = define_header(ABF_version)

		This function generates a list of header fields in the format:
			[name, bite offset, data type, data size (bytes), number of data]
	"""
	if fileVersion == 'ABF ':
		headPar = [
			['fFileSignature',			0,		'c',	1,	4],
			['fFileVersionNumber',		4,		'f',	4,	1],
			['nOperationMode',			8,		'h',	2,	1],
			['lActualAcqLength',		10,		'l',	4,	1],
			['nNumPointsIgnored',		14,		'h',	2,	1],
			['lActualEpisodes',			16,		'l',	4,	1],
			['lFileStartDate',			20,		'l',	4,	1],
			['lFileStartTime',			24,		'l',	4,	1],
			['lStopwatchTime',			28,		'l',	4,	1],
			['fHeaderVersionNumber',	32,		'f',	4,	1],
			['nFileType',				36,		'h',	2,	1],
			['nMSBinFormat',			38,		'h',	2,	1],
			['lDataSectionPtr',			40,		'l',	4,	1],
			['lTagSectionPtr',			44,		'l',	4,	1],
			['lNumTagEntries',			48,		'l',	4,	1],
			['lScopeConfigPtr',			52,		'l',	4,	1],
			['lNumScopes',				56,		'l',	4,	1],
			['lDeltaArrayPtr',			72,		'l',	4,	1],
			['lNumDeltas',				76,		'l',	4,	1],
			['lVoiceTagPtr',			80,		'l',	4,	1],
			['lVoiceTagEntries',		84,		'l',	4,	1],
			['lSynchArrayPtr',			92,		'l',	4,	1],
			['lSynchArraySize',			96,		'l',	4,	1],
			['nDataFormat',				100,	'h',	2,	1],
			['nSimultaneousScan',		102,	'h',	2,	1],
			['lStatisticsConfigPtr',	104,	'h',	2,	1],
			['lAnnotationSectionPtr',	108,	'l',	4,	1],
			['lNumAnnotations',			112,	'l',	4,	1],
			['channel_count_acquired',	118,	'h',	2,	1],
			['nADCNumChannels',			120,	'h',	2,	1],
			['fADCSampleInterval',		122,	'f',	4,	1],
			['fADCSecondSampleInterval',126,	'f',	4,	1],
			['fSynchTimeUnit',			130,	'f',	4,	1],
			['fSecondsPerRun',			134,	'f',	4,	1],
			['lNumSamplesPerEpisode',	138,	'l',	4,	1],
			['lPreTriggerSamples',		142,	'l',	4,	1],
			['lEpisodesPerRun',			146,	'l',	4,	1],
			['lRunsPerTrial',			150,	'l',	4,	1],
			['lNumberOfTrials',			154,	'l',	4,	1],
			['nAveragingMode',			158,	'h',	2,	1],
			['nUndoRunCount',			160,	'h',	2,	1],
			['nFirstEpisodeInRun',		162,	'h',	2,	1],
			['fTriggerThreshold',		164,	'f',	4,	1],
			['nTriggerSource',			168,	'h',	2,	1],
			['nTriggerAction',			170,	'h',	2,	1],
			['nTriggerPolarity',		172,	'h',	2,	1],
			['fScopeOutputInterval',	174,	'f',	4,	1],
			['fEpisodeStartToStart',	178,	'f',	4,	1],
			['fRunStartToStart',		182,	'f',	4,	1],
			['fTrialStartToStart',		186,	'f',	4,	1],
			['lAverageCount',			190,	'l',	4,	1],
			['lClockChange',			194,	'l',	4,	1],
			['nAutoTriggerStrategy',	198,	'h',	2,	1],
			['nDrawingStrategy',		200,	'h',	2,	1],
			['nTiledDisplay',			202,	'h',	2,	1],
			['nEraseStrategy',			204,	'h',	2,	1],
			['nDataDisplayMode',		206,	'h',	2,	1],
			['lDisplayAverageUpdate',	208,	'l',	4,	1],
			['nChannelStatsStrategy',	212,	'h',	2,	1],
			['lCalculationPeriod',		214,	'l',	4,	1],
			['lSamplesPerTrace',		218,	'l',	4,	1],
			['lStartDisplayNum',		222,	'l',	4,	1],
			['lFinishDisplayNum',		226,	'l',	4,	1],
			['nMultiColor',				230,	'h',	2,	1],
			['nShowPNRawData',			232,	'h',	2,	1],
			['fStatisticsPeriod',		234,	'f',	4,	1],
			['lStatisticsMeasurements',	238,	'l',	4,	1],
			['nStatisticsSaveStrategy',	242,	'h',	2,	1],
			['fADCRange',				244,	'f',	4,	1],
			['fDACRange',				248,	'f',	4,	1],
			['lADCResolution', 			252,	'l',	4,	1],
			['lDACResolution',			256,	'l',	4,	1],
			['nExperimentType',			260,	'h',	2,	1],
			['nManualInfoStrategy',		280,	'h',	2,	1],
			['fCellIDs',				282,	'f',	4,	3],
			['sCreatorInfo',			294,	'c',	1,	16],
			['nFileStartMillisecs',		366,	'h',	2,	1],
			['nCommentsEnable',			368,	'h',	2,	1],
			['nADCPtoLChannelMap',		378,	'h',	2,	16],
			['nADCSamplingSeq',			410,	'h',	2,	16],
			['sADCChannelName',			442,	'c',	1,	16*10],
			['sADCUnits',				602,	'c',	1,	16*8],
			['fADCProgrammableGain',	730,	'f',	4,	16],
			['fADCDisplayAmplification',794,	'f',	4,	16],
			['fADCDisplayOffset',		858,	'f',	4,	16],
			['fInstrumentScaleFactor',	922,	'f',	4,	16],
			['fInstrumentOffset',		986,	'f',	4,	16],
			['fSignalGain',				1050,	'f',	4,	16],
			['fSignalOffset',			1114,	'f',	4,	16],
			['fSignalLowpassFilter',	1178,	'f',	4,	16],
			['fSignalHighpassFilter',	1242,	'f',	4,	16],
			['sDACChannelName',			1306,	'c',	1,	4*10],
			['sDACChannelUnits',		1346,	'c',	1,	4*8],
			['fDACScaleFactor',			1378,	'f',	4,	4],
			['fDACHoldingLevel',		1394,	'f',	4,	4],
			['nSignalType',				1410,	'h',	2,	1],
			['nOUTEnable',				1422,	'h',	2,	1],
			['nSampleNumberOUT1',		1424,	'h',	2,	1],
			['nSampleNumberOUT2',		1426,	'h',	2,	1],
			['nFirstEpisodeOUT',		1428,	'h',	2,	1],
			['nLastEpisodeOut',			1430,	'h',	2,	1],
			['nPulseSamplesOUT1',		1432,	'h',	2,	1],
			['nPulseSamplesOUT2',		1434,	'h',	2,	1],
			['nDigitalEnable',			1436,	'h',	2,	1],
			['nActiveDACChannel',		1440,	'h',	2,	1],
			['nDigitalHolding',			1584,	'h',	2,	1],
			['nDigitalInterEpisode',	1586,	'h',	2,	1],
			['nDigitalValue',			1588,	'h',	2,	16],
			['nDigitalDACChannel',		1612,	'h',	2,	1],
			['nArithmeticEnable',		1880,	'h',	2,	1],
			['fArithmeticUpperLimit',	1882,	'f',	4,	1],
			['fArithmeticLowerLimit',	1886,	'f',	4,	1],
			['nArithmeticADCNumA',		1890,	'h',	2,	1],
			['nArithmeticADCNumB',		1892,	'h',	2,	1],
			['fArithmeticK1',			1894,	'f',	4,	1],
			['fArithmeticK2',			1898,	'f',	4,	1],
			['fArithmeticK3',			1902,	'f',	4,	1],
			['fArithmeticK4',			1906,	'f',	4,	1],
			['sArithmeticOperator',		1910,	'c',	1,	2],
			['sArithmeticUnits',		1912,	'c',	1,	8],
			['fArithmeticK5',			1920,	'f',	4,	1],
			['fArithmeticK6',			1924,	'f',	4,	1],
			['nArithmeticExpression',	1928,	'h',	2,	1],
			['nPNPosition',				1934,	'h',	2,	1],
			['nPNNumPulses',			1938,	'h',	2,	1],
			['fPNSettlingTime',			1946,	'f',	4,	1],
			['fPNInterpulse',			1950,	'f',	4,	1],
			['nBellEnable',				1968,	'h',	2,	1],
			['nBellLocation',			1972,	'h',	2,	1],
			['nBellRepetition',			1976,	'h',	2,	1],
			['nLevelHysteresis',		1980,	'h',	2,	1],
			['lTimeHysteresis',			1982,	'l',	4,	1],
			['nAllowExternalTags',		1986,	'h',	2,	1],
			['nLowpassFilterType',		1988,	'c',	1,	16],
			['nHighpassFilterType',		2004,	'c',	1,	16],
			['nAverageAlgorithm',		2020,	'h',	2,	1],
			['fAverageWeighting',		2022,	'f',	4,	1],
			['nUndoPromptStrategy',		2026,	'h',	2,	1],
			['nTrialTriggerSource',		2028,	'h',	2,	1],
			['nStatsDisplayStrategy',	2030,	'h',	2,	1],
			['nExternalTagType',		2032,	'h',	2,	1],
			['lHeaderSize',				2034,	'l',	4,	1],
			['lDACFilePtr',				2048,	'l',	4,	2],
			['lDACFileNumEpisodes',		2056,	'l',	4,	2],
			['fDACCalibrationFactor',	2074,	'f',	4,	4],
			['fDACCalibrationOffset',	2090,	'f',	4,	4],
			['lEpochPulsePeriod',		2136,	'l',	4,	2*10],
			['lEpochPulseWidth',		2216,	'l',	4,	2*10],
			['nWaveformEnable',			2296,	'h',	2,	2],
			['nWaveformSource',			2230,	'h',	2,	2],
			['nInterEpisodeLevel',		2304,	'h',	2,	2],
			['nEpochType',				2308,	'h',	2,	2*10],
			['fEpochInitLevel',			2348,	'f',	4,	2*10],
			['fEpochLevelInc',			2428,	'f',	4,	2*10],
			['lEpochInitDuration',		2508,	'l',	4,	2*10],
			['lEpochDurationInc',		2588,	'l',	4,	2*10],
			['nDigitalTrainValue',		2668,	'h',	2,	10],
			['nDigitalTrainActiveLogic',2688,	'h',	2,	1],
			['fDACFileScale',			2708,	'f',	4,	2],
			['fDACFileOffset',			2716,	'f',	4,	2],
			['lDACFileEpisodeNum',		2724,	'f',	4,	2],
			['nDACFileADCNum',			2732,	'h',	2,	2],
			['sDACFilePath',			2736,	'c',	1,	2*256],
			['nConditEnable',			3260,	'h',	2,	2],
			['lConditNumPulses',		3264,	'l',	4,	2],
			['fBaselightDuration',		3272,	'f',	4,	2],
			['fBaselineLevel',			3280,	'f',	4,	2],
			['fStepDuration',			3288,	'f',	4,	2],
			['fStepLevel',				3296,	'f',	4,	2],
			['fPostTrainPeriod',		3304,	'f',	4,	2],
			['fPostTrainLevel',			3312,	'f',	4,	2],
			['nULEnable',				3360,	'h',	2,	4],
			['nULParamToVary',			3368,	'h',	2,	4],
			['sULParamValueList',		3376,	'c',	1,	4*256],
			['nULRepeat',				4400,	'h',	2,	4],
			['nPNEnable',				4456,	'h',	2,	2],
			['nPNPolarity',				4460,	'h',	2,	2],
			['nPNADCNum',				4464,	'h',	2,	2],
			['fPNHoldingLevel',			4468,	'f',	4,	2],
			['nTelegraphEnable',		4512,	'h',	2,	16],
			['nTelegraphInstrument',	4544,	'h',	2,	16],
			['fTelegraphAdditGain',		4576,	'f',	4,	16],
			['fTelegraphFilter',		4640,	'f',	4,	16],
			['fTelegraphMembraneCap',	4704,	'f',	4,	16],
			['nTelegraphMode',			4768,	'h',	2,	16],
			['nTelDACScaleFactorEnable',4800,	'h',	2,	4],
			['nAutoAnalyseEnable',		4832,	'h',	2,	1],
			['sAutoAnalysisMacroName',	4834,	'c',	1,	64],
			['sProtocolPath',			4898,	'c',	1,	256],
			['sFileComment',			5154,	'c',	1,	128],
			['nStatsEnable',			5410,	'h',	2,	1],
			['nStatsActiveChannels',	5412,	'H',	2,	1],
			['nStatsSearchRegionFlags',	5414,	'H',	2,	1],
			['nStatsSelectedRegion',	5416,	'h',	2,	1],
			['nStatsSmoothing',			5420,	'h',	2,	1],
			['nStatsSmoothingEnable',	5422,	'h',	2,	1],
			['nStatsBaseline',			5424,	'h',	2,	1],
			['lStatsBaselineStart',		5426,	'l',	4,	1],
			['lStatsBaselineEnd',		5430,	'l',	4,	1],
			['lStatsMeasurements',		5434,	'l',	4,	8],
			['lStatsStart',				5466,	'l',	4,	8],
			['lStatsEnd',				5498,	'l',	4,	8],
			['nRiseBottomPercentile',	5466,	'h',	2,	8],
			['nRiseTopPercentile',		5546,	'h',	2,	8],
			['nDecayBottomPercentile',	5562,	'h',	2,	8],
			['nDecayTopPercentile',		5578,	'h',	2,	8],
			['nStatsChannelPolarity',	5594,	'h',	2,	16],
			['nStatsSearchMode',		5626,	'h',	2,	8],
			['nMajorVersion',			5798,	'h',	2,	1],
			['nMinorVersion',			5800,	'h',	2,	1],
			['nBugfixVersion',			5802,	'h',	2,	1],
			['nBuildVersion',			5804,	'h',	2,	1],
			['nLTPType',				5814,	'c',	1,	1],
			['nLTPUsageOfDAC',			5816,	'h',	2,	2],
			['nLTOPPresynapticPulses',	5820,	'h',	2,	2],
			['nDD132xTriggerOut',		5828,	'h',	2,	1],
			['sEpochResistanceSignalName', 5836, 'c',	1,	2*10],
			['nEpochResistanceState',	5856,	'h',	2,	2],
			['nAlternateDACOutputState',5876,	'h',	2,	1],
			['nAlternateDigitalValue',	5878,	'h',	2,	10],
			['nAlternateDigitalTrainValue', 5898, 'h',	2,	10],
			['nAlternateDigitalOutputState', 5918, 'h',	2,	1],
			['fPostProcessLowpassFilter', 5934,	'f',	4,	16],
			['nPostProcessLowpassFilterType',	5998,	'c',	1,	1]
		]

	elif fileVersion == 'ABF2':
		headPar = [
			['fFileSignature',			0,    'c',	1,	4],
			['fFileVersionNumber',		4,    'B',	1,	4],
			['uFileInfoSize',			8,    'L',	4,	1],
			['lActualEpisodes',			12,   'L',	4,	1],
			['uFileStartDate',			16,   'L',	4,	1],
			['uFileStartTimeMS',		20,   'L',	4,	1],
			['uStopwatchTime',			24,   'L',	4,	1],
			['nFileType',				28,   'h',	2,	1],
			['nDataFormat',				30,   'h',	2,	1],
			['nSimultaneousScan',		32,   'h',	2,	1],
			['nCRCEnable',				34,   'h',	2,	1],
			['uFileCRC',				36,   'L',	4,	1],
			['FileGUID',				40,   'L',	4,	1],
			['uCreatorVersion',			56,   'L',	4,	1],
			['uCreatorNameIndex',		60,   'L',	4,	1],
			['uModifierVersion',		64,   'L',	4,	1],
			['uModifierNameIndex',		68,   'L',	4,	1],
			['uProtocolPathIndex',		72,   'L',	4,	1]
		]

	else:
		raise ValueError("Unrecognized File Signature!")

	return headPar


#===============================================================================
#	"Promote" Header
#===============================================================================
def promoteHeader(oldhdr):

	hdr = deepcopy(oldhdr)
	ii = hdr['nActiveDACChannel']
	hdr['nWaveformSource'][ii] =2 if oldhdr['nWaveformSource'][ii]==2 else 1
	hdr['nWaveformEnable'][ii]=1 if not oldhdr['nWaveformSource'][ii] else 0

	return hdr


#===============================================================================
#	Read in Variable-Length Synch Arrays
#===============================================================================
def readEDVarLenMode(rawdata, hdr):

	headOffset = (hdr['lDataSectionPtr']*BLOCKSIZE +
		hdr['nNumPointsIgnored']*dataSz)

	recChIdx = hdr['nADCSamplingSeq'][:hdr['nADCNumChannels']].astype(int)
	recChInd = range(len(recChIdx))

	if (hdr['lSynchArraySize'] <= 0) and (hdr['lSynchArrayPtr'] <= 0):
		err_str = "There is no Synch Array to load!"
		raise FileError(err_str)

	SampSize = 2 if hdr['nDataFormat'] == 0 else 4
	SampPrec = 'f' if SampSize == 4 else "h"
	uAcqLength = int(hdr['lActualAcqLength'])

	# The byte offset at which the SynchArray Section starts
	h['lSynchArrayPtrByte'] = h['lSynchArrayPtr']*ABF_BLOCKSIZE

	# Check that the file is the appropriate length.  There are two numbers
	# (start and length), each SampSize bytes, for each episode.
	synchArrSize = hdr['lSynchArrayPtrByte'] + 2*SampSize*hdr['lSynchArraySize']
	if (synchArrSize < len(rawdata)):
		err_str = "File does not seem to contain complete 'Synch Array section."
		raise ValueError(err_str)

	# Read in the synch array, reshape
	readfmt = machineF + h['lSynchArraySize']*2*'i'
	synchArr = unpack(readfmt, rawdata[hdr['lSynchArrayPtrByte']:synchArrSize])
	synchArr = np.array(synchArr).reshape(hdr['synchArrSize'], 2)

	# Get the length of the episodes in terms of sample points
	segLengthInPts = (synchArr[:,1]/hdr['synchArrTimeBase']).astype(int)
	# Get the starting ticks of episodes in sample points WITHIN THE DATA
	segStartInPts = (cumsum([d*SampSize for d in ([0] + segLengthInPts[:-1]) +
		headOffset]))

	if hdr['fFileVersionNumber'] >= 1.65:
		addGain = [x*y for (x, y) in
			zip(hdr['nTelegraphEnable'], hdr['fTelegraphAdditGain'])]
		addGain = [g if g != 0. else 1. for g in addGain]
	else:
		addGain = len(hdr['fTelegraphAdditGain'])*[1.]

	data = []
	for ii in range(hdr['lActualEpisodes']):
		readfmt = machineF + segLengthInPts*SampPrec
		segEnd = segStartInPts[ii] + segLengthInPts[ii]*SampSize
		tempd = unpack(readfmt, rawdata[segStartInPts[ii]:segEnd])

		if len(tmpd)%hdr['nADCNumChannels'] != 0:
			err_str = "Number of data points does not divide into the number of"
			err_str += " recorded channels."
			raise ValueError(err_str)
		hdr['dataPtsPerChan'] = len(tmpd)/hdr['nADCNumChannels']

		tmpd = np.array(tmpd).reshape(hdr['dataPtsPerChan'],
			hdr['nADCNumChannels'])[:, recChInd]

		# If data format is integer, scale the data
		if not hdr['nDataFormat']:
			for jj in range(len(recChInd)):
				ch = recChIdx[jj]
				scaleFactor = hdr['fInstrumentScaleFactor'][ch]
				scaleFactor *= hdr['fSignalGain'][ch]
				scaleFactor *= hdr['fADCProgrammableGain'][ch]
				scaleFactor *= addGain[ch]
				tmpd[:, jj] /= scaleFactor
				tmpd[:, jj] *= hdr['fADCRange']/hdr['lADCResolution']
				tmpd[:, jj] += hdr['fInstrumentOffset'][ch]
				tmpd[:, jj] -= hdr['fSignalOffset'][ch]

		data.append(tmpd)

	return data


#===============================================================================
#	Read in Gap-Free Data
#===============================================================================
def readGapFree(rawdata, hdr):

	startPt = 0

	recChIdx = hdr['nADCSamplingSeq'][:hdr['nADCNumChannels']].astype(int)
	recChInd = range(len(recChIdx))

	SampSize = 2 if hdr['nDataFormat'] == 0 else 4
	SampPrec = 'f' if SampSize == 4 else "h"

	headOffset = (hdr['lDataSectionPtr']*ABF_BLOCKSIZE +
		hdr['nNumPointsIgnored']*SampSize)

	hdr['dataPtsPerChan'] = hdr['lActualAcqLength']/hdr['nADCNumChannels']

	tmp = 1.e-6*hdr['lActualAcqLength']*hdr['fADCSampleInterval']

	if verbose >= 2:
		print_str = ("Total Length of Recording: %5.1f s ~ %3.0f min" %
			(tmp, tmp/60.))
		print_str += ("\nSampling Interval: %5.0 us" % hdr['sampInt'])
		print_str += ("\nMemory Requirement for Complete Upload in MATLAB: ")
		print_str += ("%d Mb" % np.round(8*hdr['lActualAcqLength']/(2.**20)))

	hdr['recTime'] = [hdr['lFileStartTime'], hdr['lFileStartTime']+tmp]

	if (len(recChInd) == 1) and (hdr['nADCNumChannels'] > 16):

		startIdx = startPt*SampSize + headOffset + recChInd[0]*SampSize
		endIdx = startIdx + SampSize*hdr['dataPtsPerChan']

		readfmt = machineF + hdr['dataPtsPerChan']*SampPrec
		data = unpack(readfmt, rawdata[startIdx:endIdx])

	elif (len(recChInd)/hdr['nADCNumChannels']) < 1:
		data = np.zeros((hdr['dataPtsPerChan'], len(recChInd)))

		chunkPtsPerChan = np.floor(CHUNK*(2**20)/8/hdr['nADCNumChannels'])
		chunkPts = chunkPtsPerChan*hdr['nADCNumChannels']

		nChunk = np.floor(hdr['lActualAcqLength']/chunkPts)

		restPts = hdr['lActualAcqLength'] - nChunk*chunkPts
		restPtsPerChan = restPts/hdr['nADCNumChannels']

		tmp = np.arange(0, hdr['dataPtsPerChan'], chunkPtsPerChan)
		dix = np.zeros((len(tmp), 2))
		dix[:, 0] = tmp.T
		dix[:, 1] = dix[:, 0] + chunkPtsPerChan - 1
		dix[-1,1] = hdr['dataPtsPerChan']

		if (verbose >= 2) and nChunk:
			print_str = ("Reading file in %d chunks of ~%3.2f Mb" %
				(int(nChunk), CHUNK))

		startIdx = startPt*SampSize + headOffset

		for ci in range(len(dix)-(restpts>0)):
			readfmt = machineF + chunkPts*precision
			startIdx += ci*chunkPts
			endIdx = startIdx + chunkPts

			tmpd = unpack(readfmt, rawdata[startIdx:endIdx])

			tmpd = np.array(tmpd).reshape(chunkPtsPerChan,
				hdr['nADCNumChannels']).astype(float).T

			data[dix[ci, 0]:dix[ci, 1], :] = tmpd[recChInd, :].T

		if restPts:
			readfmt = machineF + chunkPts*precision
			startIdx += ci*chunkPts

			tmpd = unpack(readfmt, rawdata[startIdx:])

			tmpd = np.array(tmpd).reshape(restPtsPerChan,
				hdr['nADCNumChannels']).astype(float).T

			data[dix[-1, 0]:dix[-1, 1], :] = tmpd[recChInd, :].T

	else:
		readfmt = machineF + int(h['lActualAcqLength'])*SampPrec
		startIdx = int(startPt*SampSize + headOffset)
		endIdx = startIdx + int(h['lActualAcqLength']*SampSize)

		d = unpack(readfmt, data[startIdx:endIdx])

		d = np.array(d).reshape(int(h['dataPtsPerChan']),
			int(h['nADCNumChannels'])).astype(float)

	if hdr['fFileVersionNumber'] >= 1.65:
		addGain = [x*y for (x, y) in
			zip(hdr['nTelegraphEnable'], hdr['fTelegraphAdditGain'])]
		addGain = [g if g != 0. else 1. for g in addGain]
	else:
		addGain = len(hdr['fTelegraphAdditGain'])*[1.]

	if not hdr['nDataFormat']:
		for jj, ch in enumerate(recChInx):
			scaleFactor = hdr['fInstrumentScaleFactor'][ch]
			scaleFactor *= hdr['fSignalGain'][ch]
			scaleFactor *= hdr['fADCProgrammableGain'][ch]
			scaleFactor *= addGain[ch]
			tmpd[:, jj] /= scaleFactor
			tmpd[:, jj] *= hdr['fADCRange']/hdr['lADCResolution']
			tmpd[:, jj] += hdr['fInstrumentOffset'][ch]
			tmpd[:, jj] -= hdr['fSignalOffset'][ch]

	return data


#===============================================================================
#	Read in Fixed-Length Synch Arrays
#===============================================================================
def readEDFixLenMode(rawdata, hdr):

	recChIdx = hdr['nADCSamplingSeq'][:hdr['nADCNumChannels']].astype(int)
	recChInd = range(len(recChIdx))

	if (hdr['lSynchArraySize'] <= 0) and (hdr['lSynchArrayPtr'] <= 0):
		err_str = "There is no Synch Array to load!"
		raise FileError(err_str)

	SampSize = 2 if hdr['nDataFormat'] == 0 else 4
	SampPrec = 'f' if SampSize == 4 else "h"

	headOffset = (hdr['lDataSectionPtr']*ABF_BLOCKSIZE +
		hdr['nNumPointsIgnored']*SampSize)

	# The byte offset at which the SynchArray Section starts
	hdr['lSynchArrayPtrByte'] = ABF_BLOCKSIZE*hdr['lSynchArrayPtr']

	# Check that the file is the appropriate length.  There are two numbers
	# (start and length), each SampSize bytes, for each episode.
	synchArrSize = hdr['lSynchArrayPtrByte'] + 2*SampSize*hdr['lSynchArraySize']
	if (synchArrSize > len(rawdata)):
		err_str = "File does not seem to contain complete 'Synch Array section."
		raise ValueError(err_str)

	# Read in the data.
	readfmt = machineF + 2*hdr['lSynchArraySize']*'i'
	dataEnd = hdr['lSynchArrayPtrByte'] + 4*2*hdr['lSynchArraySize']
	synchArr = unpack(readfmt, rawdata[hdr['lSynchArrayPtrByte']:dataEnd])
	synchArr = np.array(synchArr).reshape(hdr['lSynchArraySize'], 2)

	if len(np.unique(synchArr[:, 1])) > 1:
		err_str = "Sweeps of unequal length recorded in fixed-length mode."
		raise ValueError(err_str)

	# The length of sweeps in terms of the number of samples.  (Note: the
	# parameter 'lLength' of the ABF synch section is expressed in samples, or
	# 'ticks', while the parameter 'lStart' is in synchArrTimeBase units.)
	hdr['sweepLengthInPts'] = int(synchArr[0, 1]/hdr['nADCNumChannels'])
	# The starting ticks of episodes in sample points (t0 = 1 = beginning)
	timefactor = hdr['synchArrTimeBase']/hdr['fADCSampleInterval']
	timefactor /= hdr['nADCNumChannels']
	hdr['sweepStartsinPts'] = synchArr[:, 0]*timefactor

	# Get the recording start and stop times in seconds from midnight
	hdr['recTime'] = hdr['sweepStartsinPts'][-1] + hdr['sweepLengthInPts']
	hdr['recTime'] *= hdr['fADCSampleInterval']*hdr['nADCNumChannels']*1.e-6
	hdr['recTime'] = np.array([0] + [hdr['recTime']]) + hdr['lFileStartTime']

	# Determine the start and end points of the data to be read.
	startPt = 0
	hdr['dataPtsPerChan'] = hdr['lActualAcqLength']/hdr['nADCNumChannels']

	if (hdr['lActualAcqLength'] % hdr['nADCNumChannels']):
		err_str = "Number of data points does not divide into number of"
		err_str += " channels."
		raise ValueError(err_str)

	if (hdr['dataPtsPerChan'] % hdr['lActualEpisodes']):
		err_str = "Number of data points does not divide into the number of"
		err_str += "episodes."
		raise ValueError(err_str)

	nSweeps = hdr['lActualEpisodes']
	sweeps = range(nSweeps)
	dataPtsPerSweep = synchArr[0, 1]
	selSegStartInPts = [s*dataPtsPerSweep*SampSize + headOffset for s in sweeps]

	data = np.zeros((hdr['sweepLengthInPts'], len(recChInd), nSweeps))

	if hdr['fFileVersionNumber'] >= 1.65:
		addGain = [x*y for (x, y) in
			zip(hdr['nTelegraphEnable'], hdr['fTelegraphAdditGain'])]
		addGain = [g if g != 0. else 1. for g in addGain]
	else:
		addGain = len(hdr['fTelegraphAdditGain'])*[1.]

	for sw in sweeps:
		readfmt = machineF + SampPrec*dataPtsPerSweep
		startPt = selSegStartInPts[sw]
		endPt = startPt + dataPtsPerSweep*SampSize
		tmpd = unpack(readfmt, rawdata[startPt:endPt])

		hdr['dataPtsPerChan'] = int(len(tmpd)/hdr['nADCNumChannels'])

		tmpd = np.array(tmpd).reshape(hdr['dataPtsPerChan'],
			hdr['nADCNumChannels']).astype(float)[:, recChInd]

		if not hdr['nDataFormat']:
			for jj, ch in enumerate(recChInd):
				scaleFactor = hdr['fInstrumentScaleFactor'][ch]
				scaleFactor *= hdr['fSignalGain'][ch]
				scaleFactor *= hdr['fADCProgrammableGain'][ch]
				scaleFactor *= addGain[ch]
				tmpd[:, jj] /= scaleFactor
				tmpd[:, jj] *= hdr['fADCRange']/hdr['lADCResolution']
				tmpd[:, jj] += hdr['fInstrumentOffset'][ch]
				tmpd[:, jj] -= hdr['fSignalOffset'][ch]

		data[:, :, sw] = tmpd

	return data


#===============================================================================
#	Show Waveform
#===============================================================================
def ShowWaveform(hdr):
	print("\n" + 40*"=")
	print("Showing Waveform")
	print(40*"=" + "\n")

	if hdr['nOperationMode'] != ABF_WAVEFORMFILE:
		err_str = "Can only show waveforms from episodic stimulation files.\n"
		raise ValueError(err_str)

	ShowEpochs(hdr, g_uWaveformEpisodeNum)

	uPerChannel = hdr['lNumSamplesPerEpisode']/hdr['nADCNumChannels']

	print_str = "Episode %d, " % g_uWaveformEpisodeNum

	waveform = GetWaveformEx(hdr, hdr['nActiveDACChannel'],
		g_uWaveformEpisodeNum)[:, 0].ravel()

	timebase = GetTimebase(hdr, 0.)

	print_str += "\t\tTime\tAmplitude\n"
	for ii in range(uPerChannel):
		print_str += "%4d\t\t%10.5g\t%10.5g\n" % (ii, timebase[ii],waveform[ii])
		if ii > 10:
			break

	print(print_str)


#===============================================================================
#	Show Description of Epochs
#===============================================================================
def ShowEpochs(hdr, uEpisode):

	if (((not hdr['nWaveformEnable'][0]) or
		 (hdr['nWaveformSource'][0] != ABF_EPOCHTABLEWAVEFORM)) and
		((not hdr['nWaveformEnable'][1]) or
		 (hdr['nWaveformSource'][1] != ABF_EPOCHTABLEWAVEFORM)) and
		(not hdr['nDigitalEnable'])):
		err_str = "No epoch descriptions available."
		raise ValueError(err_str)

	print_str = "Epoch durations, Episode %d:\nEpoch   " % uEpisode

	for ii in range(hdr['nADCNumChannels']):
		ch = hdr['nADCSamplingSeq'][ii]
		ch = hdr['nADCPtoLChannelMap'][ch]
		print_str += " Channel %02d" % ch

	print_str += "\nHolding  "
	for jj in range(hdr['nADCNumChannels']):
		ch = hdr['nADCSamplingSeq'][jj]

		uEpochStart, uEpochEnd = GetEpochLimitsEx(hdr, ch,
			hdr['nActiveDACChannel'], uEpisode, ABFH_FIRSTHOLDING)

		print_str += "%4d->%4d " % (uEpochStart, uEpochEnd)

	for ii in range(ABF_EPOCHCOUNT):
		for jj in range(hdr['nADCNumChannels']):

			ch = hdr['nADCSamplingSeq'][jj]

			uEpochStart, uEpochEnd = GetEpochLimitsEx(hdr, ch,
				hdr['nActiveDACChannel'], 1, ii)
			if uEpochStart < 0:
				break

			if jj == 0:
				print_str += "\n A%d      " % ii
			print_str += "%4d->%4d " % (uEpochStart, uEpochEnd)

	print(print_str + "\n")


#===============================================================================
#	Get Epoch Information
#===============================================================================
def GetEpochIdx(hdr, verbose=0):

	if (((not hdr['nWaveformEnable'][0]) or
		 (hdr['nWaveformSource'][0] != ABF_EPOCHTABLEWAVEFORM)) and
		((not hdr['nWaveformEnable'][1]) or
		 (hdr['nWaveformSource'][1] != ABF_EPOCHTABLEWAVEFORM)) and
		(not hdr['nDigitalEnable'])):
		err_str = "No epoch descriptions available."
		raise ValueError(err_str)

	Epochs = {}

	for nChan in range(hdr['nADCNumChannels']):
		ch = hdr['nADCSamplingSeq'][nChan]
		Epochs[ch] = np.zeros((hdr['lActualEpisodes'], ABF_EPOCHCOUNT+1))

		for nEp in range(hdr['lActualEpisodes']):
			uEpochStart, _ = GetEpochLimitsEx(hdr, ch,
				hdr['nActiveDACChannel'], nEp+1, ABFH_FIRSTHOLDING)

			Epochs[ch][nEp, 0] = uEpochStart

		for nEpoch in range(ABF_EPOCHCOUNT):
			for nEp in range(hdr['lActualEpisodes']):
				uEpochStart, _ = GetEpochLimitsEx(hdr, ch,
					hdr['nActiveDACChannel'], nEp+1, nEpoch)

				Epochs[ch][nEp, nEpoch+1] = uEpochStart

			if np.all(Epochs[ch][:, nEpoch+1] < 0):
				Epochs[ch][:, nEpoch+1] = hdr['lNumSamplesPerEpisode']
				Epochs[ch][:, nEpoch+1] /= hdr['nADCNumChannels']
				break

		Epochs[ch] = Epochs[ch][:, :nEpoch+2].astype(int)


		if verbose > 1:
			for nEp in range(hdr['lActualEpisodes']):
				ShowEpochs(hdr, nEp)
		elif verbose > 0:
			ShowEpochs(hdr, 1)

	return Epochs


#===============================================================================
#	Get Epoch Limits
#===============================================================================
def GetEpochLimitsEx(hdr, nADCChannel, uDACChannel, dwEpisode, nEpoch):

	# print("\nnADCChannel: %d" % nADCChannel)
	# print("uDACChannel: %d" % uDACChannel)
	# print("dwEpisode: %d" % dwEpisode)
	# print("nEpoch: %d" % nEpoch)

	if hdr['nOperationMode'] != ABF_WAVEFORMFILE:
		err_str = "Can only get epoch limits from episodic stimulation files.\n"
		raise ValueError(err_str)

	if ((nADCChannel < 0) and (hdr['nArithmeticEnable'] != 0)):
		nADCChannel = hdr['nArithmeticADCNumA']

	uChannelOffset = GetChannelOffset(hdr, nADCChannel)

	if hdr['nWaveformSource'][uDACChannel] == ABF_WAVEFORMDISABLED:
		err_str = "Epoch not present."
		raise ValueError(err_str)

	uHoldingDuration = GetHoldingDuration(hdr)

	if nEpoch == ABFH_FIRSTHOLDING:
		if uChannelOffset >= uHoldingDuration:
			err_str = "Channel offset is greater than the entire holding "
			raise ValueError(err_str + "duration.")

		nEpochStart = 0
		nEpochEnd = uHoldingDuration - 1
	else:
		if ((nEpoch != ABFH_LASTHOLDING) and
			(not hdr['nEpochType'][uDACChannel][nEpoch])):
			return -1, -1

		nEpochStart = uHoldingDuration
		nEpochDuration = 0

		for ii in range(nEpoch+1):
			if not hdr['nEpochType'][uDACChannel][ii]:
				nEpochDuration = 0
				continue

			nEpochDuration = GetEpochDuration(hdr, uDACChannel,
				dwEpisode, ii)
			nEpochDuration *= hdr['nADCNumChannels']
			nEpochDuration = np.max([nEpochDuration, 0])

			if ii == nEpoch:
				break

			nEpochStart += nEpochDuration

		if nEpoch == ABFH_LASTHOLDING:
			nEpochEnd = int(hdr['lNumSamplesPerEpisode'] - 1)
		elif nEpochDuration > 0:
			nEpochEnd = nEpochStart + nEpochDuration - 1
		else:
			nEpochEnd = nEpochStart

	nEpochStart /= hdr['nADCNumChannels']
	nEpochEnd /= hdr['nADCNumChannels']

	return nEpochStart, nEpochEnd


#===============================================================================
#	Get Channel Offset
#===============================================================================
def GetChannelOffset(hdr, nChannel):

	if nChannel < 0:
		if not hdr['nArithmeticEnable']:
			return None
		nChannel = hdr['nArithmeticADCNumA']

	for nOffset in range(hdr['nADCNumChannels']):
		if hdr['nADCSamplingSeq'][nOffset] == nChannel:
			return int(nOffset)

	return None


#===============================================================================
#	Get Channel Offset
#===============================================================================
def GetHoldingDuration(hdr):

	if ((not hdr['nWaveformEnable'][0]) and
		(hdr['nWaveformSource'][0] == ABF_WAVEFORMDISABLED) and
		(not hdr['nWaveformEnable'][1]) and
		(hdr['nWaveformSource'][1] == ABF_WAVEFORMDISABLED) and
		(not hdr['nDigitalEnable'])):
		return 0

	if hdr['nFileType'] == ABF_CLAMPEX:
		return 6*int(hdr['lNumSamplesPerEpisode']/512)

	return GetHoldingLength(hdr['lNumSamplesPerEpisode'],hdr['nADCNumChannels'])


#===============================================================================
#	Get Holding Duration
#===============================================================================
def GetHoldingLength(nSweepLength, nNumChannels):

	if (nSweepLength % nNumChannels) != 0:
		err_str = "Number of samples per episode does not divide into "
		err_str = "the number of channels."
		raise ValueError(err_str)

	nHoldingCount = nSweepLength / 64
	nHoldingCount -= (nHoldingCount % nNumChannels)

	if nHoldingCount < nNumChannels:
		nHoldingCount = nNumChannels

	return int(nHoldingCount)


#===============================================================================
#	Get Holding Duration
#===============================================================================
def GetEpochDuration(hdr, uDACChannel, uEpisode, nEpoch):

	if (hdr['nULEnable'][uDACChannel] and
		(hdr['nActiveDACChannel'] == uDACChannel) and
		(hdr['nULParamToVary'][uDACChannel] >= ABF_EPOCHINITDURATION) and
		(nEpoch == hdr['nULParamToVary'][uDACChannel] - ABF_EPOCHINITDURATION)):
		raise ValueError("Method to deal with this situation isnt implemented.")

	out = hdr['lEpochInitDuration'][uDACChannel][nEpoch]
	out += int((uEpisode - 1)*hdr['lEpochDurationInc'][uDACChannel][nEpoch])
	return out


#===============================================================================
#	Get Waveform
#===============================================================================
def GetWaveform(hdr, dwEpisode):
	return GetWaveformEx(hdr, hdr['nActiveDACChannel'], dwEpisode)


#===============================================================================
#	Get Waveform (Ex)
#===============================================================================
def GetWaveformEx(hdr, uDACChannel, dwEpisode):

	if hdr['nOperationMode'] != ABF_WAVEFORMFILE:
		err_str = "Can only get epoch limits from episodic stimulation files.\n"
		raise ValueError(err_str)

	if ((not hdr['nWaveformEnable'][uDACChannel]) or
		(hdr['nWaveformSource'][uDACChannel] == ABF_WAVEFORMDISABLED)):
		err_str = "No Waveform to get."
		raise ValueError(err_str)

	if (dwEpisode > hdr['lActualEpisodes']):
		err_str = "Invalid value for input argument 'dwEpisode'"
		err_str += " (%d > %d)" % (dwEpisode, hdr['lActualEpisodes'])
		raise ValueError(err_str)

	if hdr['nWaveformSource'][uDACChannel] == ABF_DACFILEWAVEFORM:
		err_str = "Cannot load waveforms from DAC at the moment."
		raise ValueError(err_str)

	rawwaveform = GenerateWaveform(hdr, uDACChannel, dwEpisode)

	sampPerEpPerChan = int(hdr['lNumSamplesPerEpisode']/hdr['nADCNumChannels'])
	waveform = np.zeros((sampPerEpPerChan, hdr['nADCNumChannels']))

	epochIdx = GetEpochIdx(hdr)[uDACChannel]

	startIdx = 0
	for ii, eIdx in enumerate(epochIdx[0, :-1]):
		eIdx2 = epochIdx[0, ii+1]
		for nChan in range(hdr['nADCNumChannels']):
			tmp = rawwaveform[startIdx+nChan*(eIdx2-eIdx):]
			waveform[eIdx:eIdx2, nChan] = tmp[:eIdx2-eIdx].ravel()
		startIdx = eIdx2*hdr['nADCNumChannels']

	return waveform


#===============================================================================
#	Generate Waveform
#===============================================================================
def GenerateWaveform(hdr, uDACChannel, uEpisode):

	uHoldingDuration = GetHoldingDuration(hdr)
	dHoldingLevel = GetHoldingLevel(hdr, uDACChannel, uEpisode)
	dLevel = dHoldingLevel
	dStartLevel = dHoldingLevel
	nDuration = 0
	bSetToHolding = False

	waveform = np.zeros((hdr['lNumSamplesPerEpisode'], 1))

	if hdr['nAlternateDACOutputState'] != 0:
		if (uEpisode % 2 == 0) and (uDACChannel == 0):
			bSetToHolding = True
		elif (uEpisode % 2 != 0) and (uDACChannel == 1):
			bSetToHolding = True

	if uEpisode > hdr['lEpisodesPerRun']:
		waveform = dHoldingLevel*np.ones_like(waveform)
		return waveform

	waveform[:uHoldingDuration] = dHoldingLevel
	nPoints = uHoldingDuration
	nEndHolding = hdr['lNumSamplesPerEpisode'] - uHoldingDuration

	for nEpoch in range(ABF_EPOCHCOUNT):
		if hdr['nEpochType'][uDACChannel][nEpoch] == ABF_EPOCHDISABLED:
			continue

		nDuration = GetEpochDuration(hdr, uDACChannel, uEpisode, nEpoch)
		nDuration *= hdr['nADCNumChannels']

		if nDuration <= 0:
			continue

		dLevel = GetEpochLevel(hdr, uDACChannel, uEpisode, nEpoch)
		nPeriod = GetEpochTrainPeriod(hdr, uDACChannel, uEpisode, nEpoch)
		nPeriod *= hdr['nADCNumChannels']
		nWidth = GetEpochTrainPulseWidth(hdr, uDACChannel, uEpisode, nEpoch)
		nWidth *= hdr['nADCNumChannels']

		nPoints += nDuration
		if nPoints > hdr['lNumSamplesPerEpisode']:
			err_str = "'nPoints' has become too large; waveform is longer than"
			err_str += " the data."
			raise ValueError(err_str)

		if bSetToHolding:
			pdValue = PopulateStep(nDuration, dHoldingLevel)
		else:
			flag = hdr['nEpochType'][uDACChannel][nEpoch]
			if flag == ABF_EPOCHDISABLED:
				pass
			elif flag == ABF_EPOCHSTEPPED:
				pdValue = PopulateStep(nDuration, dLevel)
			elif flag == ABF_EPOCHRAMPED:
				pdValue = PopulateRamp(nDuration, dStartLevel, dLevel)
			elif flag == ABF_EPOCH_TYPE_RECTANGLE:
				pdValue = PopulateRectangle(nDuration, dStartLevel, dLevel,
					nPeriod, nWidth)
			elif flag == ABF_EPOCH_TYPE_BIPHASIC:
				pdValue = PopulateBiphasic(nDuration, dStartLevel, dLevel,
					nPeriod, nWidth)
			elif flag == ABF_EPOCH_TYPE_TRIANGLE:
				pdValue = PopulateTriangle(nDuration, dStartLevel, dLevel,
					nPeriod, nWidth)
			elif flag == ABF_EPOCH_TYPE_COSINE:
				pdValue = PopulateCosine(nDuration, dStartLevel, dLevel,
					nPeriod)
			elif flag == ABF_EPOCH_TYPE_RESISTANCE:
				pdValue = PopulateResistance(nDuration, dLevel, dHoldingLevel)
			else:
				print(hdr['nEpochType'][uDACChannel][nEpoch])
				raise ValueError("Invalid value for hdr['nEpochType'].")

		waveform[nPoints-nDuration:nPoints, :] = deepcopy(pdValue)

		dStartLevel = dLevel
		nEndHolding -= nDuration

	if not hdr['nInterEpisodeLevel'][uDACChannel]:
		dStartLevel = hdr['fDACHoldingLevel'][uDACChannel]

	nPoints += nEndHolding
	if nPoints > hdr['lNumSamplesPerEpisode']:
		err_str = "'nPoints' has become too large; waveform is longer than"
		err_str += " the data."
		raise ValueError(err_str)

	waveform[nPoints-nEndHolding:] = dStartLevel

	return waveform


#===============================================================================
#	Get Holding Level
#===============================================================================
def GetHoldingLevel(hdr, uDACChannel, uEpisode):

	if uDACChannel >= ABF_WAVEFORMCOUNT:
		err_str = "Invalid value for input argument 'uDACChannel' "
		err_str += "(%d >= %d)" % (uDACChannel, ABF_WAVEFORMCOUNT)
		raise ValueError(err_str)

	fCurrentHolding = hdr['fDACHoldingLevel'][uDACChannel]

	if hdr['nConditEnable'][uDACChannel]:
		if GetPostTrainPeriod(hdr, uDACChannel, uEpisode) > 0.:
			return GetPostTrainLevel(hdr, uDACChannel, uEpisode)
		else:
			return fCurrentHolding

	if not hdr['nInterEpisodeLevel'][uDACChannel]:
		return fCurrentHolding

	for ii in np.arange(ABF_EPOCHCOUNT-1, -1, -1):
		if hdr['nEpochType'][uDACChannel][ii]:
			break

	if ((ii < 0) or (uEpisode < 2)):
		return fCurrentHolding

	return GetEpochLevel(hdr, uDACChannel, uEpisode-1, ii)


#===============================================================================
#	Get Post Training Period
#===============================================================================
def GetPostTrainPeriod(hdr, uDACChannel, uEpisode):

	if uDACChannel >= ABF_WAVEFORMCOUNT:
		err_str = "Invalid value for input argument 'uDACChannel' "
		err_str += "(%d >= %d)" % (uDACChannel, ABF_WAVEFORMCOUNT)
		raise ValueError(err_str)

	if (hdr['nULEnable'][uDACChannel] and 
		(hdr['nULParamToVary'] == ABF_CONDITPOSTTRAINDURATION)):
		raise ValueError("Cannot currently handle this condition.")

	return hdr['fPostTrainPeriod'][uDACChannel]


#===============================================================================
#	Get Post Training Period
#===============================================================================
def GetPostTrainLevel(hdr, uDACChannel, uEpisode):

	if uDACChannel >= ABF_WAVEFORMCOUNT:
		err_str = "Invalid value for input argument 'uDACChannel' "
		err_str += "(%d >= %d)" % (uDACChannel, ABF_WAVEFORMCOUNT)
		raise ValueError(err_str)

	if (hdr['nULEnable'][uDACChannel] and 
		(hdr['nULParamToVary'] == ABF_CONDITPOSTTRAINLEVEL)):
		raise ValueError("Cannot currently handle this condition.")
		
	return hdr['fPostTrainLevel'][uDACChannel]


#===============================================================================
#	Get Epoch Level
#===============================================================================
def GetEpochLevel(hdr, uDACChannel, uEpisode, nEpoch):

	if (hdr['nULEnable'][uDACChannel] and
		(hdr['nULParamToVary'][uDACChannel] >= ABF_EPOCHINITLEVEL) and
		(hdr['nULParamToVary'][uDACChannel] < ABF_EPOCHINITDURATION) and
		(nEpoch == hdr['nULParamToVary'][uDACChannel] - ABF_EPOCHINITLEVEL)):
		raise ValueError("Cannot currently handle this condition.")

	out = hdr['fEpochInitLevel'][uDACChannel][nEpoch]
	out += (uEpisode-1)*hdr['fEpochLevelInc'][uDACChannel][nEpoch]
	return out


#===============================================================================
#	Get Epoch Training Period
#===============================================================================
def GetEpochTrainPeriod(hdr, uDACChannel, uEpisode, nEpoch):

	if (hdr['nULEnable'][uDACChannel] and
		(hdr['nActiveDACChannel'] == uDACChannel) and
		(hdr['nULParamToVary'][uDACChannel] >= ABF_EPOCHTRAINPERIOD) and
		(hdr['nULParamToVary'][uDACChannel] < ABF_EPOCHTRAINPULSEWIDTH) and
		(nEpoch == hdr['nULParamToVary'][uDACChannel] - ABF_EPOCHTRAINPERIOD)):
		raise ValueError("Cannot currently handle this condition.")

	return hdr['lEpochPulsePeriod'][uDACChannel][nEpoch]


#===============================================================================
#	Get Epoch Training Pulse Width
#===============================================================================
def GetEpochTrainPulseWidth(hdr, uDACChannel, uEpisode, nEpoch):

	if (hdr['nULEnable'][uDACChannel] and
		(hdr['nActiveDACChannel'] == uDACChannel) and
		(hdr['nULParamToVary'][uDACChannel] >= ABF_EPOCHTRAINPULSEWIDTH) and
		(hdr['nULParamToVary'][uDACChannel] < ABF_EPOCHTRAINPULSEWIDTH + 
			ABF_EPOCHCOUNT) and
		(nEpoch == hdr['nULParamToVary'][uDACChannel] -
			ABF_EPOCHTRAINPULSEWIDTH)):
		raise ValueError("Cannot currently handle this condition.")

	return hdr['lEpochPulseWidth'][uDACChannel][nEpoch]


#===============================================================================
#	Populate Step
#===============================================================================
def PopulateStep(nDuration, dLevel):
	return dLevel*np.ones((nDuration, 1))


#===============================================================================
#	Populate Ramp
#===============================================================================
def PopulateRamp(nDuration, dStartLevel, dLevel):
	dInc = (dLevel - dStartLevel)/float(nDuration)
	out = np.array([dStartLevel + dInc*(ii+1) for ii in range(nDuration)])
	return out.reshape(len(out), 1)

#===============================================================================
#	Populate Cosine
#===============================================================================
def PopulateCosine(nDuration, dStartLevel, dLevel, nPeriod):
	pdValue = dStartLevel*np.ones((nDuration, 1))

	nPartialPeriod = nDuration % nPeriod
	nEndOfLastPulse = nDuration - nPartialPeriod

	dScaleX = np.pi*2./nPeriod

	pdValue[:nEndOfLastPulse] = np.sin(np.arange(nEndOfLastPulse*dScaleX))
	pdValue *= (dLevel-dStartLevel)
	pdValue += dStartLevel

	return pdValue


#===============================================================================
#	Populate Rectangle
#===============================================================================
def PopulateRectangle(nDuration, dStartLevel, dLevel, nPeriod, nWidth):
	pdValue = dStartLevel*np.ones((nDuration, 1))

	nPartialPeriod = nDuration % nPeriod
	nEndOfLastPulse = nDuration
	if nPartialPeriod < nWidth:
		nEndOfLastPulse -= nPartialPeriod

	for ii in range(nEndOfLastPulse):
		if ii % nPeriod < nWidth:
			pdValue[ii] = dLevel

	return pdValue


#===============================================================================
#	Populate Biphasic
#===============================================================================
def PopulateBiphasic(nDuration, dStartLevel, dLevel, nPeriod, nWidth):
	pdValue = dStartLevel*np.ones((nDuration, 1))

	nPartialPeriod = nDuration % nPeriod
	nEndOfLastPulse = nDuration
	if nPartialPeriod < nWidth:
		nEndOfLastPulse -= nPartialPeriod

	for ii in range(nEndOfLastPulse):
		if (ii % nPeriod) < nWidth/2.:
			pdValue[ii] = dLevel
		elif (ii % nPeriod) < nWidth:
			pdValue[ii] = dStartLevel - (dLevel - dStartLevel)

	return pdValue


#===============================================================================
#	Populate Triangle
#===============================================================================
def PopulateTriangle(nDuration, dStartLevel, dLevel, nPeriod, nWidth):
	dRiseSlope = (dLevel-dStartLevel)/nWidth
	dFallSlope = (dStartLevel-dLevel)/(nPeriod-nWidth)

	if nPeriod == 0:
		raise ValueError("Input parameter 'nPeriod' cannot be zero.")

	nPartialPeriod = nDuration % nPeriod
	nEndOfLastPulse = nDuration - nPartialPeriod

	for ii in range(nEndOfLastPulse):
		nSampleWithinCurrentPeriod = ii % nPeriod

		if nSampleWithinCurrentPeriod < nWidth:
			pdValue[ii] = dStartLevel + dRiseSlope*nSampleWithinCurrentPeriod
		else:
			pdValue[ii] = dLevel +dFallSlope*(nSampleWithinCurrentPeriod-nWidth)

	return pdValue


#===============================================================================
#	Populate Resistance
#===============================================================================
def PopulateResistance(nDuration, dLevel, dHolding):
	return PopulateRectangle(nDuration, dLevel, dHolding, nDuration,
		nDuration/2)


#===============================================================================
#	Scale DAC
#===============================================================================
def GetDACtoUUFactors(hdr, nChannel):
	if nChannel >= 4:
		raise ValueError("Invalid value for 'nChannel'.")

	fScaleFactor = hdr['fDACScaleFactor'][nChannel]
	fCalibrationFactor = hdr['fDACCalibrationFactor'][nChannel]
	fCalibrationOffset = hdr['fDACCalibrationOffset'][nChannel]

	fOutputRange = hdr['fDACRange'] * fScaleFactor
	fOutputOffset = 0.

	fDACtoUUFactor = fOutputRange / hdr['lDACResolution']*fCalibrationFactor
	fDACtoUUShift = fOutputOffset + fDACtoUUFactor*fCalibrationOffset
	return fDACtoUUFactor, fDACtoUUShift


#===============================================================================
#	Get Timebase
#===============================================================================
def GetTimebase(hdr, fTimeOffset):
	uSamplesPerSweep = hdr['lNumSamplesPerEpisode']

	if ((hdr['fADCSecondSampleInterval'] == 0.) or
		(hdr['fADCSampleInterval'] == hdr['fADCSecondSampleInterval']) or
		(hdr['nOperationMode'] != ABF_WAVEFORMFILE)):
		uClockChange = uSamplesPerSweep
	else:
		if hdr['lClockChange'] > 0:
			uClockChange = hdr['lClockChange']
		else:
			uClockChange = uSamplesPerSweep / 2

		uClockChange -= (uClockChange % hdr['nADCNumChannels'])

	uSamplesPerSweep = int(uSamplesPerSweep/hdr['nADCNumChannels'])

	dTimeInc = GetFirstSampleInterval(hdr)*hdr['nADCNumChannels']/1e3
	timebase = np.zeros((uSamplesPerSweep, 1))
	for ii in range(uSamplesPerSweep):
		timebase[ii] = ii*dTimeInc
		if ii > uClockChange:
			break

	dTimeInc = GetSecondSampleInterval(hdr)*hdr['nADCNumChannels']/1e3
	for jj in range(ii+1, uSamplesPerSweep):
		timebase[jj] = jj*dTimeInc

	return timebase


#===============================================================================
#	Get First Sample Interval
#===============================================================================
def GetFirstSampleInterval(hdr):
	return GetSampleInterval(hdr, 1)


#===============================================================================
#	Get Second Sample Interval
#===============================================================================
def GetSecondSampleInterval(hdr):
	return GetSampleInterval(hdr, 2)


#===============================================================================
#	Get Sample Interval
#===============================================================================
def GetSampleInterval(hdr, uInterval):
	if (uInterval != 1) and (uInterval != 2):
		raise ValueError("Invalid entry for argument 'uInterval'")

	fInterval = 0
	if uInterval == 1:
		fInterval = hdr['fADCSampleInterval']
	else:
		fInterval = hdr['fADCSecondSampleInterval']

	dInterval = int(fInterval * hdr['nADCNumChannels'] * 10 + 0.5)
	dInterval /= 10 * hdr['nADCNumChannels']

	return dInterval


if __name__ == "__main__":

	filename = "2012_02_01_0003.abf"

	data, hdr = ABF_read(filename, verbose=0)

	epochIdx = GetEpochIdx(hdr)

	waveform = GetWaveformEx(hdr, 0, 1)

	print(waveform[epochIdx[0][0, :-1]])

	ShowEpochs(hdr, 1)




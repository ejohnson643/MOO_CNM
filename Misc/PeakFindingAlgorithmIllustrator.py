"""
================================================================================
	Action Potential Locator Algorithm Illustrator
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, May 8, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This script will generate figures explaining how the peak finding algorithm
	in ephys_obj ("getSpikeIdx") works.

	Key elements that will be illustrated:
	 - Rolling Median Subtraction
	 - Median filter edge correction
	 - Sensitivity of peak detection to the size of the median filter
	 - Figure illustrating the parameters thresh, prom, wlen, and how a slope
	   parameter is enforced
	 - Figure showing histograms and where thresh, prom can be set.
	 - Figure showing how the number of peaks changes at max wlen as thresh and
	   prom are modified
	 - Figure showing how the number of peaks changes for given thresh, prom,
	   as wlen is reduced.
	 - Figure showing how for given prom, wlen, the number of peaks changes as 
	   a function of thresh (also violin plots vs thresh?)
	   

================================================================================
================================================================================
"""
import collections
from copy import deepcopy
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import seaborn as sns
import scipy.interpolate as intrp
import scipy.signal as sig
import scipy.stats as st
from skimage.filters import threshold_otsu as otsu

from Misc.dip import *

import Objectives.Electrophysiology.ephys_objs as epo
import Objectives.Electrophysiology.ephys_util as epu

import Utility.ABF_util as abf
import Utility.DataIO_util as DIO
import Utility.runfile_util as rfu
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
## Text Processing Functions for Loading CSV
################################################################################
def strip(text):
	try:
		return text.strip()
	except AttributeError:
		return text

def make_float(text):
	try:
		return float(strip(text))
	except TypeError:
		return strip(text)
	except ValueError:
		text = strip(text)
		if len(text) == 0:
			return np.NaN
		else:
			return text

def check_list(text):
	text = strip(text)
	if text.lower() == 'all':
		return [0, -1]
	elif text.lower() == 'none':
		return []
	else:
		return make_list(text)

def make_list(text):
	strip(text)
	text = text[1:-1].split(",")
	text = [int(t.strip()) for t in text]
	return text

def convert_time(text):
	hours, minutes = text.split(":")
	hours = int(hours)
	minutes = float(minutes)/60.
	return hours + minutes

def isWT(entry):
	if entry['Geno'] == 'WT':
		return True
	else:
		return False


################################################################################
## Runtime Commands
################################################################################
if __name__ == "__main__":

	plt.close('all')
	sns.set({'legend.frameon':True}, color_codes=True)
	matplotlib.rc("font", size=20)
	matplotlib.rc("xtick", labelsize=16)
	matplotlib.rc("ytick", labelsize=16)
	matplotlib.rc("axes", labelsize=20)
	matplotlib.rc("axes", titlesize=24)
	matplotlib.rc("legend", fontsize=18)
	matplotlib.rc("figure", titlesize=24)
"""
================================================================================
	Flourakis Data Reviewer
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, July 24, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This script will contain precedures to generate a list of all of Matt's
	abf recordings, which we can then parse and assess.

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

	figDir = "./Figures/FlourakisDataReview/"

	gen_new_data = True
	check_for_data = True

	plotEveryone = True

	infoDir = "./Runfiles/RedCheck_Overview/"

	infoDict = rfu.getInfo(infoDir, verbose=1)

################################################################################
## Procedure Flags
################################################################################

	make_all_abf_names = True


################################################################################
## Make csv of all ABF Files, generic info
################################################################################

	dataDir = DIO.get_data_dir(dataDir="")

	# if make_all_abf_names:

	# 	for yearFolder in os.listdir(dataDir):

	# 		yearPath = os.path.join(dataDir, yearFolder)

	# 		months = 

	# 		for month
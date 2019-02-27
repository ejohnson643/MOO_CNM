"""
================================================================================
	Evolutionary Algorithm Population Class
================================================================================

	Author: Eric Johnson
	Date Created: Monday, November 27, 2017
	Date Revised: Tuesday, February 26, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file will contain the class for the Evolutionary Algorithm (EA) 
	Population.

	The "Population" class will inherit the list class and will have a few
	methods allowing for it to be easily manipulated in the context of an EA.
	Specifically, its initialization will be automated, its parameters checked
	at instantiation, and it will have built-in evaluation and mutation methods.

================================================================================
================================================================================
"""

from copy import deepcopy
import importlib.util as imputl
import json
import numpy as np
import os
import pickle as pkl
import scipy.stats as st

import Utility.runfile_util as rfu
import Utility.utility as utl



class Population(list):

	def __init__(self, infoDict):

		infoDict = rfu.checkInfo(infoDict, verbose=0)

		## dataFeat = getDataFeatures(infoDict)







if __name__ == "__main__":

	infoDir = "./Runfiles/HH_Test/"

	infoDict = rfu.getInfo(infoDir, verbose=1)

	pop = Population(infoDict)
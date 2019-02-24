"""
================================================================================
	Evolutionary Algorithm Individual Class
================================================================================

	Author: Eric Johnson
	Date Created: Monday, November 27, 2017
	Date Revised: Wednesday, February 20, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains the class definition "Individual" for an individual
	solution to an optimization problem in an Evolutionary Algorithm (EA).  

	In this case, the "Individual" class will inherit the standard Python 
	"dict", indicating that the solution parameters are stored as a dictionary.
	To add functionality to a dictionary that allows it to be used in an EA, I 
	give the Individual the ability to do the following:
		- Mutate: modify its parameters using a provided function.
		- Crossover: swap parameter values with another individual.
		- Evaluate Objectives: evaluate model performance using individual's
				parameters according to supplied objectives.
		- Compare to Individuals: assess whether the individual is 'dominated'
				by another individual (determine whether individual is on the
				Pareto front).

	For ease of use, I will also define a __str__ method to make the
	individual's parameters easy to see, and I will institute some parameter
	checks.

	While I hope to keep this class fairly general in application, since this
	code is intended to be used to fit a neuron model, I may sacrifice
	generality for brevity at some points.  To this end, I will create an
	Individual subclass for each model that inherits this class.  That 
	model.Individual subclass will contain most of the specific actions for that
	model.  In this file, I will institute generic routines or example routines.
	
	This class will *ASSUME* that infoPath has been checked and that a model has
	been loaded.

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


class Individual(dict):

	def __init__(self,
		info,
		popID=None,
		parents=None,
		verbose=0):

		#=======================================================================
		#	Check inputs
		#=======================================================================
		verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)
		self.verbosity = deepcopy(verbose)

		err_str = "Argument 'info' must be a dictionary.  (Use Utility.runfile_"
		err_str += "util(infoPath) to load.)"
		assert isinstance(info, dict), err_str

		self.info = deepcopy(info)

		if popID is not None:
			if not utl.is_floatable(popID, name='Ind.popID',
				verbose=self.verbosity):
				if verbose > 1:
					warn_str = "keyword argument 'popID' is not floatable, "
					warn_str += "setting to 'None'"
					print(warn_str)
				popID = None
			else:
				if not isinstance(popID, int):
					if verbose > 1:
						warn_str = "keyword argument 'popID' is not an integer,"
						warn_str += " setting to integer."
						print(warn_str)
					popID = int(popID)
		
		self.popID = deepcopy(popID)

		if parents is not None:
			err_str = "Keyword argument 'parents' is not list."
			assert isinstance(parents, list), err_str
		else:
			parents = []

		self.parents = deepcopy(parents)

		#=======================================================================
		#	Use modelDir to initialize param Name:Value
		#=======================================================================

		modelDict = self.load_model()

		self.init_params(modelDict)

		return


	def load_model(self):

		if self.verbosity:
			print("Loading model!")

		modelPath = os.path.join(self.info['modelDir'], "modelDict.pkl")

		if self.verbosity > 1:
			print(f"Trying to load {modelPath}...")

		modelDict = self._load_model_dict(modelPath)

		if self.verbosity > 1:
			print(f"Loaded model {modelDict['name']}!")

		self.model = modelDict['model']

		return modelDict.copy()

	def _load_model_dict(self, modelPath):

		jsonPath = os.path.join(self.info['modelDir'], "model.json")
		with open(jsonPath, "r") as f:
			modelDict = json.load(f)

		indSpec = imputl.spec_from_file_location("model",
			os.path.join(self.info['modelDir'], "model.py"))
		foo = imputl.module_from_spec(indSpec)
		indSpec.loader.exec_module(foo)
		model = foo.model

		modelDict['model'] = model

		return modelDict


	def init_params(self, modelDict):



		return
		# if self.verbosity:
		# 	print("Initializing parameters!")

		# paramPath = os.path.join(self.info['modelDir'], "paramDict.pkl")
		# if self.verbosity > 1:
		# 	print(f"Trying to load {paramPath}...")

		# try:
		# 	with open(paramPath, "wb") as f:
		# 		paramDict = 


if __name__ == "__main__":

	infoPath = "./Runfiles/HH_Test/"

	info = rfu.getInfo(infoPath, verbose=1)

	ind = Individual(info)








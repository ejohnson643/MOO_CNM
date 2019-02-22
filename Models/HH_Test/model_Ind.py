"""
================================================================================
	HH_Test Individual Subclass
================================================================================

	Author: Eric Johnson
	Date Created: Wednesday, February 20, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file contains the subclass definition "Individual" for an individual
	solution to an optimization problem in an Evolutionary Algorithm (EA) using
	the HH_Test neuron model.

	Details....

================================================================================
================================================================================
"""
import importlib.util as imputl
import json
import numpy as np
import os
import pickle as pkl

from Base.Individual import Individual as BaseInd
import Utility.runfile_util as rfu

################################################################################
#	Set Model Subclass
################################################################################

class Individual(BaseInd):

	def __init__(self, info,
		popID=None,
		parents=None,
		verbose=0):

		BaseInd.__init__(self, info,
			popID=popID,
			parents=parents,
			verbose=verbose)

		return

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


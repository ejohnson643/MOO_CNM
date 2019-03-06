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
		infoDict,
		popID=None,
		parents=None):

	############################################################################
	#	Check inputs
	############################################################################

		## Check the input infoDict, create self.infoDict attribute.
		self._checkInfo(infoDict)

		## Check and set self.popID
		if popID is not None:
			if not utl.is_floatable(popID, name='Ind.popID',
				verbose=self.verbose):
				if self.verbose > 1:
					warn_str = "keyword argument 'popID' is not floatable, "
					warn_str += "setting to 'None'"
					print(warn_str)
				popID = None
			else:
				if not isinstance(popID, int):
					if self.verbose > 1:
						warn_str = "keyword argument 'popID' is not an integer,"
						warn_str += " setting to integer."
						print(warn_str)
					popID = int(popID)
		
		self.popID = deepcopy(popID)

		## Check and set self.parents to keep track of who contributed to whom
		if parents is not None:
			err_str = "Keyword argument 'parents' is not list."
			assert isinstance(parents, list), err_str
		else:
			parents = []

		self.parents = deepcopy(parents)

	############################################################################
	#	Use modelDir to initialize param Name:Value
	############################################################################

		## Load model dictionary with model func and parameters
		modelDict = self.load_model()

		## Cross-reference infoDict and modelDict to ensure that all parameters
		## are legal and non-contradictory.
		infoDict = self._checkModelInfo(infoDict, modelDict)

		## Initialize the parameters of Individual using the infoDict.
		self.init_params(infoDict)

		return


	############################################################################
	#	Check Info Dict, Store Useful Info (Private Method)
	############################################################################
	def _checkInfo(self, infoDict):

		## This should already have been checked, so do it quietly.
		infoDict = rfu.checkInfo(infoDict)

		## Set individual's verbosity
		self.verbose = infoDict['ind']['verbose']

		## Check the runtime parameters specifically to make sure they match
		## with model.ind (if, for example, we want a model to have different
		## initialization, mutation, or xover processes).
		infoDict = self._checkRuntimeParams(infoDict)

		## Create infoDict attribute with limited contents compared to infoDict
		self.infoDict = {}

		## Store directories
		self.infoDict['infoDir'] = infoDict['infoDir']
		self.infoDict['modelDir'] = infoDict['modelDir']
		self.infoDict['logDir'] = infoDict['logDir']
		self.infoDict['cpDir'] = infoDict['cpDir']

		## Store mutation parameters
		self.mutDict = infoDict['mutation'].copy()

		## Store crossover parameters
		self.xOverDict = infoDict['crossover'].copy()

		## Store simulation parameters
		self.simDict = infoDict['simulation'].copy()

		## Store objectives parameters
		self.objDict = infoDict['objectives'].copy()

		return


	############################################################################
	#	Check Info Runtime Parameters (Private Method)
	############################################################################
	def _checkRuntimeParams(self, infoDict):

		return infoDict


	############################################################################
	#	Check Info Model Dict (Private Method)
	############################################################################
	def _checkModelInfo(self, infoDict, modelDict):
		"""
			This function will simply check that any runtime parameters are
			formatted correctly.  When the model is loaded, these can be checked
			again and then when actually implementing parameter selection, we 
			should use model.json, then check back with runfile/info.json to 
			make sure all parameters are in bounds.

			Particularly, this function will make sure that any modifications
			of the parameters and their bounds are legal.  This returns an
			infoDict that can be used to initialize an Individual.
		"""

		try:
			mDict = infoDict['modelParams'].copy()
		except:
			if self.verbose > 1:
				warn_str = "No 'modelParams' parameters have been set, assuming"
				print(warn_str + " defaults...")
			infoDict['modelParams'] = modelDict['params'].copy()
			return infoDict

		if mDict == 'all':
			infoDict['modelParams'] = modelDict['params'].copy()
			return infoDict
		else:
			err_str = "infoDict['modelParams'] must be a dictionary!"
			assert isinstance(mDict, dict), err_str

		if len(mDict) > 0:
			for key in mDict:
				try:
					_ = modelDict['params'][key]
				except KeyError:
					err_str = "No parameter '{key}' in model!"
					raise ValueError(err_str)

				tmp = mDict[key]
				if not isinstance(tmp, list):
					err_str = f"infoDict['modelParams']['{key}']"
					err_str += "is not floatable!"
					assert utl.is_floatable(tmp,
						name=f"infoDict['modelParams']['{key}']",
						verbose=self.verbose), err_str

					minmax = modelDict['params'][key][1:]
					infoDict['modelParams'][key] = [tmp, minmax[0], minmax[1]]

				else:
					err_str = f"infoDict['modelParams']['{key}'] must be a list"
					err_str += " with 3 elements!"
					assert len(tmp) == 3, err_str

					err_str = f"min > max for infoDict['modelParams']['{key}']!"
					assert tmp[1] <= tmp[2], err_str

		for key in modelDict['params']:
			try:
				_ = mDict[key]
			except KeyError:
				infoDict['modelParams'][key]= deepcopy(modelDict['params'][key])

		try:
			outCol = infoDict['simulation']['outCol']
			assert outCol in modelDict['colDict'].keys()
		except KeyError:
			infoDict['simulation']['outCol'] = modelDict['colDict'].keys()[0]
		except AssertionError:
			err_str = "Unknown simulation variable "
			err_str += f"{infoDict['simulation']['outCol']}..."
			raise ValueError(err_str)

		return infoDict


	############################################################################
	#	Load Model Dictionary
	############################################################################
	def load_model(self):

		if self.verbose:
			print("Loading model!")

		if self.verbose > 1:
			print(f"Trying to load model from {self.infoDict['modelDir']}...")

		modelDict = self._load_model_dict()

		if self.verbose > 1:
			print(f"Loaded model {modelDict['name']}!")

		return modelDict.copy()


	############################################################################
	#	Load Model Dict (Private Method)
	############################################################################
	def _load_model_dict(self):

		jsonPath = os.path.join(self.infoDict['modelDir'], "model.json")
		with open(jsonPath, "r") as f:
			modelDict = json.load(f)

		indSpec = imputl.spec_from_file_location("model",
			os.path.join(self.infoDict['modelDir'], "model.py"))
		foo = imputl.module_from_spec(indSpec)
		indSpec.loader.exec_module(foo)
		model = foo.model

		modelDict['model'] = model
		self.model = modelDict['model']

		return modelDict


################################################################################
#	Initialize Parameters
################################################################################
	def init_params(self, infoDict):

		for key in infoDict['modelParams']:
			if len(np.unique(infoDict['modelParams'][key])) == 1:
				self[key] = deepcopy(infoDict['modelParams'][key][0])

			elif not infoDict['ind']['randomInit']:
				self[key] = deepcopy(infoDict['modelParams'][key][0])

			else:
				self[key] = self._set_init_param(infoDict, key)

		return


	############################################################################
	#	Initialize Random Parameters (Private Method)
	############################################################################
	def _set_init_param(self, infoDict, key):
		"""
			This is the default random init function:

			Uses randn about the initial guess with 5% of the max-min difference
			as the width of the normal.

			For conductances, does this on the *log* of the value.
		"""

		if key[0] == 'g':
			tmp = np.log(infoDict['modelParams'][key])

			stddev = infoDict['ind']['sigma']*(tmp[2] - tmp[1])

			out = tmp[0] + st.norm.rvs(0, scale=stddev)

			if out < tmp[1]:
				out = deepcopy(tmp[1])

			if out > tmp[2]:
				out = deepcopy(tmp[2])

			return np.exp(out)

		else:
			tmp = infoDict['modelParams'][key]

			stddev = infoDict['ind']['sigma']*(tmp[2] - tmp[1])

			out = tmp[0] + st.norm.rvs(0, scale=stddev)

			if out < tmp[1]:
				out = deepcopy(tmp[1])

			if out > tmp[2]:
				out = deepcopy(tmp[2])

			return out


################################################################################
#	Mutation Operators
################################################################################
	def mutate(self):
		"""
			Here we will interpret the mutation "method" and mutate the
			individual *in-place*

			The parameters for mutation are grabbed from the 'mutDict', which is
			created at initialization of the Individual.

			Specifically, if mutDict['method'] = 'normal', then the mutation
			operator is a uniform Gaussian (normal) perturbation to a random
			selection of parameters.  The method will only attempt to perturb 
			parameters that are not fixed (i.e. that don't have lb=ub).  The
			width of this Gaussian is given by mutDict['sigma'], which is the 
			percentage of the total range (ub - lb) to use as a width.  The
			default is 0.1 ( = 10% of total range).

			If mutDict['method'] = 'polynomial', then the polynomial method
			introduced in the original NSGA-II algorithm by Deb will be used.
			This method has an equal likelihood of moving a parameter to the
			left or right in its total range, allowing for greater exploration
			of all of state space as well as allowing for points to remain near
			the boundaries.
		"""

		if self.verbose:
			print(f"\nMutating Individual {self.popID}!")

		infoDict = rfu.getInfo(self.infoDict['infoDir'], verbose=self.verbose)
		modelDict = self.load_model()
		infoDict = self._checkModelInfo(infoDict, modelDict)

		if self.mutDict['method'] == 'normal':
			self.mutGaussian(infoDict['modelParams'].copy())

		elif self.mutDict['method'] == 'polynomial':
			self.mutPolynomialBounded(infoDict['modelParams'].copy())

		else:
			err_str = f"Unknown mutation method {self.mutDict['method']}."
			raise ValueError(err_str)

	############################################################################
	#	Gaussian Mutation
	############################################################################
	def mutGaussian(self, paramBounds):
		"""This method is based on the DEAP implementation found at 
		"https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py"
		where parameters of the Individual are selected to be mutated with 
		probability indpb = mutDict['NMut']/noParams (so that on average NMut 
		params are mutated).  The selected parameters are then mutated with a
		Gaussian centered at the current value with width = sigma*(ub-lb).
		"""

		varyParams = {}
		for key in paramBounds:
			if len(np.unique(paramBounds[key])) == 1:
				continue
			else:
				bnds = paramBounds[key]
				varyParams[key] = deepcopy(bnds)
				sig = self.mutDict['sigma']*(bnds[2]-bnds[1])
				varyParams[key] += [sig]

				print(key, varyParams[key])

		NParams = len(varyParams)
		indPb = np.min([self.mutDict['NMut']/float(NParams), 1.])

		for key in varyParams:
			if np.random.rand() < indPb:

				if key[0] == 'g':
					param = np.log10(varyParams[key])

				else:
					param = np.log10(varyParams[key])

				print(key, param)

				sigma = varyParams[key][-1]

				print("sigma", sigma)

				perturb = st.norm.rvs(0, scale=sigma)

				print("perturb", perturb)

				if key[0] == 'g':
					self[key] = np.log10(self[key] + perturb)
				else:
					self[key] += perturb

				print(key, self[key])

			if self[key] < param[1]:
				self[key] = param[1]

			if self[key] > param[2]:
				self[key] = param[2]

			if key[0] == 'g':
				self[key] = 10**self[key]

		return








################################################################################
#	Run if __name__=="__main__"
################################################################################
if __name__ == "__main__":

	infoPath = "./Runfiles/HH_Test/"

	infoDict = rfu.getInfo(infoPath, verbose=2)

	ind = Individual(infoDict)

	for key in ind:
		if key[0] == 'g':		
			print(f"{key}: {ind[key]}")

	ind.mutate()

	for key in ind:
		if key[0] == 'g':		
			print(f"{key}: {ind[key]}")








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

################################################################################
#	Initialize Object
################################################################################
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
		modelDict = self.load_model(verbose=self.verbose)

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

		## First check that the specified "outCol" is actually a parameter in
		## the model.
		try:
			outCol = infoDict['simulation']['outCol']
			assert outCol in modelDict['colDict'].keys()
		## If not specified, set to the first parameter in the model
		except KeyError:
			infoDict['simulation']['outCol'] = modelDict['colDict'].keys()[0]
		## Otherwise raise an error
		except AssertionError:
			err_str = "Unknown simulation variable "
			err_str += f"{infoDict['simulation']['outCol']}..."
			raise ValueError(err_str)

		## Then check the given model parameters to vary.
		## If no parameters have been specified to be varied, then revert to 
		## model defaults from modelDict (from model.json)
		try:
			mDict = infoDict['modelParams'].copy()
		except:
			if self.verbose > 1:
				warn_str = "No 'modelParams' parameters have been set, assuming"
				print(warn_str + " defaults...")
			infoDict['modelParams'] = modelDict['params'].copy()
			return infoDict

		## If infoDict['modelParams'] is "all", then use model defaults and
		## vary all allowed parameters.
		if mDict == 'all':
			infoDict['modelParams'] = modelDict['params'].copy()
			return infoDict

		else:  ## Otherwise, check that what was given is a dictionary.
			err_str = "infoDict['modelParams'] must be a dictionary!"
			assert isinstance(mDict, dict), err_str

		## If list of parameters to vary is a non-empty dictionary...
		if len(mDict) > 0:
			for key in mDict:

				## Check that each key is a parameter in the model.
				try:
					_ = modelDict['params'][key]
				except KeyError:
					err_str = "No parameter '{key}' in model!"
					raise ValueError(err_str)

				## Check if the value is a list...
				tmp = mDict[key]
				if not isinstance(tmp, list):

					if tmp is None:
						mP = modelDict['params'][key].copy()
						infoDict['modelParams'][key] = mP
						continue

					err_str = f"infoDict['modelParams']['{key}']"
					err_str += "is not floatable!"
					## Check that it is floatable
					assert utl.is_floatable(tmp,
						name=f"infoDict['modelParams']['{key}']"), err_str

					## Set the bounds using the defaults
					minmax = modelDict['params'][key][1:]
					infoDict['modelParams'][key] = [tmp, minmax[0], minmax[1]]

				## If it is a list, check that it is a 3-element list of floats
				## and check that the given minimum is less than the max.
				else:
					err_str = f"infoDict['modelParams']['{key}'] must be a list"
					err_str += " with 3 elements!"
					assert len(tmp) == 3, err_str

					for ii, el in enumerate(tmp):
						err_str = f"Element {ii} of "
						err_str += f"infoDict['modelParams']['{key}'] is not "
						err_str += "float able!"
						assert utl.is_floatable(el), err_str

					err_str = f"min > max for infoDict['modelParams']['{key}']!"
					assert tmp[1] <= tmp[2], err_str

		## Now check the inverse, that every parameter in modelDict is in 
		## infoDict['modelParam']
		for key in modelDict['params']:
			try:
				_ = mDict[key]
			except KeyError:
				infoDict['modelParams'][key]= deepcopy(modelDict['params'][key])

		return infoDict


	############################################################################
	#	Load Model Dictionary
	############################################################################
	def load_model(self, verbose=0):

		if verbose:
			print("Loading model!")

		if verbose > 1:
			print(f"Trying to load model from {self.infoDict['modelDir']}...")

		modelDict = self._load_model_dict()

		if verbose > 1:
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
		"""Individual.mutate(): Mutates Individual object *in-place.*

		The method and parameters for mutation are grabbed from the 'mutDict', 
		which is created at initialization of the Individual.

		Specifically, if mutDict['method'] = 'normal', then the mutation
		operator is a uniform Gaussian (normal) perturbation to a random
		selection of parameters.  

		If mutDict['method'] = 'polynomial', then the polynomial method
		introduced in the original NSGA-II algorithm by Deb will be used.
		This method has an equal likelihood of moving a parameter to the
		left or right in its total range, allowing for greater exploration
		of all of state space as well as allowing for points to remain near
		the boundaries.
		"""

		if self.verbose:
			if self.popID is not None:
				print(f"\nMutating Individual {self.popID}!\n")
			else:
				print(f"\nMutating Individual!\n")

		# infoDict = rfu.getInfo(self.infoDict['infoDir'], verbose=self.verbose)
		# modelDict = self.load_model()
		# infoDict = self._checkModelInfo(infoDict, modelDict)

		paramBounds = self._getParamBounds()

		if self.mutDict['method'] == 'normal':
			self.mutGaussian(paramBounds.copy())

		elif self.mutDict['method'] == 'polynomial':
			self.mutPolynomialBounded(paramBounds.copy())

		else:
			err_str = f"Unknown mutation method {self.mutDict['method']}."
			raise ValueError(err_str)

	############################################################################
	#	Gaussian Mutation
	############################################################################
	def mutGaussian(self, paramBounds):
		""" Individual.mutGaussian(): Mutates Individual object in-place using a
		Gaussian perturbation to a random selection of parameters.

		This method is loosely based on the DEAP implementation found at 
		"https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py"
		where parameters of the Individual are selected to be mutated with 
		probability indpb = mutDict['NMut']/noParams (so that on average NMut 
		params are mutated).  

		The selected parameters are then mutated with a Gaussian centered at 
		the current value with width = sigma*(ub-lb). The width of this Gaussian
		is given by mutDict['sigma'], which is the percentage of the total range
		(ub - lb) to use as a width.  The default is 0.1 ( = 10% of total range).

		Also, by default, conductance parameters, which are assumed to start
		with 'g' have the Gaussian perturbation applied to the *log* of their
		values so that they can explore a larger parameter range more easily.

		Inputs:
		=======
		paramBounds:	(Dict)	Dictionary with same keys as Individual object,
								but where each parameter key yields a list with
								the parameter's [default, lower bound, upper 
								bound].  

								(Can be obtained with self._getParamBounds())
		"""

		## Find parameters that are subject to variation
		varyParams = {}
		for key in paramBounds:
			if len(np.unique(paramBounds[key])) == 1:
				continue
			else:
				bnds = paramBounds[key]
				varyParams[key] = deepcopy(bnds)
				varyParams[key][0] = self[key]

				## If a conductance parameter, want to operate on log scale
				if key[0] == 'g':
					sig = self.mutDict['sigma']*np.log10(bnds[2]-bnds[1])
				## Else sig = fraction of total range
				else:
					sig = self.mutDict['sigma']*(bnds[2]-bnds[1])
				varyParams[key] += [sig]

		if self.verbose > 1:
			print("Variable Parameters:")
			for key in varyParams:
				print_str = f"{key}:\t"
				print_str += ",\t".join([f"{el:7.3g}"
					for el in varyParams[key]])
				print(print_str)

		## Calculate probability of individual parameter being mutated
		NParams = len(varyParams)
		indPb = np.min([self.mutDict['NMut']/float(NParams), 1.])

		if self.verbose > 1:
			print(f"\nThere are {NParams} to vary, " + 
				f"so mutation pb = {indPb:.2f}")

		## Iterate through parameters
		for key in varyParams:
			## If not selected for mutation, continue
			if np.random.rand() >= indPb:
				continue

			## If a conductance parameter, want to operate on log scale
			if key[0] == 'g':
				param = list(np.log10(varyParams[key][:-1]))
				param += [varyParams[key][-1]]
			## Else sig = fraction of total range
			else:
				param = varyParams[key]

			## Generate normal perturbation with correct width
			perturb = st.norm.rvs(0, scale=varyParams[key][-1])

			self[key] += perturb

			## Check bounds
			self[key] = min(max(self[key], param[1]), param[2])

			## If conductance parameter, return from log-scaling
			if key[0] == 'g':
				self[key] = 10**self[key]

		return


	############################################################################
	#	Bounded Polynomial Mutation
	############################################################################
	def mutPolynomialBounded(self, paramBounds):
		"""Individual.mutPolynomialBounded(): Mutates Individual object in-place
		using the bounded polynomial method implemented in the original NSGA-II
		algorithm by Deb (2002).

		This method has an equal likelihood of moving a parameter to the left or
		right in its total range, allowing for greater exploration of all of
		state space as well as allowing for points to remain near the
		boundaries.  This method requires the specification of a "crowding
		degree" parameter called eta, where a large eta will produce mutants
		that are similar to their parents, while smaller eta produce mutatnts
		that are more different from their parents.  The default value is 20.

		This method also selects parameters to mutate uniformly with a
		probability so that mutDict['NMut'] out of all varying parameters are
		mutated.  Only parameters that are not fixed (i.e. that don't have their
		lower bound == upper bound) are subject to this selection.

		This code is based on the implementation of this method in the deap
		package: https://github.com/DEAP/

		Inputs:
		=======
		paramBounds:	(Dict)	Dictionary with same keys as Individual object,
								but where each parameter key yields a list with
								the parameter's [default, lower bound, upper 
								bound].  

								(Can be obtained with self._getParamBounds())
		"""
		
		## Find parameters that are subject to variation
		varyParams = {}
		for key in paramBounds:
			if len(np.unique(paramBounds[key])) == 1:
				continue
			else:
				bnds = paramBounds[key]
				varyParams[key] = deepcopy(bnds)
				varyParams[key][0] = self[key]

		if self.verbose > 1:
			print("Variable Parameters:")
			for key in varyParams:
				print_str = f"{key}:\t"
				print_str += ",\t".join([f"{el:7.3g}"
					for el in varyParams[key]])
				print(print_str)

		## Get eta parameter
		eta = self.mutDict['eta']
		mutPow = 1. / (eta + 1.)

		if self.verbose > 1:
			print(f"\neta = {eta}\t(mut pow = {mutPow:.3g})")

		## Calculate probability that an individual parameter will be varied
		NParams = len(varyParams)
		indPb = np.min([self.mutDict['NMut']/float(NParams), 1.])

		if self.verbose > 1:
			print(f"\nThere are {NParams} to vary, " + 
				f"so mutation pb = {indPb:.2f}")

		## Iterate through 
		for ii, key in enumerate(varyParams):

			## If not selected for mutation, continue
			if np.random.rand() >= indPb:
				continue

			x, lb, ub = varyParams[key]

			## Draw random number
			rand = np.random.rand()

			## If rand < 0.5, move left
			if rand < 0.5:
				delta1 = (x - lb) / (ub - lb)
				xy = 1. - delta1
				val = 2.*rand + (1. - 2.*rand) * xy**(eta + 1.)
				deltaQ = val**mutPow - 1.
			## If rand > 0.5, move right
			else:
				delta2 = (ub - x) / (ub - lb)
				xy = 1. - delta2
				val = 2.*(1. - rand) + 2.*(rand - 0.5) * xy**(eta + 1.)
				deltaQ = 1. - val**mutPow

			## Update with perturbation and check bounds
			x += deltaQ*(ub - lb)
			self[key] = min(max(x, lb), ub)

		return


	############################################################################
	#	Get Parameter Bounds
	############################################################################
	def _getParamBounds(self, verbose=0):
		"""self._getParamBounds(verbose=0): Quickly grabs the dictionary of
		parameters and their bounds by compiling the *full* info dict and
		returning infoDict['modelParams']
		"""

		verbose = utl.force_pos_int(verbose, name='verbose', zero_ok=True)

		infoDict = rfu.getInfo(self.infoDict['infoDir'], verbose=verbose)
		modelDict = self.load_model(verbose=verbose)
		infoDict = self._checkModelInfo(infoDict, modelDict)

		return infoDict['modelParams'].copy()



################################################################################
#	Run if __name__=="__main__"
################################################################################
if __name__ == "__main__":

	infoPath = "./Runfiles/HH_Test/"

	infoDict = rfu.getInfo(infoPath, verbose=0)

	ind = Individual(infoDict)

	print("\nShowing variable parameters:")
	for key in ind:
		if key[0] == 'g':		
			print(f"{key}: {ind[key]:.3g}")

	ind.mutate()

	print("\nShowing mutated parameters:")
	for key in ind:
		if key[0] == 'g':		
			print(f"{key}: {ind[key]:.3g}")








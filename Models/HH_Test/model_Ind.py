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

from Base.Individual import Individual as BaseInd

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
"""
================================================================================
	Evolutionary Algorithm Main Routine
================================================================================

	Author: Eric Johnson
	Date Created: Monday, November 20, 2017
	Date Revised: Wednesday, February 20, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	This file will contain the main evolutionary algorithm program for this 
	project.  The outline of the procedures here are as follows:

	1. Population is initialized
		- Individuals are evaluated (this is a distributed process)
	2. Population is duplicated, children are mutated and crossed
	3. Mutated individuals are evaluated (distributed)
	4. Fitnesses of individuals are compared and members of the Pareto front
	   are saved to the Hall of Fame (HoF)
	5. Repeat

================================================================================
================================================================================
"""
from copy import deepcopy
import datetime
import importlib.util as imputl
import logging
from mpi4py import MPI
import numpy as numpy
import os
import pickle as pkl
import socket
import sys
from time import sleep

# from Base.Individual import Individual
from Base.Population import Population
# from Base.halloffame import HallofFame
# from EAMPI.distributeEvaluation import distributeEvaluation
# from EAMPI.setupProcesses import setupProcesses
import Utility.runfile_util as rfu

#===============================================================================
#===============================================================================

# Get runtime and host name
time = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
host = str(socket.gethostname())

# Get infoDir
infoDir = str(sys.argv[-1]) # The infoDir should be provided as a command-line
							# input.
infoDict = rfu.getInfo(infoDir, verbose=1)
infoDict['time'] = deepcopy(time)
infoDict['host'] = deepcopy(host)

# indSpec = imputl.spec_from_file_location("Individual",
# 	os.path.join(infoDict['modelDir'], "model_Ind.py"))
# foo = imputl.module_from_spec(indSpec)
# indSpec.loader.exec_module(foo)
# Individual = foo.Individual

if __name__ == "__main__":

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	if rank == 0:
		
		pass







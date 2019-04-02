"""
================================================================================
	DIP TEST
================================================================================

	Author: Johannes Bauer
	Adapted: Eric Johnson
	Date Created: June 17, 2015
	Date Adapted: Friday, March 22, 2019
	Email: ericjohnson1.2015@u.northwestern.edu

================================================================================
================================================================================

	From Johannes Bauer's Github:

	Module for computing the Hartigans' dip statistic

	The dip statistic measures unimodality of a sample from a random process

	See:
	Hartigan, J. A.; Hartigan, P. M. The Dip Test of Unimodality.  The Annals
	of Statistics 13 (1985), no. 1, 70-84. doi:10.1214/aos/117634577.
	http://projecteuclid.org/euclid.aos/1176346577

	This code is also translated from R code that interfaces with the original
	C code that was written by Hartigan & Hartigan.

================================================================================
================================================================================
"""

import collections
from copy import deepcopy
import numpy as np



def getGCM(pts, idx):

	gcm = [pts[0]]
	tPts = [0]

	while len(pts) > 1:

		dists = idx[1:] - idx[0]
		slopes = (pts[1:] - pts[0])/dists

		minSlope = slopes.min()
		minIdx = np.where(slopes == minSlope)[0][0] + 1

		gcm.extend(pts[0] + dists[:minIdx]*minSlope)
		tPts.append(tPts[-1] + minIdx)

		pts = pts[minIdx:]
		idx = idx[minIdx:]

	return np.array(gcm), np.array(tPts).astype(int)


def getLCM(pts, idx):

	lcm, tPts = getGCM(pts.max() - pts[::-1], idx.max() - idx[::-1])

	return 1 - lcm[::-1], np.array(len(pts) - 1 - tPts[::-1]).astype(int)


def dip(hist=None, idx=None, returnAll=False):

	if idx is None:
		idx = np.arange(len(histogram))

	elif hist is None:
		h = collections.Counter(idx)
		idx = np.msort(list(h.keys()))
		hist = np.asarray([h[i] for i in idx])

	else:
		if len(hist) != len(idx):
			raise ValueError("Input arguments 'hist' and 'idx' must be "+
				"the same size! (Must have same numbers of indices and bins.)")

		if len(idx) != len(set(idx)):
			raise ValueError("All elements of 'idx' must be unique (histogram "+
				"cannot have multiple bins with identical values.)")

		if not np.all(np.msort(idx) == idx):
			idxIdx = np.argsort(idx)
			idx = np.array(idx)[idxIdx]
			hist = np.array(hist)[idxIdx]

	errStr = "Keyword argument 'returnAll' must be boolean."
	assert isinstance(returnAll, bool), errStr

	hist = hist/np.sum(hist)

	cdf = np.cumsum(hist)/np.sum(hist)

	tmpIdx, tmpPDF, tmpCDF = idx.copy(), hist.copy(), cdf.copy()

	lVals, rVals, D = [0], [1], 0.

	while True:

		lPart, lTps = getGCM((tmpCDF-tmpPDF).copy(), tmpIdx.copy())
		rPart, rTps = getLCM(tmpCDF.copy(), tmpIdx.copy())

		lDiffs = np.abs((lPart[lTps] - rPart[lTps]))
		rDiffs = np.abs((lPart[rTps] - rPart[rTps]))
		dl, dr = lDiffs.max(), rDiffs.max()

		if dr > dl:
			xr = rTps[dr == rDiffs][-1]
			xl = lTps[lTps <= xr][-1]
			d = dr
		else:
			xl = lTps[dl == lDiffs][0]
			xr = rTps[rTps >= xl][0]
			d = dl

		if (d <= D) or (xr == 0) or (xl == len(tmpCDF)):
			theDip = max(np.abs(cdf[:len(lVals)] - lVals).max(),
				np.abs(cdf[-len(rVals)-1:-1] - rVals).max())

			if returnAll:
				return theDip/2., (cdf, idx, lVals, lPart, rVals, rPart)
			else:
				return theDip/2.

		else:
			D = max(D, np.abs(lPart[:xl+1] - tmpCDF[:xl+1]).max(),
				np.abs(rPart[xr:] - tmpCDF[xr:] + tmpPDF[xr:]).max())

		tmpCDF = tmpCDF[xl:xr+1]
		tmpPDF = tmpPDF[xl:xr+1]
		tmpIdx = tmpIdx[xl:xr+1]

		lVals[len(lVals):] = lPart[1:xl+1]
		rVals[:0] = rPart[xr:-1]


def plot_dip(hist=None, idx=None):

	import matplotlib.pyplot as plt

	d, (cdf, idx, lVals, lPart, rVals, rPart) = dip(hist=hist, idx=idx,
		returnAll=True)
	print(d)

	plt.plot(idx[:len(lVals)], lVals, color='red')
	plt.plot(idx[len(lVals)-1:len(lVals)+len(lPart) - 1], lPart, color='gray')
	plt.plot(idx[-len(rVals):], rVals, color='blue')
	plt.plot(idx[len(cdf)-len(rVals)+1-len(rPart):len(cdf)-len(rVals)+1], 
		rPart, color='gray')

	plt.plot(np.repeat(idx,2)[1:], np.repeat(cdf,2)[:-1], color='black')
	plt.scatter(idx, cdf)

	the_dip = max(np.abs(cdf[:len(lVals)] - lVals).max(),
		np.abs(cdf[-len(rVals)-1:-1] - rVals).max())
	l_dip_idx = np.abs(cdf[:len(lVals)] - lVals) == the_dip
	r_dip_idx = np.abs(cdf[-len(rVals)-1:-1] - rVals) == the_dip

	print(the_dip)

	if np.any(l_dip_idx):
		plt.vlines(x=idx[:len(lVals)][l_dip_idx],
			ymin=cdf[:len(lVals)][l_dip_idx],
			ymax = cdf[:len(lVals)][l_dip_idx] - the_dip, color='g')
	if np.any(r_dip_idx):
		plt.vlines(x=idx[-len(rVals):][r_dip_idx],
			ymin=cdf[-len(rVals)-1:-1][r_dip_idx],
			ymax = cdf[-len(rVals)-1:-1][r_dip_idx] + the_dip, color='g')

	plt.show()


def dipTest(vals, simPVal=True, nSim=2000, verbose=0):

	vals = np.sort(np.array(vals).astype(float))

	assert isinstance(simPVal, bool)

	if simPVal:
		assert nSim == int(nSim)
		nSim = int(nSim)

	if verbose:
		print("\n\n" + 55*"=" + "\n" + "\tHartigans' Dip Test for Unimodality")
		print(55*"=")

	N = len(vals)
	D = dip(idx=vals)

	if N <= 3:
		P = 1.
	elif simPVal:
		if verbose > 1:
			print(f"\nCalculating p-value with {nSim} simulations.")

		P = np.mean(D <= np.array([dip(idx=np.random.rand(N))
			for b in range(nSim)]))
	
	else:
		raise ValueError("Lookup table has not been implemented.")


	if verbose > 1:
		print(f"\nD = {D:.4g}, P(d < D) = {P:.4g}")
		print("Alternative Hypothesis: Data are not unimodal.\n")

	return D, P
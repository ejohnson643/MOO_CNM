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

	lcm, tPts = getGCM(1 - pts[::-1], idx.max() - idx[::-1])

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


def dipPlot(hist=None, idx=None, ax=None, showGlobals=True, showLegend=True):

	import matplotlib.pyplot as plt
	import seaborn as sns

	sns.set(color_codes=True)

	if idx is None:
		idx = np.arange(len(histogram))

	elif hist is None:
		h = collections.Counter(idx)
		idx = np.msort(list(h.keys()))
		hist = np.asarray([h[i] for i in idx])

	else:
		if len(hist) != len(idx):
			raise ValueError("Input arguments 'hist' and 'idx' must be the"+
				" same size! (Must have same numbers of indices and bins.)")

		if len(idx) != len(set(idx)):
			raise ValueError("All elements of 'idx' must be unique (histo"+
				"gram cannot have multiple bins with identical values.)")

		if not np.all(np.msort(idx) == idx):
			idxIdx = np.argsort(idx)
			idx = np.array(idx)[idxIdx]
			hist = np.array(hist)[idxIdx]

	hist = hist/np.sum(hist)
	print(hist, len(hist), len(idx))

	D, (cdf, _, lVals, lPart, rVals, rPart) = dip(hist=hist, idx=idx,
		returnAll=True)

	N = len(cdf)

	if ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(12, 6))
		resize_at_end = True

	ax.plot(np.repeat(idx, 2)[1:], np.repeat(cdf, 2)[:-1], color='c',
		label='CDF')

	if showGlobals:
		bigGCM, bigLTps = getGCM(cdf - hist, idx.copy())
		bigLCM, bigRTps = getLCM(cdf.copy(), idx.copy())

		ax.plot(idx, bigGCM, 'r', alpha=0.5, lw=1, label='Global GCM')
		ax.scatter(idx[bigLTps], bigGCM[bigLTps], color='r', alpha=0.5)

		ax.plot(idx, bigLCM, 'b', alpha=0.5, lw=1, label='Global LCM')
		ax.scatter(idx[bigRTps], bigLCM[bigRTps], color='b', alpha=0.5)

	ax.plot(np.repeat(idx[:len(lVals)], 2)[1:], 
		np.repeat(cdf[:len(lVals)], 2)[:-1], color='gray')
	ax.plot(np.repeat(idx[-len(rVals):], 2)[1:], 
		np.repeat(cdf[-len(rVals):], 2)[:-1] , color='gray')

	xlim = ax.get_xlim()
	ax.plot([xlim[0], idx[0], idx[0]], [0, 0, cdf[0]], color='gray')
	ax.plot([idx[-1], idx[-1], xlim[1]], [cdf[-1], 1, 1], color='gray')

	ax.plot(idx[len(lVals)-1:-len(rVals)+1], lPart, color='r',
		label='GCM in [xL, xU]')
	ax.plot(idx[len(lVals)-1:-len(rVals)+1], rPart, color='b',
		label='LCM in [xL, xU]')

	ax.hlines(y=[0, 1], xmin=xlim[0], xmax=xlim[1], color='gray', linestyle=':')
	ax.set_xlim(xlim)

	ylim = ax.get_ylim()
	ax.vlines(x=[idx[len(lVals)-1], idx[-len(rVals)]], ymin=ylim[0], ymax=ylim[1],
		color='green', linestyle='--')
	ax.set_ylim(ylim)

	ax.text(idx[len(lVals)-1]+0.1, 0.02, r"$x_L$")
	ax.text(idx[-len(rVals)]+0.1, 0.02, r"$x_U$")

	l_dip_idx = np.abs(cdf[:len(lVals)]-lVals) == D*2
	r_dip_idx = np.abs(cdf[-len(rVals)-1:-1]-rVals) == D*2
	if np.any(l_dip_idx):
		ax.vlines(x=idx[:len(lVals)][l_dip_idx], ymin=cdf[:len(lVals)][l_dip_idx],
			ymax=cdf[:len(lVals)][l_dip_idx]-D*2, color='purple',
			linewidth=3, label=f'The Dip = {D:.4g}', zorder=3)
		ax.scatter(2*[idx[:len(lVals)][l_dip_idx]], [cdf[:len(lVals)][l_dip_idx],
			cdf[:len(lVals)][l_dip_idx]-D*2], color='purple', zorder=3)
	if np.any(r_dip_idx):
		ax.vlines(x=idx[-len(rVals):][r_dip_idx],
			ymax=cdf[-len(rVals)-1:-1][r_dip_idx]+D*2,
			ymin=cdf[-len(rVals)-1:-1][r_dip_idx],  color='purple',
			linewidth=3, label=f'The Dip = {D:.4g}', zorder=3)
		ax.scatter(2*[idx[-len(rVals):][r_dip_idx]],
			[cdf[-len(rVals)-1:-1][r_dip_idx],
			cdf[-len(rVals)-1:-1][r_dip_idx]+D*2], color='purple', zorder=3)

	ax.set_xlabel("Spike Height", fontsize=16)
	ax.set_ylabel(r"$P(h \leq H_{spike})$", fontsize=16)

	ax.legend(fontsize=12)

	if resize_at_end:
		fig.tight_layout()
		return fig, ax


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
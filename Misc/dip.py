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

================================================================================
================================================================================
"""

from copy import deepcopy
import numpy as np

def _gcm_(cdf, idxs):
	work_cdf = cdf
	work_idxs = idxs
	gcm = [work_cdf[0]]
	touchpoints = [0]
	while len(work_cdf) > 1:
		distances = work_idxs[1:] - work_idxs[0]
		slopes = (work_cdf[1:] - work_cdf[0]) / distances
		minslope = slopes.min()
		minslope_idx = np.where(slopes == minslope)[0][0] + 1
		gcm.extend(work_cdf[0] + distances[:minslope_idx] * minslope)
		touchpoints.append(touchpoints[-1] + minslope_idx)
		work_cdf = work_cdf[minslope_idx:]
		work_idxs = work_idxs[minslope_idx:]
	return np.array(np.array(gcm)),np.array(touchpoints)

def _lcm_(cdf, idxs):
	g,t = _gcm_(1-cdf[::-1], idxs.max() - idxs[::-1])
	return 1-g[::-1], len(cdf) - 1 - t[::-1]

def _touch_diffs_(part1, part2, touchpoints):
	diff = np.abs((part2[touchpoints] - part1[touchpoints]))
	return diff.max(), diff

def dip(histogram=None, idxs=None):
	"""
		Compute the Hartigans' dip statistic either for a histogram of
		samples (with equidistant bins) or for a set of samples.
	"""
	if idxs is None:
		idxs = np.arange(len(histogram))
	elif histogram is None:
		h = collections.Counter(idxs)
		idxs = np.msort(h.keys())
		histogram = np.array([h[i] for i in idxs])
	else:
		if len(histogram) != len(idxs):
			raise ValueError("Need exactly as many indices as histogram bins.")
		if len(idxs) != len(set(idxs)):
			raise ValueError("idxs must be unique if histogram is given.")
		if not np.array_equal(np.msort(idxs), idxs):
			idxs_s = np.argsort(idxs)
			idx = np.asarray(idxs)[idxs_s]
			histogram = np.asarray(histogram)[idxs_s]

	cdf = np.cumsum(histogram, dtype=float)
	cdf /= cdf[-1]

	work_idxs = idxs
	work_histogram = np.asarray(histogram, dtype=float) / np.sum(histogram)
	work_cdf = cdf

	D = 0
	left = [0]
	right = [1]

	while True:
		left_part, left_touchpoints   = _gcm_(work_cdf - work_histogram, work_idxs)
		right_part, right_touchpoints = _lcm_(work_cdf, work_idxs)

		d_left, left_diffs   = _touch_diffs_(left_part, right_part, left_touchpoints)
		d_right, right_diffs = _touch_diffs_(left_part, right_part, right_touchpoints)

		if d_right > d_left:
			xr = right_touchpoints[d_right == right_diffs][-1]
			xl = left_touchpoints[left_touchpoints <= xr][-1]
			d  = d_right
		else:
			xl = left_touchpoints[d_left == left_diffs][0]
			xr = right_touchpoints[right_touchpoints >= xl][0]
			d  = d_left

		left_diff  = np.abs(left_part[:xl+1] - work_cdf[:xl+1]).max()
		right_diff = np.abs(right_part[xr:]  - work_cdf[xr:] + work_histogram[xr:]).max()

		if d <= D or xr == 0 or xl == len(work_cdf):
			the_dip = max(np.abs(cdf[:len(left)] - left).max(), np.abs(cdf[-len(right)-1:-1] - right).max())
			return the_dip/2, (cdf, idxs, left, left_part, right, right_part)
		else:
			D = max(D, left_diff, right_diff)

		work_cdf = work_cdf[xl:xr+1]
		work_idxs = work_idxs[xl:xr+1]
		work_histogram = work_histogram[xl:xr+1]

		left[len(left):] = left_part[1:xl+1]
		right[:0] = right_part[xr:-1]

# def gcm(cdf, idx):

# 	tmp_cdf = deepcopy(cdf)
# 	tmp_idx = deepcopy(idx)

# 	gcm, tps = [tmp_cdf[0]], [0]

# 	while len(tmp_cdf) > 1:
		
# 		dists = tmp_idx[1:] - tmp_idx[0]
		
# 		slopes = (tmp_cdf[1:] - tmp_cdf[0])/dists
# 		minslope = slopes.min()
# 		minslope_idx = np.where(slopes == minslope)[0][0] + 1

# 		gcm.extend(tmp_cdf[0] + dists[:minslope_idx]*minslope)

# 		tps.append(tps[-1] + minslope_idx)

# 		tmp_cdf = tmp_cdf[minslope_idx:]
# 		tmp_idx = tmp_idx[minslope_idx:]

# 	return np.array(np.array(gcm)), np.array(tps)


# def lcm(cdf, idx):
# 	g, t = gcm(1 - cdf[::-1], idx.max() - idx[::-1])

# 	return 1 - g[::-1], len(cdf) - 1 - t[::-1]


# def touch_diffs(part1, part2, tps):

# 	diff = np.abs((part2[tps] - part1[tps]))

# 	return diff.max(), diff

# def dip(hist, idx=None):

# 	if idx is None:
# 		idx = np.arange(len(hist))

# 	tmp_hist = deepcopy(hist)
# 	tmp_idx = deepcopy(idx)

# 	cdf = np.cumsum(hist, dtype=float)
# 	cdf /= cdf[-1]

# 	tmp_cdf = deepcopy(cdf)

# 	D, left, right = 0, [0], [1]

# 	while True:

# 		left_part, left_tps = gcm(tmp_cdf - tmp_hist, tmp_idx)
# 		right_part, right_tps = lcm(tmp_cdf, tmp_idx)

# 		d_left, left_diffs = touch_diffs(left_part, right_part, left_tps)
# 		d_right, right_diffs = touch_diffs(left_part, right_part, right_tps)

# 		if d_right > d_left:
# 			xr = right_tps[d_right == right_diffs][-1]
# 			xl = left_tps[left_tps <= xr][-1]
# 			d = d_right
# 		else:
# 			xl = left_tps[d_left == left_diffs][0]
# 			xr = right_tps[right_tps >= xl][-1]
# 			d = d_left

# 		left_diff = np.abs(left_part[:xl+1] - tmp_cdf[:xl+1]).max()
# 		right_diff = np.abs(right_part[xr:] - tmp_cdf[xr:] + tmp_hist[xr:]).max()

# 		if (d <= D) or (xr == 0) or (xl == len(tmp_cdf)):
# 			the_dip = max(np.abs(cdf[:len(left)] - left).max(),
# 				np.abs(cdf[-len(right)-1:-1] - right).max())
# 			return the_dip/2, (cdf, idx, left, left_part, right, right_part)
# 		else:
# 			D = max(D, left_diff, right_diff)

# 		tmp_cdf = tmp_cdf[xl:xr+1]
# 		tmp_idx = tmp_idx[xl:xr+1]
# 		tmp_hist = tmp_hist[xl:xr+1]

# 		left[len(left):] = left_part[1:xl+1]
# 		right[:0] = right_part[xr:-1]


def plot_dip(hist, idx=None):

	import matplotlib.pyplot as plt

	d, (cdf, idx, left, left_part, right, right_part) = dip(hist, idx)

	plt.plot(idx[:len(left)], left, color='red')
	plt.plot(idx[len(left)-1:len(left)+len(left_part) - 1],
		left_part, color='gray')
	plt.plot(idx[-len(right):], right, color='blue')
	plt.plot(idx[len(cdf)-len(right)+1-len(right_part):len(cdf)-len(right)+1], 
		right_part, color='gray')

	the_dip = max(np.abs(cdf[:len(left)] - left).max(),
		np.abs(cdf[-len(right)-1:-1] - right).max())
	l_dip_idx = np.abs(cdf[:len(left)] - left) == the_dip
	r_dip_idx = np.abs(cdf[-len(right)-1:-1] - right) == the_dip
	print(the_dip/2, d)

	plt.vlines(x=idx[:len(left)][l_dip_idx],
		ymin=cdf[:len(left)][l_dip_idx],
		ymax = cdf[:len(left)][l_dip_idx] - the_dip)
	plt.vlines(x=idx[-len(right):][r_dip_idx],
		ymin=cdf[-len(right)-1:-1][r_dip_idx],
		ymax = cdf[-len(right)-1:][r_dip_idx] + the_dip)

	plt.plot(np.repeat(idx,2)[1:], np.repeat(cdf,2)[:-1], color='black')
	plt.scatter(idx, cdf)

	plt.show()
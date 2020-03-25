
import matplotlib.pyplot as plt
import seaborn as sns

plt.close('all')
sns.set(color_codes=True)

import Utility.DataIO_util as DIO

import Objectives.Electrophysiology.ephys_objs_v2 as epo

fileName = "12/06/2012"
dataNum = 8

data = DIO.loadABF(fileName, dirName = "./Data/TaeHeeData/", dirType='Han',
	dataNum=dataNum)
dt = 1/data.dataRate

## These are spike locations and heights
spikeIdx, spikeVals = epo.getSpikeIdx(data, dt=dt)

## Get ISI and FR
ISI = epo.getISI(spikeIdx, dt=dt)
FR = epo.getFR(ISI)

## Get fit to spike amplitude
amp = epo.getSpikeAmp(spikeIdx, spikeVals, dt=dt)

## Get inter-spike voltage
ISV = epo.getInterSpikeVoltage(data, dt=dt)

for sNo in range(data.sweepCount):
	print("Setting sweep to ", sNo)
	data.setSweep(sNo)

	## If I only want depol step
	idx1, idx2 = data.sweepEpochs.p1s[2], data.sweepEpochs.p2s[2]
	depolData = data.sweepY[idx1:idx2]

	dplSpikes, dplSVals = epo.getSpikeIdx(depolData, dt=dt, verbose=-1)

	dplFR = epo.getFR(epo.getISI(dplSpikes, dt=dt))
	print("Firing rate is", dplFR)

	dplX = data.sweepX[idx1:idx2]

	plt.plot(dplX, depolData)




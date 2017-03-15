import random
import numpy as np
import matplotlib.pyplot as plt
import json


def testGraphOsc():
	vals = random.sample(xrange(1,200), 100)

	vals = sorted(vals)

	v = []

	for i in range(len(vals)):
		count = 0
		for j in range(i, len(vals)):
			if vals[j] > 25:
				count += 1
		v.append(count)

	print(v)

	fig = plt.figure()
	ax = fig.add_subplot(211)

	x = np.linspace(0, len(vals), len(vals))
	ax.scatter(x, vals)

	ax = fig.add_subplot(212)

	ax.scatter(x, v)
	plt.show()

def testHistogram(data, binsize):
	assert binsize < len(data) + 1, "Binsize cannot be greater than the size of the data array"

	data_hist = []
	for i in range(0, len(data)):
		num = 0
		if i - binsize < 0:
			num = sum(data[0:i+binsize+1])
		elif i + binsize > len(data):
			num = sum(data[i - binsize: len(data)])
		else:
			num = sum(data[i-binsize : i+ binsize + 1]) 

		data_hist.append(num)
	return data_hist

def graphHist(data, binsize):
	dataHist = testHistogram(data, binsize)
	fig = plt.figure()
	x = np.linspace(0, len(dataHist), len(dataHist))
	ax = fig.add_subplot(211)
	ax.plot(x, dataHist)

	ax = fig.add_subplot(212)
	ax.plot(x, data)
	plt.show()

def graphLFPs(lfp, elfp, ilfp):
	with open(lfp) as total:
		total_lfp = json.load(total)
	with open(elfp) as e:
		exc_lfp = json.load(e)
	with open(ilfp) as i:
		inh_lfp = json.load(i)
#	print(len(total_lfp), len(exc_lfp), len(inh_lfp))
	t = np.linspace(0, len(total_lfp), len(total_lfp))
	#print(len(t))
	fig = plt.figure()
	fig.suptitle('Total, Excitatory and Inhibitory LFPs', fontsize = 18)
	plt.plot(t, total_lfp, label = 'Total LFP')
	plt.plot(t, exc_lfp, label = 'Excitatory LFP')
	plt.plot(t, inh_lfp, label = 'Inhibitory LFP')
	plt.legend(loc = 'best')
	plt.xlabel('Time (ms)')
	plt.show()

def raster():
	 #set the random seed
	np.random.seed(0)

	# create random data
	data1 = np.random.random([6, 50])

	# set different colors for each set of positions
	colors1 = np.array([[1, 0, 0],
	                    [0, 1, 0],
	                    [0, 0, 1],
	                    [1, 1, 0],
	                    [1, 0, 1],
	                    [0, 1, 1]])

	# set different line properties for each set of positions
	# note that some overlap
	lineoffsets1 = np.array([-15, -3, 1, 1.5, 6, 10])
	linelengths1 = [5, 2, 1, 1, 3, 1.5]

	fig = plt.figure()

	# create a horizontal plot
	ax1 = fig.add_subplot(221)
	ax1.eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
	              linelengths=linelengths1)
	ax1.set_title('horizontal eventplot 1')


	# create a vertical plot
	ax2 = fig.add_subplot(223)
	ax2.eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
	              linelengths=linelengths1, orientation='vertical')
	ax2.set_title('vertical eventplot 1')

	# create another set of random data.
	# the gamma distribution is only used fo aesthetic purposes
	data2 = np.random.gamma(4, size=[60, 50])

	# use individual values for the parameters this time
	# these values will be used for all data sets (except lineoffsets2, which
	# sets the increment between each data set in this usage)
	colors2 = [[0, 0, 0]]
	lineoffsets2 = 1
	linelengths2 = 1

	# create a horizontal plot
	ax1 = fig.add_subplot(222)
	ax1.eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
	              linelengths=linelengths2)
	ax1.set_title('horizontal eventplot 2')


	# create a vertical plot
	ax2 = fig.add_subplot(224)
	ax2.eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
	              linelengths=linelengths2, orientation='vertical')
	ax2.set_title('vertical eventplot 2')

	plt.show()


#raster()
# test = random.sample(xrange(0,5000), 5000)
# graphHist(test, 10)

#graphLFPs('Run Data/LFP.json', 'Run Data/exclfp.json', 'Run Data/inhlfp.json')




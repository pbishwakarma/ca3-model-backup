#
#	plot LFP of the excitatory and inhibitory neurons, to look at phase relations
#	decrease frequency of inhibitory neurons firing
#	how to incorporate AMPA/GABA receptors into model
#

import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, rfft, fftfreq
import json
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import correlate
from scipy.signal import argrelextrema
import random
from testGraphOsc import graphLFPs


Exc_info = open("Exc.txt", 'r')
Inh_info = open("Inh.txt", 'r')
total_time = 0.0
time_step = 0.0
pyramidal_count = 200
interneuron_count = 50

def jsonToList(json_file):
	"""
	Extracts and returns a list containing the information from a
	json file.

	Args
	----------------------------------------------------------------------
	json_file (string) -- a string containing the name of the json file
		to be extracted

	Returns
	----------------------------------------------------------------------
	output (list) -- a list containing the information from the json file
	"""
	with open(json_file) as myFile:
		output = json.load(myFile)

	return output

def getMeanFiringRate(firingrates):
	# Print average firing rate information
	print("Excitatory firing rate: %f spikes/s" % (sum(firingrates[0])/pyramidal_count))
	print("Inhibitory firing rate: %f spikes/s" %(sum(firingrates[1])/interneuron_count))

def getData():
	"""
	Uses the function jsonToList() to extract all of the data from
	a simulation. Returns all of the data as lists in alphabetical
	order.

	Returns:
	----------------------------------------------------------------------
	data (list) -- Contains the following lists:
		LFP (list) -- a list of floats. Contains the average field potential
			of all the neurons at each timestep of the simulation

		exc_nrn (list) -- a list of lists. Contains lists with data from a
			single excitatory neuron over the entire simulation

		exclfp (list) -- a list of floats. Contains the average field
			potential of all excitatory neurons at each timestep of the
			simulation

		excspikes (list) -- a list of integers. Contains the number of 
			spikes from excitatory neurons at each timestep of the
			simulation

		inh_nrn (list) -- a list of lists. Contains lists with data from a 
			single inhibitory neuron over the entire simulation

		inhlfp (list) -- a list of floats. Contains the average field
			potential of all excitatory neurons at each timestep fo the
			simulation

		inhspikes (list) -- a list of integers. Contains the number of spikes
			from inhibitory neurons at each timestep of the simulation

		spikes (list) -- a list of integers. Contains the number of spikes
			from all neurons at each timestep of the simulation
	"""

	data = []
	print(sorted(glob.glob('Run Data/*.json')))

	for item in sorted(glob.glob('Run Data/*.json')):
		data.append(jsonToList(item))

	return data
	


def LFP(lfp_file):
	#extract LFP information from file
	LFP = open('LFPOutput.txt', 'r')
	field_potential = []
	line = LFP.readline().split()

	#print and store total running time of model
	global total_time
	total_time = int(line[-1])
	print("%s %s %s" % (line[0], line[1], line[2]))

	#print and store time step
	line = LFP.readline().rsplit()
	global time_step
	time_step = float(line[-1])
	print("%s %s %s" % (line[0], line[1], line[2]))

	#print information about connection probabilities
	for i in range(4):
		line = LFP.readline().split()
		print("%s %s %s" % (line[0], line[1], line[2]))

	#print information about synaptic dynamics
	for j in range(5):
		line = LFP.readline().rstrip().split("=")
		print("%s = %s" % (line[0], line[-1]))

	print("")
	print("")
	# extract LFP information
	with open(lfp_file) as lfp_info:
		field_potential = json.load(lfp_info)

	# exclude first 100ms of simulation
	trunc_LFP = field_potential[int(100/time_step):]
	N = len(trunc_LFP)
	sample_rate = time_step/1000

	# convolve LFP data with gaussian
	x = np.linspace(100.0, total_time, N)
	y = trunc_LFP
	#y = gaussian_filter(trunc_LFP, sigma = 15)

	fig = plt.figure()
	fig.suptitle('Local Field Potential and Corresponding Power Spectrum', fontsize=14, fontweight='bold')

	#graph LFP
	ax = fig.add_subplot(211)
	ax.set_ylabel('Local Field Potential')
	ax.set_xlabel('Time (ms)')
	ax.plot(x,y)
	plt.xticks(np.arange(100.0, total_time, 10))
	plt.xlim(100,total_time)


	# graph power spectrum
	xf = abs(fftfreq(N, sample_rate))
	yf = np.abs(fft(trunc_LFP))**2

	ax = fig.add_subplot(212)
	ax.set_ylabel('Power')
	ax.set_xlabel('Frequency (Hz)')
	ax.plot(xf, yf)
	plt.xticks(np.arange(0, total_time, 10.0))
	plt.xlim(5,500)
	#plt.ylim(0,10000)
	plt.savefig('Run Figures/lfp.png', dpi = 100)

	LFP.close()
	field_potential = field_potential - np.mean(field_potential)
	return field_potential, total_time, time_step

def graphLFP(excLFP, inhLFP):

	time = np.linspace(0,total_time, len(excLFP))
	exc_LFP = np.asarray(excLFP)
	#exc_LFP = gaussian_filter(exc_LFP, sigma = 15)
	inh_LFP = np.asarray(inhLFP)
	#inh_LFP = gaussian_filter(inh_LFP, sigma = 15)

	fig = plt.figure()
	fig.suptitle('Excitatory and Inhibitory Field Potentials', fontsize=14, fontweight='bold')

	ax = fig.add_subplot(211)
	ax.set_ylabel('Excitatory Field Potential')
	ax.plot(time, exc_LFP)
	plt.xlim(0,total_time)

	ax = fig.add_subplot(212)
	ax.set_ylabel('Inhibitory Field Potential')
	ax.set_xlabel('Time (ms)')
	ax.plot(time, inh_LFP)
	plt.xlim(0,total_time)
	plt.savefig('Run Figures/ExcInhLFP.png', dpi = 100)

	plt.show()


def graphNeuron(nrn_type, mempot, inh, exc, background, nmda, gate):
	assert int(nrn_type) == 0 or int(nrn_type) == 1, "Must be either inh or exc"

	fig = plt.figure()
	if nrn_type == 0:
		fig.suptitle('Inhibitory Neuron', fontsize=14, fontweight='bold')
	elif nrn_type == 1:
		fig.suptitle('Excitatory Neuron', fontsize=14, fontweight='bold')


	
	ax = fig.add_subplot(611)
	ax.set_ylabel('Membrane Potential')
	mem_pot = np.asarray(mempot)
	time = np.linspace(0.0, total_time, len(mem_pot))
	ax.plot(time, mem_pot)
	plt.xlim(0,total_time)

	ax = fig.add_subplot(612)
	ax.set_ylabel('Inhibition')
	inh_input = np.asarray(inh)
	ax.plot(time, inh_input)
	plt.xlim(0,total_time)

	ax = fig.add_subplot(613)
	ax.set_ylabel('Excitation')
	exc_input = np.asarray(exc)
	ax.plot(time, exc_input)
	plt.xlim(0,total_time)

	ax = fig.add_subplot(614)
	ax.set_ylabel('Background Current')
	back_current = np.array(background)
	ax.plot(time, back_current)
	plt.xlim(0,total_time)

	ax = fig.add_subplot(615)
	ax.set_ylabel('NMDA Current')
	nmda_current = np.array(nmda)
	ax.plot(time, nmda)
	plt.xlim(0, total_time)

	ax = fig.add_subplot(616)
	ax.set_ylabel('Mg gate sigmoid')
	gata_data = np.array(gate)
	ax.set_xlabel('Time (ms)')
	ax.plot(time, gate)
	if nrn_type == 0:
		plt.savefig('Run Figures/PN.png', dpi = 100)
	elif nrn_type == 1:
		plt.savefig('Run Figures/IN.png', dpi = 100)
	plt.show()

def spikesOutput(spikeFile, peaks, binsize):
	with open(spikeFile) as spk:
		spikes = json.load(spk)

	excSpikes = spikes[0]
	inhSpikes = spikes[1]
	total_spikes = [sum(x) for x in zip(excSpikes, inhSpikes)]

	spike_sum = 0
	bin_sums = []
	spike_means = []
	#exc_sub_spikes, inh_sub_spikes, total_sub_spikes = [],[],[]
	print(peaks)

	for ndx in peaks:
		spike_time_sum = 0
		nrn_spikes = 0
		nrn_spike_times = []
		exc_spike_sum, inh_spike_sum, total_spike_sum = 0,0,0
		if ndx - binsize//2 > 0 and ndx + binsize//2 < len(total_spikes):
			exc_sub_spikes = excSpikes[int(ndx - (binsize//2)):int(ndx + (binsize//2))]
			inh_sub_spikes = inhSpikes[ndx - (int(binsize//2)):int(ndx + (binsize//2))]
			total_sub_spikes = total_spikes[int(ndx - (binsize//2)):int(ndx + (binsize//2))]

			exc_spike_sum += sum(exc_sub_spikes)
			inh_spike_sum += sum(inh_sub_spikes)
			total_spike_sum += sum(exc_sub_spikes) + sum(inh_sub_spikes)

			bin_sums.append((exc_spike_sum, inh_spike_sum, total_spike_sum))

			for i in range(int(ndx-binsize//2), int(ndx+binsize//2)):
				spike_time_sum += total_spikes[i] * i
				nrn_spikes += total_spikes[i]
			mean_spikes = spike_time_sum / nrn_spikes
			spike_means.append(mean_spikes)


	#print(bin_sums)
	#print(len(bin_sums))
	#spike_means = [x/10 for x in spike_means]
	print(spike_means)

def getCellInfo(data):

	cell_mempot = data[0]
	cell_inh = data[1]
	cell_exc = data[2]
	cell_nmda = data[4]
	cell_background = data[3]
	cell_gate = data[5]

	return cell_mempot, cell_inh, cell_exc, cell_background, cell_nmda, cell_gate

def differentiate(func):

	dfunc = []
	for i in range(0, len(func)):
		if i == 0:
			dfunc.append(func[1] - func[0])
		elif i == len(func) - 1:
			dfunc.append(func[-1] - func[-2])
		else:
			dfunc.append((func[i+1] - func[i-1])/2)
	return dfunc

def findExtrema(func):
	dfunc = differentiate(func)
	peaks, troughs = [], []

	for i in range(1, len(dfunc)-1):
		if dfunc[i-1] > 0 and dfunc[i+1] < 0:
			peaks.append(i)
		elif dfunc[i-1] < 0 and dfunc[i+1] > 0:
			troughs.append(i)

	return peaks[0:-1:2], troughs[0:-1:2]


def markExtrema(func):
	marked = []
	peaks, troughs = findExtrema(func)

	for i in range(len(func)):
		if i in peaks or i in troughs:
			marked.append(func[i])
		else:
			marked.append(0)

	x = np.linspace(0, len(func), len(func))
	plt.plot(x, func)
	plt.scatter(x, marked)
	plt.xlim(0, len(func))
	plt.show()


def getOscillationPds(func, graph = False, timestep = 0.1):
	"""
	Determines the oscillation periods of a discrete function
	represented by a list of values. Checks for the values where
	the function crosses 0 from the negative side. Represents each
	oscillation pd as a tuple of length = 2, denoting the beginning
	and end of the oscillation period.

	Args
	----------------------------------------------------------------------
	avg_FP (list) -- A list containing the average field potential of the 
		neurons over the simulation.

	Returns
	----------------------------------------------------------------------
	osc_pds (list) -- A list containing tuples of size n = 2. The tuples
		contain the beginning and end points of an oscillation period.

	"""

	func = gaussian_filter(func, sigma = 7)
	peaks, troughs = findExtrema(func)
	#markExtrema(func)
	peak_pds, trough_pds = [], []
	peak_lengths, trough_lengths = [], []

	for i in range(0, len(peaks) - 1):
		time = int((peaks[i+1] - peaks[i])/(1/timestep))
		if time != 0:
			peak_pds.append((peaks[i] , peaks[i+1]))
			peak_lengths.append(time)

	for j in range(0, len(troughs) - 1):
		time = int((troughs[j+1] - troughs[j])/(1/timestep))
		if time != 0:
			trough_pds.append((troughs[j], troughs[j+1]))
			trough_lengths.append(time)

	if graph:
		return peak_lengths, trough_lengths
	else:
		return peak_pds, trough_pds


def excTimeToThresh(pds, field_pot, spikes, timestep = 0.1):
	"""
	Calculates the mean time it takes for the excitatory cells to reach
	threshold after the end of inhibition in an oscillation period.
	If no excitatory cells reach threshold in the oscillation period 
	after the end of excitation, that oscillation period is not included
	in the calculaiton.

	Args
	----------------------------------------------------------------------
	osc_period (list) -- a list of tuples of size n = 2. The tuples
		contain the beginning and end indices of oscillation periods

	field_pot (list) -- a list of floats. Contains the average membrane 
		potential of the excitatory cells at each timestep of the 
		simulation

	spikes (list) -- a list of the number of excitatory cell spikes at
		each timestep of the simulation

	timestep (float) -- the timestep used in the simulation. Defaults
		to 0.1ms

	Returns
	----------------------------------------------------------------------
	mean_time (float) -- the mean of the timescales to threshold for
		the oscillation periods.

	std_time (float) -- the standard deviation of the timescales to
		threshold for the oscillation periods.
	"""


	timescales = []
	# Iterate through oscillation periods.
	for period in pds:
		pd_beg, pd_end = period[0], period[1]
		i = pd_beg + 1
		# Check all values of the excitatory field potential whithin 
		# the oscillation period and check if the field potential 
		# changes signs from (-) to (+). Store value as end of inhibition.
		while i < pd_end:
			if field_pot[i-1] < 0 and field_pot[i+1] > 0:
				j = i
				# Check the spikes starting from the end of inhibition.
				# Check for non-zero spike number.
				# Difference between timestep and end of inhibition is
				# timescale for exc cells to reach threshold.
				while j < pd_end:
					if spikes[j] > 0:
						time = (j - i) / (1 / timestep)
						timescales.append(time)
						break
					else:
						j += 1
				break
			else:
				i += 1

	# Calculate mean and std dev of the timecales for each period
	mean_time = np.mean(timescales)
	std_time = np.std(timescales)

	print('Mean time to threshold for excitatory cells: %4.2fms' % (mean_time))
	print('Standard deviation of time to threshold for excitatory cells: % 4.2fms \n' % (std_time))
	#return mean_time, std_time


def inhTimeToThresh(pds, exc_field_pot, exc_spikes, inh_spikes, timestep = 0.1):
	"""
	Calculates the mean time it takes for the inhibitory cells to reach
	threshold. Calculated as the differnece in time between the first
	excitatory cell spike and the first inhibitory cell spike to follow.
	If no inhibitory cell spikes following the first excitatory cell
	spike in the oscillation period, then that oscillation period is not
	included in the calculations

	Args
	----------------------------------------------------------------------
	osc_period (list) -- a list of tuples of size n = 2. The tuples
		contain the beginning and end indices of oscillation periods

	exc_field_pot (list) -- a list of floats. Contains the average membrane 
		potential of the excitatory cells at each timestep of the 
		simulation

	exc_spikes (list) -- a list of integers. Contains the number of 
		excitatory cell spikes at each timestep of simulation.

	inh_spikes (list) -- a list of integers. Contains the number of
		inhbitory cell spikes at each timestep of simulation.

	timestep (float) -- the timestep used in the simulation. Defaults
		to 0.1ms

	Returns
	----------------------------------------------------------------------
	mean_time (float) -- the mean of the timescales to threshold for
		all oscillation periods.

	std_time (float) -- the standard deviation of the timescales to
		threshold for all oscillation periods.
	"""

	timescales = []
	
	# Iterate through oscillation periods. 
	for period in pds:
		pd_beg, pd_end = period[0], period[1]
		i = pd_beg

		# Check for change of signs in excitatory field potential that
		# represents the end of inhibition.
		while i < pd_end:
			if exc_field_pot[i-1] < 0 and exc_field_pot[i+1] > 0:
				j = i

				# Check for first excitatory cell spike after end of
				# inhibition.
				while j < pd_end:
					if exc_spikes[j] > 0:
						k = j

						# Check for first inhibitory cell spike after
						# first pyramidal cell spike
						while k < pd_end:
							if inh_spikes[k] > 0:
								time = (k - j) / (1/timestep)
								timescales.append(time)
								break
							else:
								k += 1
						break
					else:
						j += 1
				break
			else:
				i += 1

	# Calculate mean and std dev of the timecales for each period
	mean_time = np.mean(timescales)
	std_time = np.std(timescales)

	print('Mean time to threshold for inhibitory cells: %4.2fms' % (mean_time))
	print('Standard deviation of time to threshold for inhibitory cells: % 4.2fms \n' % (std_time))
	#return mean_time, std_time


def excTimeActive(pds, exc_lfp, exc_spikes, timestep = 0.10):
	"""
	Calculates the amount of time that excitatory cells are active
	by finding the time difference between the first excitatory cell
	spike and the last excitatory cell spike in the oscillation period.

	Args
	----------------------------------------------------------------------
	osc_periods (list) -- a list of tuples. Contains tuples of size n = 2,
		that contain the beginning and end of the oscillation period.

	exc_lfp (list) -- a list of floats. Contains the average field
		potential of all ecitatory neurons at each timestep of the
		simulation.

	exc_spikes (list) -- a list of ints. Contains the number of spikes
		from excitatory neurons at each timestep of the simulation.

	timestep (float) -- the timestep of the simulation Defaults to 0.10ms
	"""

	time_active = []
	for period in pds:

		pd_beg, pd_end = period[0], period[1]

		active, inactive = 0,0

		for i in range(pd_beg, pd_end):
			if exc_spikes[i] != 0:
				active = i
				break
		for j in range(pd_end, pd_beg, -1):
			if exc_spikes[j] != 0:
				inactive = j
				break

		time = (inactive - active) / ( 1 / timestep)
		time_active.append(time)


	mean_time = np.mean(time_active)
	std_time = np.std(time_active)

	print('Mean time active for excitatory cells: %4.2fms' % (mean_time))
	print('Standard deviation of time active for excitatory cells: % 4.2fms \n' % (std_time))


def inhTimeActive(pds, inh_lfp, inh_spikes, timestep = 0.10):
	"""
	Calculates the amount of time that excitatory cells are active
	by finding the time difference between the first excitatory cell
	spike and the last excitatory cell spike in the oscillation period.

	Args
	----------------------------------------------------------------------
	osc_periods (list) -- a list of tuples. Contains tuples of size n = 2,
		that contain the beginning and end of the oscillation period.

	exc_lfp (list) -- a list of floats. Contains the average field
		potential of all ecitatory neurons at each timestep of the
		simulation.

	exc_spikes (list) -- a list of ints. Contains the number of spikes
		from excitatory neurons at each timestep of the simulation.

	timestep (float) -- the timestep of the simulation Defaults to 0.10ms
	"""

	time_active = []
	for period in pds:

		pd_beg, pd_end = period[0], period[1]

		active, inactive = 0,0 

		for i in range(pd_beg, pd_end + 1):
			if inh_spikes[i] != 0:
				active = i
				break

		for j in range(pd_end, pd_beg - 1, -1):
			if inh_spikes[j] != 0:
				inactive = j
				break

		time = (inactive - active) / (1 / timestep)
		time_active.append(time)


	mean_time = np.mean(time_active)
	std_time = np.std(time_active)

	print('Mean time active for inhibitory cells: %4.2fms' % (mean_time))
	print('Standard deviation of time active for inhibitory cells: % 4.2fms \n' % (std_time))

def timescales(lfp, exclfp, inhlfp, excspikes, inhspikes):
	peak_pds, trough_pds = getOscillationPds(lfp)
	excTimeToThresh(peak_pds, exclfp, excspikes)
	inhTimeToThresh(peak_pds, exclfp, excspikes, inhspikes)
	excTimeActive(peak_pds, exclfp, excspikes)
	inhTimeActive(peak_pds, inhlfp, inhspikes)

def graphOscPds(lfp):
	oscPdDist(lfp)
	oscPdHist(lfp)

def oscPdDist(lfp):
	peaks, troughs = getOscillationPds(lfp, True)

	peaks = [1000.0/i for i in peaks]
	troughs = [1000.0/i for i in troughs]

	x = np.linspace(0, 500, 100)
	peaks_dist, troughs_dist = [], []
	for i in range(len(x)):
		peaks_dist.append(float(sum(peaks[j] > x[i] for j in range(len(peaks))))/len(peaks))
	for i in range(len(x)):
		troughs_dist.append(float(sum(troughs[j] > x[i] for j in range(len(troughs))))/len(troughs))


	fig = plt.figure()
	fig.suptitle('Distribution of oscillation period frequency')
	ax = fig.add_subplot(211)
	ax.set_ylabel('Proportion of peak periods frequencies greater than f')
	ax.plot(x, peaks_dist)

	ax = fig.add_subplot(212)
	ax.set_ylabel('Proportion of trough period frequencies greater than f')
	ax.plot(x, troughs_dist)
	ax.set_xlabel('Frequency f(Hz)')
	plt.savefig('Run Figures/OscPdDist.png', dpi = 100)
	plt.show()
	
def oscPdHist(lfp):
	peaks, troughs = getOscillationPds(lfp, True)
	peaks = [1000.0/i for i in peaks]
	troughs = [1000.0/i for i in troughs]
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.set_title('Oscillation Periods Based on Peaks')
	p_bins = np.arange(0, max(peaks), 10)
	ax.hist(peaks, p_bins)
	ax.set_xticks(p_bins)
	ax.set_ylabel('Count')

	ax = fig.add_subplot(212)
	ax.set_title('Oscillation Periods Based on Troughs')
	t_bins = np.arange(0, max(troughs), 10)
	ax.hist(troughs, t_bins)
	ax.set_xticks(t_bins)
	ax.set_ylabel('Count')
	ax.set_xlabel('Frequency (Hz)')
	plt.plot()
	plt.savefig('Run Figures/OscPdHist.png', dpi = 100)
	plt.show()

def getFreqDist(data, binsize):
	assert binsize <= len(data), 'Binsize cannot exceed length of data array'
	freqDist = []
	for i in range(0, len(data)):
		num = 0
		if i - binsize < 0:
			num = sum(data[0:i+binsize+1])
		elif i + binsize > len(data):
			num = sum(data[i - binsize: len(data)])
		else:
			num = sum(data[i-binsize : i+ binsize + 1]) 

		freqDist.append(num)
	return freqDist

def graphFreqDist(excdata, inhdata, binsize):
	excFreqDist = getFreqDist(excdata, binsize)
	inhFreqDist = getFreqDist(inhdata, binsize)

	x_exc = np.linspace(0, len(excdata), len(excdata))
	x_inh = np.linspace(0, len(inhdata), len(inhdata))

	fig = plt.figure()
	fig.suptitle('Frequency Distribution of Excitatory(top) and Inhibitory(bottom) Cell Spikes')
	ax = fig.add_subplot(211)
	ax.plot(x_exc, excFreqDist)
	ax.set_ylabel('Frequency of excitatory spikes')

	ax = fig.add_subplot(212)
	ax.plot(x_inh, inhFreqDist)
	ax.set_ylabel('Frequency of inhibitory spikes')
	ax.set_xlabel('Time (ms)')

	plt.show()
	plt.savefig('Run Figures/freqdist.png', dpi = 100)

def phaseDiff(total_LFP, exclfp, inhlfp, tstep, totaltime):
	peakpd, troughpd = getOscillationPds(total_LFP)
	period = np.mean(peakpd)
	nsamples = totaltime / tstep - int(100/tstep)

	#print(len(total_LFP))
	#print(len(exclfp))
	#print(len(inhlfp))

	t = np.linspace(0, totaltime - int(100/tstep), nsamples, endpoint = False)
	#print(len(t))
	exc_corr = correlate(total_LFP, exclfp)
	inh_corr = correlate(total_LFP, inhlfp)
	#dt = np.arange(1-nsamples, nsamples)
	dt = np.linspace(-t[-1], t[-1], 2*nsamples - 1)
	#print(len(dt))
	inh_timeshift = ((dt[inh_corr.argmax()] *tstep % period) / period )* 2 * np.pi
	exc_timeshift = ((dt[exc_corr.argmax()] * tstep % period) / period )* 2 * np.pi
	#inh_timeshift = (dt[inh_corr.argmax()])
	#exc_timeshift = (dt[exc_corr.argmax()])

	print("Timeshift for exc and total: %f" % exc_timeshift)
	print("Timeshift for inh and total: %f" % inh_timeshift)
	print("")

def raster(excnrns, color = 'k'):
	ax = plt.gca()

	for ith, trial in enumerate(excnrns):
		plt.vlines(trial, ith + 0.5, ith + 1.5, color = color)
	plt.ylim(0.5, len(excnrns) + 0.5)
	return ax

def main():

	total_LFP, totaltime, tstep = LFP('Run Data/LFP.json')

	#data = getData()

	lfp = jsonToList('Run Data\LFP.json')
	exclfp = jsonToList('Run Data\exclfp.json')
	excnrns = jsonToList('Run Data\excnrns.json')
	excspikes = jsonToList('Run Data\excspikes.json')
	inhlfp = jsonToList('Run Data\inhlfp.json')
	inhnrns = jsonToList('Run Data\inhnrns.json')
	inhspikes = jsonToList('Run Data\inhnrns.json')
	spikes = jsonToList('Run Data\spikes.json')
	firingrates = jsonToList("Run Data/firingrates.json")

	# exclfp = data[1]
	# excnrns = data[2]
	# excspikes = data[3]
	# inhlfp = data[4]
	# inhnrns = data[5]
	# inhspikes = data[6]
	# spikes = data[7]


	#total_LFP, exclfp, inhlfp = total_LFP[int(100/tstep):], exclfp[int(100/tstep):], inhlfp[int(100/tstep):]
	total_LFP, exclfp, inhlfp = lfp[int(100/tstep):], exclfp[int(100/tstep):], inhlfp[int(100/tstep):]
	excspikes, inhspikes = excspikes[int(100/tstep):], inhspikes[int(100/tstep):]
	getMeanFiringRate(firingrates)
	graphLFP(exclfp, inhlfp)

	#
	#	calculate phase differences
	#

	# period = 25
	# nsamples = totaltime / tstep - int(100/tstep)

	# exc_corr = correlate(total_LFP, exclfp)
	# inh_corr = correlate(total_LFP, inhlfp)
	# dt = np.arange(1-nsamples, nsamples)
	# inh_timeshift = (dt[inh_corr.argmax()] * tstep % period) / period * 2 * np.pi
	# exc_timeshift = (dt[exc_corr.argmax()] * tstep % period) / period * 2 * np.pi

	# print("Timeshift for exc and total: %f" % exc_timeshift)
	# print("Timeshift for inh and total: %f" % inh_timeshift)
	# print("")
	graphLFPs('Run Data/LFP.json', 'Run Data/exclfp.json', 'Run Data/inhlfp.json')
	phaseDiff(total_LFP, exclfp, inhlfp, tstep, totaltime)

	#raster(excnrns)
	# LFP_peaks = argrelextrema(np.asarray(total_LFP), np.greater)[0]
	# bin_size = period / tstep


	#spikesOutput('spikes.json', LFP_peaks, bin_size)

	#
	#	Timescale information
	#

	#timescales(total_LFP, exclfp, inhlfp, excspikes, inhspikes)
	#graphOscPds(total_LFP)

	exc_nrn = excnrns[random.randint(0, len(excnrns))]
	inh_nrn = inhnrns[random.randint(0, len(inhnrns))]

	p_mempot, p_inh, p_exc, p_back, p_nmda, p_gate  = getCellInfo(exc_nrn)
	b_mempot, b_inh, b_exc, b_back, b_nmda, b_gate  = getCellInfo(inh_nrn)

	#graphFreqDist(excspikes, inhspikes, 50)
	graphNeuron(1,p_mempot, p_inh, p_exc, p_back, p_nmda, p_gate)	#excitatory neuron
	graphNeuron(0,b_mempot, b_inh, b_exc, b_back, b_nmda, b_gate)	#inhibitory neuron



if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()












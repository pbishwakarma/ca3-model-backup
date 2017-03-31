from neuron import Neuron
import graphs as gp
import numpy as np
import scipy.io as sio
import random
import json


nParts = 16

# splits a list into n roughly equal parts and sums each part
# returns a list of n integers
def splitAndSum(l, n):
  avg = len(l) / float(n)
  sums = []
  last = 0.0

  while last < len(l):
    sums.append(sum(l[int(last):int(last + avg)]))
    last += avg

  return sums

##Instantaneous cycle frequencies for Figure 8
def getCycleFreqs(lfp):
	max_freq = 200
	window = 10
	peaks, troughs = gp.getOscillationPds(lfp)
	cycle_freqs = []
	for cyc in peaks:
		end, beg = cyc[1], cyc[0]
		cycle_freqs.append(1/((end - beg)/10))

	cycle_freqs.sort()

	cycles = []

	for i in range(max_freq):
		count = 0
		for j in range(0, len(cycle_freqs)):
			if abs(i - cycle_freqs[j]) < window:
				count += 1
		cycles.append(count)

	return cycles






def getOscPdInfo(lfp, exc, inh, nmda):
	peaks, troughs = gp.getOscillationPds(lfp)
	e_pds, i_pds, nmda_pds = [], [], []

	for i in range(0, len(peaks)):
		e_pds.append(exc[peaks[i][0]:peaks[i][1]])
		i_pds.append(inh[peaks[i][0]:peaks[i][1]])
		nmda_pds.append(nmda[peaks[i][0]:peaks[i][1]])

	return e_pds, i_pds, nmda_pds


#
# Returns two lists of lists containing the spike sums
# in normalized periods. Periods are normalized to 2pi
# scale with bins of pi/8
#
def getConvertedPhase(lfp, exc, inh):
	peaks, troughs = gp.getOscillationPds(lfp)
	e_spikes, i_spikes = [], []
	e_spikeSum, i_spikeSum = [], []
	e_pd_spikes, i_pd_spikes = [], []
	e_pd_sums, i_pd_sums = [], []

	#extract a list of all neuron's spike times at each time t of simulation
	for i in range(len(exc)):
		e_spikes.append(exc[i].getSpikeInfo())

	for i in range(len(inh)):
		i_spikes.append(inh[i].getSpikeInfo())

	# sum of all excitatory neuron's spikes at each time t 
	for j in range(len(e_spikes[0])):
		sum_spikes = 0
		for i in range(len(e_spikes)):
			sum_spikes += e_spikes[i][j]
		e_spikeSum.append(sum_spikes)

	# sum of all inhibitory neuron's spikes at each time t
	for j in range(len(i_spikes[0])):
		sum_spikes = 0
		for i in range(len(i_spikes)):
			sum_spikes += i_spikes[i][j]
		i_spikeSum.append(sum_spikes)


	# returns a list of lists containing the number of spikes at
	# each time step in a given period
	for i in range(len(peaks)):
		beg = peaks[i][0]
		end = peaks[i][1]
		e_pd_spikes.append(e_spikeSum[beg:end])
		i_pd_spikes.append(i_spikeSum[beg:end])

	#iterate over each osc period 
	for i in range(len(e_pd_spikes)):
		#normalize osc pd to 2pi scale and sum
		#with bins of size pi/8
		e_pd_sums.append(splitAndSum(e_pd_spikes[i], nParts))

	#iterate over each osc period
	for i in range(len(i_pd_spikes)):
		#normalize osc pd to 2pi scale and sum
		#with bins of size pi/8
		i_pd_sums.append(splitAndSum(i_pd_spikes[i], nParts))

	return np.array(e_pd_sums), np.array(i_pd_sums)



def getAllNeuronInputs(lfp, exc, inh, nmda):
	peaks, troughs = gp.getOscillationPds(lfp)
	e_pds, i_pds, nmda_pds = [], [], []

	n = random.randint(0, len(peaks))
	beg = peaks[n][0]
	end = peaks[n][1]

	for i in range(0, len(exc)):
		e_pds.append(exc[i][beg:end])

	for j in range(0, len(inh)):
		i_pds.append(inh[j][beg:end])

	for k in range(0, len(nmda)):
		nmda_pds.append(nmda[k][beg:end])

	e_integrated, i_integrated, nmda_integrated = [], [], []

	for i in range(0, len(e_pds)):
		e_integrated.append(abs(np.trapz(np.array(e_pds[i]))))

	for j in range(0, len(i_pds)):
		i_integrated.append(abs(np.trapz(np.array(i_pds[j]))))

	for k in range(0, len(nmda_pds)):
		nmda_integrated.append(abs(np.trapz(np.array(nmda_pds[k]))))


	return e_integrated, i_integrated, nmda_integrated


def integrateCurrent(exc, inh, nmda):
	exc_current, inh_current, nmda_current = [], [], []
	for i in range(0, len(exc)):
		exc_current.append(abs(np.trapz(np.array(exc[i]))))

	for j in range(0, len(inh)):
		inh_current.append(abs(np.trapz(np.array(inh[j]))))

	for k in range(0, len(nmda)):
		nmda_current.append(abs(np.trapz(np.array(nmda[k]))))

	return np.array(exc_current), np.array(inh_current), np.array(nmda_current)



def getNrns(exc, inh):
	excnrns, inhnrns = [], []
	e_mempot, e_inh, e_exc, e_nmda = [], [], [], []
	i_mempot, i_inh, i_exc, i_nmda = [], [], [], []

	for i in range(len(exc)):
		info = exc[i].getInfo()
		excnrns.append(info)
		e_mempot.append(info[0])
		e_inh.append(info[1])
		e_exc.append(info[2])
		e_nmda.append(info[4])


	for j in range(len(inh)):
		info = inh[j].getInfo()
		inhnrns.append(info)
		i_mempot.append(info[0])
		i_inh.append(info[1])
		i_exc.append(info[2])
		i_nmda.append(info[4])

	e_info = [e_mempot, e_exc, e_inh, e_nmda]
	i_info = [i_mempot, i_exc, i_inh, i_nmda]

	return excnrns, inhnrns, e_info, i_info


def getSpikes(exc, inh):
	excnrns, inhnrns = [], []
	for i in range(0, len(exc)):
		spikes = exc[i].getSpikeInfo()
		excnrns.append(spikes)


	for j in range(0, len(inh)):
		spikes = inh[j].getSpikeInfo()
		inhnrns.append(spikes)

	return excnrns, inhnrns


def firingRates(exc, inh, time ):
	excnrns, inhnrns = [], []

	for i in range(len(exc)):
		excnrns.append(exc[i].getFiringRate(time))

	for j in range(len(inh)):
		inhnrns.append(inh[j].getFiringRate(time))

	return [excnrns, inhnrns]


def updateNrns(exc, inh, current_time, timestep):
	for i in range(0, len(exc)):
		exc[i].update(current_time, timestep)

	for j in range(0, len(inh)):
		inh[j].update(current_time, timestep)

def main():
	"""
	Contains the main simulation. 

	This function makes the network, which consists of two types of neurons: excitatory pyramidal cells 
	and inhibitory basket cells. The neuron objects are stored in two lists, and the connectivity is
	established according to the physiological data. This function runs the simulation and stores the
	data from the run in json files. 
	"""

	# constants used in the simulation
	interneuron_count = 50
	pyramidal_count = 200
	
	pp_probability = 0.05
	pb_probability = 0.25	#0.85
	
	bp_probability = 0.15	#0.15
	bb_probability = 0.25   #0.15
	
	time_step = 0.1
	total_time = 600
	current_time = 0.0
	total_neurons = interneuron_count + pyramidal_count
	connProb = 0.0

	inhibitory_neurons = []
	excitatory_neurons = []
	spikes = []
	exc_spike_times, inh_spike_times = [], []
	avg_field_potential = []
	exc_nrn = []	
	inh_nrn= []
	exc_field_potential =[]
	inh_field_potential = []

	print("Model running. \n")

	# initialize neurons
	# type 0 = interneuron, type 1 = pyramidal cell

	for i in range(0, interneuron_count):
		neuron = Neuron(0)
		inhibitory_neurons.append(neuron)

	for i in range(0, pyramidal_count):
		neuron = Neuron(1)
		excitatory_neurons.append(neuron)

	print("Checkpoint 1: Initialized Neurons. \n")

	# make connections within network according to probabilities
	# denoted above

	for i in range(0, interneuron_count):
		for j in range(0, pyramidal_count):
			if random.random() < bp_probability:
				inhibitory_neurons[i].MakeExcConnection(excitatory_neurons[j])
		for k in range(0, interneuron_count):
			if i != k:
				if random.random() < bb_probability:
					inhibitory_neurons[i].MakeInhConnection(inhibitory_neurons[k])

	for i in range(0, pyramidal_count):
		for j in range(0, interneuron_count):
			if random.random() < pb_probability:
				excitatory_neurons[i].MakeInhConnection(inhibitory_neurons[j])
		for k in range(0, pyramidal_count):
			if i != k:
				if random.random() < pp_probability:
					excitatory_neurons[i].MakeExcConnection(excitatory_neurons[k])

	print("Checkpoint 2: Made all connections\n")

	# Choose a random pyramidal neuron and a basket cell to monitor during the simulation
	pyramidal_cell = random.randint(0, pyramidal_count - 1)
	basket_cell = random.randint(0, interneuron_count-1)

	#
	#	Run the simulation. At each time step, calculate the AMPA, GABA, and NMDA conductances
	#	and then calculate the membrane potential
	#

	while(current_time < total_time):
		field_potential = 0.0
		membrane_potential = 0.0
		exc_potential, inh_potential = 0,0
		exc_spikes, inh_spikes = 0,0

		updateNrns(excitatory_neurons, inhibitory_neurons, current_time, time_step)

		for i in range(len(excitatory_neurons)):
			if current_time - excitatory_neurons[i].time_fired > Neuron.refractory or excitatory_neurons[i].time_fired == -1:
				membrane_potential = excitatory_neurons[i].CalcMembranePotential(time_step)
				field_potential += membrane_potential
				exc_potential += membrane_potential
				if membrane_potential >= 1.0:
					excitatory_neurons[i].fire(current_time)
					exc_spikes += 1
				else:
					excitatory_neurons[i].save()
			else:
				excitatory_neurons[i].save()



		for i in range(len(inhibitory_neurons)):
			if current_time - inhibitory_neurons[i].time_fired > Neuron.refractory or inhibitory_neurons[i].time_fired == -1:
				membrane_potential = inhibitory_neurons[i].CalcMembranePotential(time_step)
				field_potential += membrane_potential
				inh_potential += membrane_potential
				if membrane_potential >= 1.0:
					inhibitory_neurons[i].fire(current_time)
					inh_spikes += 1
				else:
					inhibitory_neurons[i].save()
			else:
				inhibitory_neurons[i].save()

		# store the chosen exc/inh neurons' membrane potential, 
		# exc conductance and inh conductance at each time step

		exc_nrn.append(excitatory_neurons[pyramidal_cell].getInfo())
		inh_nrn.append(inhibitory_neurons[basket_cell].getInfo())


		# store the number of spikes at each time step for
		# excitatory and inhibitory neurons
		exc_spike_times.append(exc_spikes)
		inh_spike_times.append(inh_spikes)

		# calculate and store the total LFP
		LFP = field_potential / total_neurons
		avg_field_potential.append(LFP)

		#calculate and store excitatory LFP
		eLFP = exc_potential/pyramidal_count
		iLFP = inh_potential/interneuron_count
		exc_field_potential.append(eLFP)
		inh_field_potential.append(iLFP)

		#move to next timestep
		current_time += time_step
	
	# spike info	
	spikes = [exc_spike_times, inh_spike_times]
	exc_nrns, inh_nrns, e_info, i_info = getNrns(excitatory_neurons, inhibitory_neurons)
	exc_nrn_spikes, inh_nrn_spikes = getSpikes(excitatory_neurons, inhibitory_neurons)
	firing_rates = firingRates(excitatory_neurons, inhibitory_neurons, total_time)


	# Figure 4 information
	e_mempot, e_inh, e_exc, e_nmda = e_info[0], e_info[1], e_info[2], e_info[3]
	i_mempot, i_inh, i_exc, i_nmda = i_info[0], i_info[1], i_info[2], i_info[3]

	e_spike_times = []

	# Integrate currents for figure 4 bottom left
	n = random.randint(0, len(e_inh) - 1)
	m = random.randint(0, len(i_inh) - 1)
	e_exc_pds, e_inh_pds, e_nmda_pds = getOscPdInfo(avg_field_potential, e_exc[n], e_inh[n], e_nmda[n])
	i_exc_pds, i_inh_pds, i_nmda_pds = getOscPdInfo(avg_field_potential, i_exc[m], i_inh[m], i_nmda[m])

	e_integrated_exc, e_integrated_inh, e_integrated_nmda = integrateCurrent(e_exc_pds, e_inh_pds, e_nmda_pds)
	i_integrated_exc, i_integrated_inh, i_integrated_nmda = integrateCurrent(i_exc_pds, i_inh_pds, i_nmda_pds)

	# Integrate currents for figure 4 bottom right
	all_e_exc_pds, all_e_inh_pds, all_e_nmda_pds = getAllNeuronInputs(avg_field_potential, e_exc, e_inh, e_nmda)
	all_i_exc_pds, all_i_inh_pds, all_i_nmda_pds = getAllNeuronInputs(avg_field_potential, i_exc, i_inh, i_nmda)
	

	

	#
	#	Figure 4 
	#		- - 
	# 		- x
	#

	e_periods, i_periods = getConvertedPhase(avg_field_potential, excitatory_neurons, inhibitory_neurons)

	

	for i in range(0, len(exc_spike_times)):
		if i - 50 > 0 or i + 50 < len(exc_spike_times):
			e_spike_times.append(sum(exc_spike_times[i-50:i+50]))
		elif i - 50 < 0:
			e_spike_times.append(sum(exc_spike_times[0:i+50]))
		elif i + 50 > len(exc_spike_times) - 1:
			e_spike_times.append(sum(exc_spike_times[i-50:len(exc_spike_times) - 1]))

	i_spike_times = []

	for i in range(0, len(inh_spike_times)):
		if i - 50 > 0 or i + 50 < len(inh_spike_times):
			i_spike_times.append(sum(inh_spike_times[i-50:i+50]))
		elif i - 50 < 0:
			i_spike_times.append(sum(inh_spike_times[0:i+50]))
		elif i + 50 > len(inh_spike_times) - 1:
			i_spike_times.append(sum(inh_spike_times[i-50: len(inh_spike_times) - 1]))


	#
	#		Fiure 8
	#
	
	freq_sums = getCycleFreqs(avg_field_potential)

	sio.savemat('Run Data\e_integrated_exc.mat', {'e_exc':e_integrated_exc})
	sio.savemat('Run Data\cycle_freqs.mat', {'freq_sums':freq_sums})



	print("Checkpoint 3: Calculated LFPs\n")

	LFP_output = open("LFPOutput.txt", 'w')

	with open('Run Data/excnrns.json', 'w') as excnrns_file:
		json.dump(exc_nrns, excnrns_file)

	with open('Run Data/inhnrns.json', 'w') as inhnrns_file:
		json.dump(inh_nrns, inhnrns_file)

	with open('Run Data/exclfp.json', 'w') as exclfp:
		json.dump(exc_field_potential, exclfp)

	with open('Run Data/inhlfp.json', 'w') as inhlfp:
		json.dump(inh_field_potential, inhlfp)

	with open('Run Data/spikes.json', 'w') as spk_file:
		json.dump(spikes, spk_file)

	with open('Run Data/excspikes.json', 'w') as exc_spk_file:
		json.dump(exc_spike_times, exc_spk_file)

	with open('Run Data/inhspikes.json', 'w') as inh_spk_file:
		json.dump(inh_spike_times, inh_spk_file)

	with open('Run Data/LFP.json', 'w') as lfp_file:
		json.dump(avg_field_potential, lfp_file)

	with open('Run Data/firingrates.json', 'w') as firing_rate_file:
		json.dump(firing_rates, firing_rate_file)

	# with open('Run Data/exc_nrn.json', 'w') as exc:
	# 	json.dump(exc_nrn, exc)

	# with open('Run Data/inh_nrn.json', 'w') as inh:
	# 	json.dump(inh_nrn, inh)


	e_mempot = np.array(e_mempot)
	e_inh = np.array(e_inh)
	e_exc = np.array(e_exc)
	e_nmda = np.array(e_nmda)

	i_mempot = np.array(i_mempot)
	i_inh = np.array(i_inh)
	i_exc = np.array(i_exc)
	i_nmda = np.array(i_nmda)


	# save sums of converted phases to matlab file
	sio.savemat('Run Data\e_periods.mat', {'e_periods':e_periods})
	sio.savemat('Run Data\i_periods.mat', {'i_periods':i_periods})

	#save integrated current (top right) information
	sio.savemat('Run Data\e_integrated_exc.mat', {'e_exc':e_integrated_exc})
	sio.savemat('Run Data\e_integrated_inh.mat', {'e_inh':e_integrated_inh})
	sio.savemat('Run Data\i_integrated_exc.mat', {'i_exc':i_integrated_exc})
	sio.savemat('Run Data\i_integrated_inh.mat', {'i_inh':i_integrated_inh})
	sio.savemat('Run Data\e_integrated_nmda.mat', {'e_nmda':e_integrated_nmda})
	sio.savemat('Run Data\i_integrated_nmda.mat', {'i_nmda':i_integrated_nmda})

	#save integrated current (bottom right) information
	sio.savemat('Run Data\e_all_integrated_exc.mat', {'e_exc':all_e_exc_pds})
	sio.savemat('Run Data\e_all_integrated_inh.mat', {'e_inh':all_e_inh_pds})
	sio.savemat('Run Data\e_all_integrated_nmda.mat', {'e_nmda':all_e_nmda_pds})

	sio.savemat('Run Data\i_all_integrated_exc.mat', {'i_exc':all_i_exc_pds})
	sio.savemat('Run Data\i_all_integrated_inh.mat', {'i_inh':all_i_inh_pds})
	sio.savemat('Run Data\i_all_integrated_nmda.mat', {'i_nmda':all_i_nmda_pds})

	sio.savemat('Run Data\e_mempot.mat', {'e_mempot':e_mempot})
	sio.savemat('Run Data\e_inh.mat', {'e_inh':e_inh})
	sio.savemat('Run Data\e_exc.mat', {'e_exc':e_exc})
	sio.savemat('Run Data\e_nmda.mat', {'e_nmda':e_nmda})

	sio.savemat('Run Data\i_mempot.mat', {'i_mempot':i_mempot})
	sio.savemat('Run Data\i_inh.mat',{'i_inh':i_inh})
	sio.savemat('Run Data\i_exc.mat',{'i_exc': i_exc})
	sio.savemat('Run Data\i_nmda.mat', {'i_nmda':i_nmda})

	# FIGURE 2
	#	
	#	x -   and - - 
	#	- -   and x - 
	#

	exc_nrn_spikes = np.array(exc_nrn_spikes)
	inh_nrn_spikes = np.array(inh_nrn_spikes)

	np.savetxt('Run Data\exc_spikes', exc_nrn_spikes)
	np.savetxt('Run Data\inh_spikes', inh_nrn_spikes)
	#np.savetxt('Run Data\e_firing_rates', firing_rates)


	exc_fire_rate, inh_fire_rate = firing_rates[0], firing_rates[1]

	exc_fire_rate = np.array(exc_fire_rate)
	inh_fire_rate = np.array(inh_fire_rate)
	i_spike_times = np.array(i_spike_times)
	e_spike_times = np.array(e_spike_times)

	sio.savemat('Run Data/exc_nrn_spikes.mat', {'exc_spikes': exc_nrn_spikes})
	sio.savemat('Run Data/inh_nrn_spikes.mat', {'inh_spikes': inh_nrn_spikes})
	sio.savemat('Run Data/exc_fire_rate.mat', {'e_firerate': exc_fire_rate})
	sio.savemat('Run Data/inh_fire_rate.mat', {'i_firerate': inh_fire_rate})
	sio.savemat('Run Data/e_spike_times.mat', {'e_spike_times':e_spike_times})
	sio.savemat('Run Data/i_spike_times.mat', {'i_spike_times':i_spike_times})

	LFP_output.write("total_time = %d \n" % total_time)
	LFP_output.write("time_step = %f \n" % time_step)
	LFP_output.write("pp_probability = %f \n" % pp_probability)
	LFP_output.write("pb_probability = %f \n" % pb_probability)
	LFP_output.write("bp_probability = %f \n" % bp_probability)
	LFP_output.write("bb_probability = %f \n" % bb_probability)
	LFP_output.write("(exc_amp, inh_amp) = %s \n" % str((Neuron.exc_amp, Neuron.inh_amp)))
	LFP_output.write("(tau_pp, tau_pb) = %s \n" % str((Neuron.tau_pp, Neuron.tau_pb)))
	LFP_output.write("(tau_bp, tau_bb) = %s \n" % str((Neuron.tau_bp, Neuron.tau_bb)))
	LFP_output.write("(pp_latency, pb_latency) = %s \n" % str((Neuron.pp_latency, Neuron.pb_latency)))
	LFP_output.write("(bp_latency, bb_latency) = %s \n" % str((Neuron.bp_latency, Neuron.bb_latency)))

	for i in range(len(avg_field_potential)):
		output = "%f \n" % avg_field_potential[i]
		LFP_output.write(output)
		

	LFP_output.close()


	print("Checkpoint 4: Wrote data to files")


if __name__ == "__main__":
	main()

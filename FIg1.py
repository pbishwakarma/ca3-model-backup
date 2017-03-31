from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

exc_tag = True
inh_tag = False

# exc_tag = False
# inh_tag = True

def saveSimulation(in_neuron, PN, IN):
	in_neuron_mempot = in_neuron.getInfo()[0]
	PN_mempot = PN.getInfo()[0]
	IN_mempot = IN.getInfo()[0]

	


	filename = "D:\W&M\W&M Senior Year\Honors Thesis\CA3 Paper - 3.1.17\Model\Run Data\Fig1Data"
	
	global exc_tag

	if exc_tag:
		filename += "\exc"
	else:
		filename += "\inh"

	input_filename = filename + "_input_mempot.mat"
	PN_filename = filename + "_PN_mempot.mat"
	IN_filename = filename + "_IN_mempot.mat"

	sio.savemat(input_filename, {'in_neuron_mempot': in_neuron_mempot})
	sio.savemat(PN_filename, {'PN_mempot': PN_mempot})
	sio.savemat(IN_filename, {'IN_mempot': IN_mempot})







def runSimulation():

	SIM_TIME = 250
	TIME_STEP = 0.1
	CUR_TIME = 0.0

	# global exc_tag
	# global inh_tag

	# exc_tag = False
	# inh_tag = True

	if exc_tag:
		in_neuron = Neuron(1)
	elif inh_tag:
		in_neuron = Neuron(0)

	PN = Neuron(1)
	IN = Neuron(0)

	out_neurons = [PN, IN]
	
	for i in range(0, len(out_neurons)):
		out_neurons[i]._membrane_potential = 0
		if exc_tag:
			out_neurons[i].MakeExcConnection(in_neuron)
		if inh_tag:
			out_neurons[i].MakeInhConnection(in_neuron)

	spikes = 0
	while CUR_TIME < SIM_TIME:
		in_neuron.update(CUR_TIME, TIME_STEP)
		if inh_tag:
			in_neuron._background_current += 0.1
		if CUR_TIME > 100 and spikes < 1:
			if CUR_TIME - in_neuron.time_fired > Neuron.refractory or in_neuron.time_fired == -1:
				mem_pot = in_neuron.CalcMembranePotential(TIME_STEP)
				if mem_pot >= 1.0:
					in_neuron.fire(CUR_TIME)
					spikes += 1
		else:
			in_neuron._prev_membrane_potential = 0
			in_neuron._membrane_potential = 0
		
		for i in range(0, len(out_neurons)):
			out_neurons[i].update(CUR_TIME, TIME_STEP)
			out_neurons[i]._background_current = 0
			out_neurons[i]._prev_background_current = 0

			if CUR_TIME - out_neurons[i].time_fired > Neuron.refractory or out_neurons[i].time_fired == -1:
				mem_pot = out_neurons[i].CalcMembranePotential(TIME_STEP)
				if mem_pot >= 1.0:
					out_neurons[i].fire(CUR_TIME)

		CUR_TIME += TIME_STEP

	return out_neurons, in_neuron

def graphSimulation(in_neuron, PN, IN):

	global exc_tag
	global inh_tag

	PN_info = PN.getInfo()
	IN_info = IN.getInfo()
	in_neuron_info = in_neuron.getInfo()

	in_neuron_mempot = in_neuron_info[0]
	print(len(in_neuron_mempot))
	PN_mempot = PN_info[0]
	IN_mempot = IN_info[0]

	x1 = np.linspace(0, len(in_neuron_mempot), len(in_neuron_mempot))
	x = np.linspace(0, len(PN_mempot), len(PN_mempot))
	t = np.linspace(0, len(IN_mempot), len(IN_mempot))


	fig = plt.figure()

	if exc_tag:
		fig.suptitle("Input from a PN")
	elif inh_tag:
		fig.suptitle("Input from an IN")

	ax = fig.add_subplot(311)
	ax.set_ylabel("input Neuron")
	ax.plot(x1, in_neuron_mempot)

	ax = fig.add_subplot(312)
	ax.set_ylabel("PN")
	ax.plot(x, PN_mempot)
	#plt.ylim((0,0.1))

	ax = fig.add_subplot(313)
	ax.set_ylabel("IN")
	ax.plot(t, IN_mempot)
	#plt.ylim((0,1.0))

	plt.show()


def main():

	global exc_tag
	global inh_tag

	exc_tag, inh_tag = True, False
	neurons, in_neuron = runSimulation()
	PN = neurons[0]
	IN = neurons[1]
	saveSimulation(in_neuron, PN, IN)
	graphSimulation(in_neuron, PN, IN)

	exc_tag, inh_tag = False, True
	neurons, in_neuron = runSimulation()
	PN = neurons[0]
	IN = neurons[1]
	saveSimulation(in_neuron, PN, IN)
	graphSimulation(in_neuron, PN, IN)



main()





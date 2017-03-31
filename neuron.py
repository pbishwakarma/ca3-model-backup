import math
import random
import numpy as np

class Neuron():

	exc_amp = 0.10		# 0.10
	inh_amp = 0.65		# 0.65
	tau_pp = 1.1		# 1.7
	tau_pb = 3.3		# known = 3.3
	tau_bp = 1.6		# 1.6
	tau_bb = 1.2		# known = 1.2
	bb_latency = 0.6	# known = 0.6
	bp_latency = 0.5
	pb_latency = 1.1	# known = 1.1
	pp_latency = 1.8
	E_inh = -0.67		# known = -0.67
	E_exc = 4.67		# known = 4.67
	gl = 0.05			# known = 0.05
	refractory = 2

	# nmda constants
	b_act_step, b_deact_step = 0.3, 7.0		# 0.1, 3.0
	p_act_step, p_deact_step = 0.28, 2.8	# 0.12, 3.8
	tau_nmda_act = 6.5
	tau_nmda_deact = 145.0
	E_nmda = 5.50 		# 


	def __init__(self, neuron):
		"""
		Constructs a neuron object

		Args
		------------------------------------------------------------------
		neuron (int) -- denotes type of neuron; 
			0 = basket cell 
			1 = pyramidal cell

		"""
		assert neuron == 0 or neuron == 1,"Neuron type must be 0 or 1"
	
		self._type = neuron
		self._membrane_potential, self._prev_membrane_potential = random.random(), 0.0
		self._background_current, self._prev_background_current = 0.0, 0.0
		self._exc_conduc, self._prev_exc_conduc = 0.0, 0.0
		self._inh_conduc, self._prev_inh_conduc = 0.0, 0.0

		self._nmda_conduc, self._prev_nmda_conduc = 0.0, 0.0
		self._prev_nmda_act, self._nmda_act = 0,0
		self._prev_nmda_deact, self._nmda_deact = 0 ,0
		self._mgblock = 0

		self.time_fired = -1.0
	
		# Lists store data about membrane potential, spikes and
		# excitatory/inhibitory connections to other neurons
		self._mem_pot = []
		self._excConn, self._inhConn = [], []
		self._spikes = []
		self._exc, self._inh, self._nmda = [],[],[]
		self._back, self._gate = [], []

	def MakeExcConnection(self, neuron):
		"""
		Makes a connection with an excitatory neuron object by storing
		it in a list 

		Args
		------------------------------------------------------------------
		neuron (Neuron) -- neuron object to make connection with

		"""
		self._excConn.append(neuron)

	def MakeInhConnection(self, neuron):
		"""
		Makes a connection with an inhibitory neuron object by storing
		it in a list

		Args
		------------------------------------------------------------------
		neuron (Neuron)-- neuron object to make connection with

		"""
		self._inhConn.append(neuron)


	def CalcNMDAConductance(self, time, timestep):
		"""
		Calculates the conductance using the Euler method on the 
		differential equations that govern the activation and deactivation 
		of the NMDA channel. Model based on the classic model of the NMDA 
		current from Moradi et al (2013).

		Args
		------------------------------------------------------------------
		time (float) -- current time step during the simulation
		timestep (float) -- length of timestep used in the simulation

		"""

		assert type(time) == float, 'Time must be a float'
		nmda_sum = 0.0
		self._nmda_act, self._nmda_deact = 0, 0
		for i in range(0, len(self._excConn)):
			if self._type == 1:
				if self._excConn[i].time_fired != -1.0 and time > self._excConn[i].time_fired + Neuron.pb_latency:
					self._nmda_act += Neuron.p_act_step
					self._nmda_deact += Neuron.p_deact_step
			elif self._type == 0:
				if self._excConn[i].time_fired != -1.0 and time > self._excConn[i].time_fired + Neuron.pp_latency:
					self._nmda_act += Neuron.b_act_step
					self._nmda_deact += Neuron.b_deact_step

		dactdt = -self._nmda_act / Neuron.tau_nmda_act
		ddeactdt = -self._nmda_deact / Neuron.tau_nmda_deact

		#self._mgblock = 1/(1+ (math.exp(-6*(self._membrane_potential - 0.3))))
		#self._mgblock = 1/(1+ (math.exp(-8.0*(self._membrane_potential - 0.3))))
		self._mgblock = 1/(1+ (math.exp(-10*(self._membrane_potential-0.6))))
		self._nmda_act = self._prev_nmda_act + (dactdt * timestep)
		self._nmda_deact = self._prev_nmda_deact + (ddeactdt * timestep)

		nmda_sum = self._mgblock * (self._nmda_deact - self._nmda_act)

		self._prev_nmda_conduc = self._nmda_conduc
		self._nmda_conduc = nmda_sum


	def CalcExcConductance(self, time):
		"""
		Iterates through excitatory connections and calculates the
		conductance using an exponential decay mdoel

		Args
		------------------------------------------------------------------
		time (float) -- current time step during the simulation
		
		"""

		assert type(time) == float, "Time must be a float"
		exc_sum = 0.0
		time_diff = 0.0

		for i in range(0, len(self._excConn)):
			if self._type == 0:
				if self._excConn[i].time_fired != -1.0 and time > self._excConn[i].time_fired + Neuron.pb_latency:
					exc_sum += Neuron.exc_amp * math.exp(-(time-self._excConn[i].time_fired + Neuron.pb_latency)/Neuron.tau_pb)
			elif self._type == 1:
				if self._excConn[i].time_fired != -1.0 and time > self._excConn[i].time_fired + Neuron.pp_latency:
					exc_sum += Neuron.exc_amp * math.exp(-(time-self._excConn[i].time_fired + Neuron.pp_latency)/Neuron.tau_pp)

			self._prev_exc_conduc = self._exc_conduc
			self._exc_conduc = exc_sum


	def CalcInhConductance(self, time):
		"""
		Iterates the inhibitory connections and calculates the
		conductance using an exponential decay model

		Args
		------------------------------------------------------------------
		time (float) -- current time step during the simulation
		
		"""

		assert type(time) == float, "Time must be a float"
		inh_sum = 0.0
		time_diff = 0.0

		for i in range(0, len(self._inhConn)):
			if self._type == 0:
				if self._inhConn[i].time_fired != -1.0 and time > self._inhConn[i].time_fired + Neuron.bb_latency:
					inh_sum += Neuron.inh_amp * math.exp(-(time-self._inhConn[i].time_fired + Neuron.bb_latency)/Neuron.tau_bb)
			elif self._type == 1:
				if self._inhConn[i].time_fired != -1.0 and time > self._inhConn[i].time_fired + Neuron.bp_latency:
					inh_sum += Neuron.inh_amp * math.exp(-(time-self._inhConn[i].time_fired + Neuron.bp_latency)/Neuron.tau_bp)

		self._prev_inh_conduc = self._inh_conduc
		self._inh_conduc = inh_sum


	def CalcBackgroundCurrent(self):
		"""
		Calculates background current coming in from Schaeffer
		Collaterals. Current is a randomly generated number
		between 0 and 0.1 for pyramidal cells and between
		0 and 0.01 for basket cells.

		"""

		# make background current different for pyramidal/basket cell
		# poisson process 
		# save background excitation for pyr/basket in plot
		# 10x lower excitation to basket cells 
		if self._type == 1:
			self._prev_background_current = self._background_current
			self._background_current = random.random()/5.75 
		else:
			self._prev_background_current = self._background_current
			self._background_current = random.random()/100


	def CalcMembranePotential(self, timestep):
		"""
		Calculates the membrane potential of the neuron object
		using the Euler method on the membrane equation which is denoted
		below:

		dvdt  =	-g(V) - g_inh(V - E_inh) - g_exc(V - E_exc) - 
				g(V - E_nmda) + i_background

		Args
		------------------------------------------------------------------
		timestep (float) -- current timestep of the simulation

		Returns
		------------------------------------------------------------------
		self._membrane_potential (float) -- returns the membrane potential
		"""


		self._prev_membrane_potential = self._membrane_potential
		dvdt = -Neuron.gl * self._prev_membrane_potential - \
		self._inh_conduc * (self._prev_membrane_potential - Neuron.E_inh) - \
		self._exc_conduc * (self._prev_membrane_potential - Neuron.E_exc) + \
		self._nmda_conduc * (self._prev_membrane_potential - Neuron.E_nmda) + \
		self._background_current


		self._membrane_potential = self._prev_membrane_potential + (dvdt * timestep)	
		return self._membrane_potential

	def update(self, current_time, timestep):
		self.CalcExcConductance(current_time)
		self.CalcInhConductance(current_time)
		self.CalcBackgroundCurrent()

		#if self._type == 0:
		#	self.CalcNMDAConductance(current_time, timestep)

		# if self._type == 1:
		# 	self.CalcNMDAConductance(current_time, timestep)

		self.CalcNMDAConductance(current_time, timestep)
		self.saveInfo()

	def saveInfo(self):
		inh = -self._inh_conduc * (self._prev_membrane_potential - Neuron.E_inh)
		exc = -self._exc_conduc * (self._prev_membrane_potential - Neuron.E_exc)
		nmda = self._nmda_conduc * (self._prev_membrane_potential - Neuron.E_nmda)
	
		self._mem_pot.append(self._membrane_potential)
		self._inh.append(inh)
		self._exc.append(exc)
		self._nmda.append(nmda)
		self._back.append(self._background_current)
		self._gate.append(self._mgblock)

	def getInfo(self):
		"""
		Calculates and returns a list containing the following information:
		(membrane potential, inhibitory current, excitatory current, 
		background current, NMDA current)

		Returns
		------------------------------------------------------------------
		info (list) -- a list containing float values for membrane potential,
		inhibitory current, excitatory current, background current, and NMDA
		current

		"""
		return [self._mem_pot, self._inh, self._exc, self._back, self._nmda, self._gate]

	def fire(self, time):
		"""
		Simulates a neuron spike. Resets the membrane potential and 
		previous membrane potential to zero. Saves the time at which 
		the spike occurs.

		Args
		------------------------------------------------------------------
		time (float) -- the current time step in the simulation 

		"""

		self._prev_membrane_potential = 0
		self._membrane_potential = 0
		self.time_fired = time
		self._spikes.append(1)

	def save(self):
		self._spikes.append(0)
		
	def getSpikeInfo(self):
		return self._spikes

	def getFiringRate(self, time):

		numspikes = self._spikes.count(1)
		avg_spikes = 1000 * numspikes / time
		return avg_spikes







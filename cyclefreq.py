import json
import graphs as gp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

NMDA_lfp_file = "D:\W&M\W&M Senior Year\Honors Thesis\CA3 Paper - 3.1.17\Model\cyclefreq\withNMDALFP.json"
no_NMDA_lfp_file = "D:\W&M\W&M Senior Year\Honors Thesis\Ca3 Paper - 3.1.17\Model\cyclefreq\withoutNMDALFP.json"

def getCycles(lfp_file):
	with open(lfp_file) as myFile:
		lfp = json.load(myFile)


	pds = gp.getOscillationPds(lfp)[0]

	freqs = []

	for pd in pds:
		beg, end = pd[0], pd[1]
		freqs.append((1/(end-beg)*10000))

	freqs.sort()
	maxFreq = freqs[-1]
	window = 10

	cycles = []

	for i in range(int(maxFreq)):
		count = 0
		for freq in freqs:
			if abs(i - freq) < window:
				count += 1
		cycles.append(count)

	return cycles

nmda_cycles = getCycles(NMDA_lfp_file)
no_nmda_cycles = getCycles(no_NMDA_lfp_file)


t = np.linspace(0, len(nmda_cycles), len(nmda_cycles))
x = np.linspace(0, len(no_nmda_cycles), len(no_nmda_cycles))

print("NMDA calculated over " + str(len(nmda_cycles)) + " cycles.")
print("No NMDA calculated over " + str(len(no_nmda_cycles)) + " cycles.")
# print(t)
# print(cycles)
plt.plot(t, nmda_cycles)
#plt.plot(x, no_nmda_cycles)

#plt.plot(t, nmda_cycles, x, no_nmda_cycles)
plt.legend()

plt.show()

nmda_cycles = np.array(nmda_cycles)
no_nmda_cycles = np.array(no_nmda_cycles)


filename = "D:\W&M\W&M Senior Year\Honors Thesis\CA3 Paper - 3.1.17\Model\Run Data\Fig1Data"
nmda_file = filename + "\cycles_nmda.mat"
no_nmda_file = filename + "\cycles_no_nmda.mat"

sio.savemat(nmda_file, {'nmda_cycles':nmda_cycles})
sio.savemat(no_nmda_file, {'no_nmda_cycles':no_nmda_cycles})







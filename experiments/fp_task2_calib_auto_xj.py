#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 2 - Calibrating Membrane Time Constant

Andreas Baumbach, October 2017, andreas.baumbach@kip.uni-heidelberg.de
'''
# load PyNN interface for the Spikey neuromorphic hardware
import os
import copy
import numpy as np

import pyNN.hardware.spikey as pynn
# for plotting without X-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa

####################################################################
# experiment parameters
# in biological time and parameter domain
####################################################################

use_other_spikey_half = False
runtime = 2000.0  # ms -> 0.1ms on hardware

neuron_offset = 0

neuronParams = {
    'v_reset'   : -80.0,  # mV
    'e_rev_I'   : -75.0,  # mV
    'v_rest'    : -50.0,  # mV
    'v_thresh'  : -55.0,  # mV  - default value.
                          # Change to result of your calculation!
    'g_leak'    :  20.0   # nS  -> tau_mem = 0.2nF / 20nS = 10ms
}
pynn.setup(calibTauMem=False, mappingOffset=neuron_offset)


targetrate = 20.
if not os.path.exists('gls.dat'):
	gls = np.ones(192) * 20.
else:
	gls = np.loadtxt('gls.dat')


neurons = pynn.Population(192, pynn.IF_facets_hardware1, neuronParams)
print(neurons)
neurons.set({'g_leak' : 20})
neurons.record()

pynn.record_v([neurons[0], neurons[1]], '')
pynn.run(runtime)

spikes = neurons.getSpikes()

rates = []
for i in range(192):
	times = []
	for j in range(np.shape(spikes)[0]):
		if (spikes[j][0] == i):
			times.append(spikes[j][1])
	current_rate = np.mean(np.diff(times))
	rates.append(current_rate)
	
# plot histogram of used gls and resulting rates
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(rates)
ax.set_title("Firing rates")
ax.set_xlabel("Rate [Hz]")
ax.set_ylabel("Frequency")
plt.savefig("firing_rates.pdf")
plt.show()
pynn.end()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 1 - Investigating a Single Neuron

Andreas Gruebl, July 2016, agruebl@kip.uni-heidelberg.de
'''
# load PyNN interface for the Spikey neuromorphic hardware
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
runtime = 200.0  # ms -> 0.1ms on hardware

neuronParams = {
    'v_reset'   : -80.0,  # mV
    'e_rev_I'   : -75.0,  # mV
    'v_rest'    : -50.0,  # mV - default is -75.0 mV
    'v_thresh'  : -55.0,  # mV
    'g_leak'    :  20.0   # nS  -> tau_mem = 0.2nF / 20nS = 10ms
}

####################################################################
# procedural experiment description
####################################################################
# necessary setup - do not load calibration of membrane time constant
# (i.e. g_leak) because we want to do this manually.
if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0
pynn.setup(calibTauMem=False, mappingOffset=neuron_offset)

# set up one neuron
neuron = pynn.Population(1, pynn.IF_facets_hardware1, neuronParams)

# increase refractory period by reducing hardware parameter icb
# -> play around with this parameter to obtain a visible refractory period.
#    it is a technical parameter only, with no direct translation to
# biological parameters! value range is from 0 to 2.5
pynn.hardware.hwa.setIcb(0.2)

# define which observables to record
# spike times (digital: neuron number and time stamp)
neuron.record()

# membrane potential (analog: digitized membrane voltage over time)
pynn.record_v(neuron[0], '')

# execute the experiment
pynn.run(runtime)

# evaluate results
spikes = neuron.getSpikes()[:, 1]

# mean firing rate
mean_firing_rate = len(spikes) / (spikes[-1] - spikes [0])
print("mean firing rate")
print(mean_firing_rate)

# time differences
timeDifferences = np.diff(spikes)

print("time differences mean")
print(np.mean(timeDifferences))
print("time differences std deviation")
print(np.std(timeDifferences))

membrane = pynn.membraneOutput
membraneTime = pynn.timeMembraneOutput

pynn.end()

####################################################################
# data visualization
####################################################################

print 'average membrane potential:', np.mean(membrane), 'mV'
print 'sampling step for membrane potential: {} ms'.format(
                                membraneTime[1] - membraneTime[0])

# draw raster plot
ax = plt.subplot(211)  # row, col, nr
for spike in spikes:
    ax.axvline(x=spike)
ax.set_xlim(0, runtime)
ax.set_ylabel('spikes')
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

# draw membrane potential
axMem = plt.subplot(212)
axMem.plot(membraneTime, membrane)
axMem.set_xlim(0, runtime)
axMem.set_xlabel('time (ms)')
axMem.set_ylabel('membrane potential (mV)')

plt.savefig('SingleNeuron/fp_task1_1membrane.png')

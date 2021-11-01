#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 5 - Feedforward Networks (Synfire Chain)

Andreas Gruebl, July 2016, agruebl@kip.uni-heidelberg.de

Simple example of synfire chain with feedforward inhibition.
See the following publication for more details:

Pfeil et al. (2013).
Six networks on a universal neuromorphic computing substrate.
Front. Neurosci. 7 (11).
'''
import pyNN.hardware.spikey as pynn
import numpy as np

# for plotting without X-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa

use_other_spikey_half   = False
close_chain             = False
runtime = 500.0     # ms
noPops = 9         # chain length
popSize = {'exc': 10, 'inh': 10}  # size of each chain link
# connection probabilities
probExcExc = 1.0
probExcInh = 1.0
probInhExc = 1.0

# refractory period of neurons can be tuned for optimal synfire chain behavior
neuronParams = {'tau_refrac' : 10.0}

if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0
pynn.setup(mappingOffset=neuron_offset)

# define weights in digital hardware values
# --> these should be tuned first to obtain synfire chain behavior!

#original: 12, 12, 8, 10, 7
# we tested 5,4,3,2,15

see = 2
sei = 0
ee = 2
ei = 0
ie = 0

weightStimExcExc = see * pynn.minExcWeight()
weightStimExcInh = sei * pynn.minExcWeight()
weightExcExc     = ee * pynn.minExcWeight()
weightExcInh     = ei * pynn.minExcWeight()
weightInhExc     = ie * pynn.minInhWeight()

# kick starter input pulse(s)
stimSpikes = np.array([100.0])  # one trigger pulse
# stimSpikes = np.array([100.0, 200.0, 300.0])  # multiple trigger pulses

stimExc = pynn.Population(popSize['exc'], pynn.SpikeSourceArray,
                                            {'spike_times': stimSpikes})

# create neuron populations
popCollector = {'exc': [], 'inh': []}
for synType in ['exc', 'inh']:
    for popIndex in range(noPops):
        pop = pynn.Population(popSize[synType], pynn.IF_facets_hardware1,
                                                            neuronParams)
        pop.record()
        popCollector[synType].append(pop)

# connect stimulus
pynn.Projection(stimExc, popCollector['exc'][0],
                pynn.FixedProbabilityConnector(p_connect=probExcExc,
                                               weights=weightStimExcExc),
                                               target='excitatory')
pynn.Projection(stimExc, popCollector['inh'][0],
                pynn.FixedProbabilityConnector(p_connect=probExcInh,
                                               weights=weightStimExcInh),
                                               target='excitatory')

# connect synfire chain populations
# see figure ... in script for the illustration of the network topology
# for closing the loop you need to change the for loop range
# i.e. if popIndex < noPops - 1: open chain
if close_chain:
    lastiter = noPops
else:
    lastiter = noPops - 1
for popIndex in range(lastiter):
    pynn.Projection(popCollector['exc'][popIndex],
                    popCollector['exc'][(popIndex + 1) % noPops],
                    pynn.FixedProbabilityConnector(p_connect=probExcExc,
                                    weights=weightExcExc), target='excitatory')
    pynn.Projection(popCollector['exc'][popIndex],
                    popCollector['inh'][(popIndex + 1) % noPops],
                    pynn.FixedProbabilityConnector(p_connect=probExcInh,
                                    weights=weightExcInh), target='excitatory')
    pynn.Projection(popCollector['inh'][popIndex],
                    popCollector['exc'][popIndex],
                    pynn.FixedProbabilityConnector(p_connect=probInhExc,
                                    weights=weightInhExc), target='inhibitory')

# record from first neuron of first excitatory population of chain
pynn.record_v(popCollector['exc'][0][0], '')

# run chain...
pynn.run(runtime)

# collect all spikes in one array
spikeCollector = np.array([]).reshape(0, 2)
for synType in ['exc', 'inh']:
    for popIndex in range(noPops):
        spikeCollector = np.vstack((spikeCollector,
                                popCollector[synType][popIndex].getSpikes()))

# get membrane
membrane = pynn.membraneOutput
membraneTime = pynn.timeMembraneOutput

pynn.end()

# visualize
print 'number of spikes:', len(spikeCollector)

color = 'k'

ax = plt.subplot(211)  # row, col, nr
ax.plot(spikeCollector[:, 1], spikeCollector[:, 0], ls='', marker='o',
                                                ms=1, c=color, mec=color)
ax.set_xlim(0, runtime)
ax.set_xticklabels([])
ax.set_ylim(-0.5, (popSize['exc'] + popSize['inh']) * noPops - 0.5)
ax.set_ylabel('neuron ID')
# color excitatory and inhibitory neurons
ax.axhspan(-0.5, popSize['exc'] * noPops - 0.5, color='r', alpha=0.2)
ax.axhspan(popSize['exc'] * noPops - 0.5,
        (popSize['exc'] + popSize['inh']) * noPops - 0.5, color='b', alpha=0.2)

axMem = plt.subplot(212)
axMem.plot(membraneTime, membrane)
axMem.set_xlim(0, runtime)
axMem.set_xlabel('time (ms)')
axMem.set_ylabel('membrane potential (mV)')
plt.show()
plt.savefig('FeedForwardNetworks/task1_with_disabeld_inhibition.png')

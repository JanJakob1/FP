#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 4 - Short Term Plasticity

Andreas Gruebl, July 2016, agruebl@kip.uni-heidelberg.de

This network demonstrates short-term plasticity (STP) on hardware.
The postsynaptic neuron is stimulated by a single input with STP enabled.
For high input rates the impact of each presynaptic spike on the membrane
potential decreases.
For low input rates the synaptic efficacy recovers.
'''
import pyNN.hardware.spikey as pynn
import numpy as np

# for plotting without X-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa

# row and column of synapse
# you can play around with these parameters to find a "nice" synapse...
neuronIndex         = 42
synapseDriverIndex  = 42
use_other_spikey_half = False

weight = 15.0
stimParams = {'spike_times': np.concatenate((np.arange(100.0, 401.0, 50.0),
                                            [700.0]))}

# STP parameters (depression and facilitation cannot be enabled
# simultaneously!):
# U: Usable synaptic efficacy (U_SE, see script) - scales the size of PSPs.
# U has to lie between 0 and 1
# tau_rec: time constant of short term depression
# tau_facil: time constant of short term facilitation
# either tau_rec or tau_facil must be zero
stpParams = {'U': 0.4, 'tau_rec': 100.0, 'tau_facil': 0.0}
runtime = 1000.0

if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0
pynn.setup(mappingOffset=neuronIndex+neuron_offset)
if weight > 0:
    weight *= pynn.minExcWeight()
    synapsetype = 'excitatory'
else:
    weight *= pynn.minInhWeight()
    synapsetype = 'inhibitory'

neuron = pynn.Population(1, pynn.IF_facets_hardware1)
dummy = pynn.Population(synapseDriverIndex, pynn.SpikeSourceArray, stimParams)
stimulus = pynn.Population(1, pynn.SpikeSourceArray, stimParams)

# enable and configure STP
stp_model = pynn.TsodyksMarkramMechanism(**stpParams)
pynn.Projection(stimulus, neuron,
                method=pynn.AllToAllConnector(weights=weight),
                target='excitatory',
                synapse_dynamics=pynn.SynapseDynamics(fast=stp_model))

pynn.record_v(neuron[0], '')

pynn.run(runtime)

membrane = np.array(zip(pynn.timeMembraneOutput, pynn.membraneOutput))

pynn.end()

# plot
plt.plot(membrane[:, 0], membrane[:, 1])
plt.xlabel('time (ms)')
plt.ylabel('membrane potential (mV)')
plt.savefig('stp.png')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 6 - Decorrelation Network

Andreas Gruebl, July 2016, agruebl@kip.uni-heidelberg.de

Random network with purely inhibitory connections.
Neurons are driven by setting resting potential over spiking threshold.

See also:
Pfeil et al. (2014).
The effect of heterogeneity on decorrelation mechanisms in spiking neural
networks: a neuromorphic-hardware study.
arXiv:1411.7916 [q-bio.NC].
'''
import pyNN.hardware.spikey as pynn
import numpy as np

# for plotting without X-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa

use_other_spikey_half   = False
if use_other_spikey_half:
    neuron_offset = 192
else:
    neuron_offset = 0
pynn.setup(mappingOffset=neuron_offset)

runtime = 10000.0  # ms
popSize = 192

# tune these two parameters
active_connections = False
weight = 4.0 * pynn.minInhWeight()
numInhPerNeuron = 25

neuronParams = {
    'v_reset'   : -80.0,  # mV
    'e_rev_I'   : -80.0,  # mV
    'v_rest'    : -30.0,  # mV  # for const-current emulation set to > v_thresh
    'v_thresh'  : -55.0,  # mV
    'g_leak'    :  20.0   # nS  -> tau_mem = 0.2nF / 20nS = 10ms
}

neurons = pynn.Population(popSize, pynn.IF_facets_hardware1, neuronParams)

# the inhibitory projection of the complete population to itself, with
# identical number  of presynaptic neurons. Enable after measuring the
# regular-firing case.
if active_connections:
    pynn.Projection(neurons, neurons, pynn.FixedNumberPreConnector(
                        numInhPerNeuron, weights=weight), target='inhibitory')
else:
    print("Info: Connection inactive, activate by setting active_connections.")

# record spikes
neurons.record()

# record membrane potential of first 4 neurons
pynn.record_v([neurons[0], neurons[1], neurons[2], neurons[3]], '')

# start experiment
pynn.run(runtime)

spikes = neurons.getSpikes()

# end experiment (network keeps running...)
pynn.end()

# retrieve spikes and sort neuron-wise.
snglnrn_spikes = []
snglnrn_spikes_neo = []
for i in range(popSize):
    snglnrn_spikes.append(spikes[np.nonzero(np.equal(i, spikes[:, 0])), 1][0])

# generate raster-plot
for i, spiketrain in enumerate(snglnrn_spikes):
    plt.plot(spiketrain, i*np.ones_like(spiketrain), 'k.', markersize=2)
plt.axis('tight')
plt.xlim(0, runtime)
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('Spike Train Index', fontsize=16)
plt.gca().tick_params(axis='both', which='major', labelsize=14)
plt.savefig('decorr_rasterplot.png')

# calculate ISIs and coefficient of variation (CV)
rate_list = [(np.size(spiketrain) / runtime * 1e3)
                                            for spiketrain in snglnrn_spikes]
isi_list  = [spiketrain[1:]-spiketrain[:-1] for spiketrain in snglnrn_spikes]
mean_list = [np.mean(isis) if len(isis) > 0 else 0. for isis in isi_list]
std_list  = [np.std(isis) if len(isis) > 0 else 0. for isis in isi_list]
cv_list   = [std / mean if mean != 0. else np.nan
                                    for std, mean in zip(std_list, mean_list)]

# rate against cv
plt.clf()
plt.scatter(cv_list, rate_list)
plt.xlabel('CV', fontsize=16)
plt.ylabel('av rate [Hz]', fontsize=16)
plt.gca().tick_params(axis='both', which='major', labelsize=14)
plt.savefig('decorr_rate_over_cv.png')

# plot isi histo:
plt.clf()
plt.hist(np.nan_to_num(isi_list))
plt.xlabel('ISI', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.gca().tick_params(axis='both', which='major', labelsize=14)
plt.savefig('decorr_isi_histo.png')

# plot cv histo:
plt.clf()
plt.hist(np.nan_to_num(cv_list))
plt.xlabel('CV', fontsize=16)
plt.ylabel('count', fontsize=16)
plt.gca().tick_params(axis='both', which='major', labelsize=14)
plt.savefig('decorr_cv_histo.png')


# to get a feeling for the average activity...:
print 'mean firing rate: {:3.1f} Hz'.format(np.nanmean(rate_list))
print 'mean CV: {:3.1f}'.format(np.nanmean(cv_list))

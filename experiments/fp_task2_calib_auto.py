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

neuronParams = {
    'v_reset'   : -80.0,  # mV
    'e_rev_I'   : -75.0,  # mV
    'v_rest'    : -50.0,  # mV
    'v_thresh'  : -55.0,  # mV  - default value.
                          # Change to result of your calculation!
    'g_leak'    :  20.0   # nS  -> tau_mem = 0.2nF / 20nS = 10ms
}


def getFiringRates(gls):
    '''Returns a list of the average firing rate of each neuron

    Input:
        gls     list    value to be inserted for g_leak

    Output
        rates   list    average firing rate for the used neurons
    '''
    if use_other_spikey_half:
        neuron_offset = 192
    else:
        neuron_offset = 0
    pynn.setup(calibTauMem=False, mappingOffset=neuron_offset)

    populations = []
    for gl in gls:
        # add next neuron with appropriate gl
        #
        continue

    [p.record() for p in populations]
    pynn.end()
    # calculate mean firing rate
    #

    return rates


if __name__ == "__main__":
    targetrate = 20.
    if not os.path.exists('gls.dat'):
        gls = np.ones(192) * 20.
    else:
        gls = np.loadtxt('gls.dat')

    rates = getFiringRates(list(gls))
    print(rates)
    gls_new = [targetrate/r*gl for r, gl in zip(rates, gls)]

    np.savetxt('gls.dat', gls_new)

    # plot histogram of used gls and resulting rates
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(rates)
    ax.set_title("Firing rates")
    ax.set_xlabel("Rate [Hz]")
    ax.set_ylabel("Frequency")
    plt.savefig("firing_rates.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(gls_new)
    ax.set_title("Leak conductances")
    ax.set_xlabel("Software value leak conductance")
    ax.set_ylabel("Frequency")
    plt.savefig("leak_conductances.pdf")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Fortgesschrittenenpraktikum F09/10 - Neuromorphic Computing
Task 6 - Decorrelation Network - Plotting

Andreas Baumbach, October 2017, andreas.baumbach@kip.uni-heidelberg.de

'''
import numpy as np

# for plotting without X-server
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa


def borders(points):
    out = np.zeros(len(points) + 1)
    out[1:-1] = (points[1:] + points[:-1]) / 2.
    out[0] = 2 * points[0] - out[1]
    out[-1] = 2 * points[-1] - out[-2]
    return out


def get_data_from_file(filename):
    connections = []
    weights = []
    meanrates = []
    meanCVs = []
    with open(filename, 'r') as f:
        for line in f:
            k, w, r, c = line.strip().split()
            connections.append(int(k))
            weights.append(float(w))
            meanrates.append(float(r))
            meanCVs.append(float(c))

    out_connections = np.array(sorted(set(connections)))
    out_weights     = np.array(sorted(set(weights)))

    plot_connections = borders(out_connections)
    plot_weights     = borders(out_weights)

    rates = -1 * np.ones((len(out_weights), len(out_connections)))
    CVs   = -1 * np.ones((len(out_weights), len(out_connections)))
    for k, w, r, c in zip(connections, weights, meanrates, meanCVs):
        i, = np.where(out_weights == w)
        j, = np.where(out_connections == k)
        rates[i, j] = r
        CVs[i, j]   = c

    rates = np.ma.masked_where(rates == -1, rates)
    CVs = np.ma.masked_where(CVs == -1, CVs)

    return plot_weights, plot_connections, rates, CVs


def main(filename):
    weights, connections, rates, CVs = get_data_from_file(filename)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.pcolor(weights, connections, rates.T)
    fig.colorbar(cax)
    ax.set_title("Mean activity")
    ax.set_xlabel('connection weight [HW units]')
    ax.set_ylabel('connection number [1]')
    plt.savefig('mean_activity.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.pcolor(weights, connections, CVs.T)
    fig.colorbar(cax)
    ax.set_title("CVs")
    ax.set_xlabel('connection weight [HW units]')
    ax.set_ylabel('connection number [1]')
    plt.savefig('CVs.pdf')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        main(filename=sys.argv[1])
    else:
        print("Use as: python fp_task6_plot.py filename\n"
              "  filename should contain a list of your data\n"
              "  format: K w mean_rate mean_CV")

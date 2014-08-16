#!/usr/bin/env python3
"""
Plot the data generates by the main generate script.
"""

import Distribution, DataIO
import argparse, numpy

import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from matplotlib.patches import Ellipse    

from collections import deque
from math import sqrt, pi, atan2

from numpy import linalg
from scipy.stats import chi2

# Global constant
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
gClassColors = {}

def getCovEllipseParams(iCovar, iPerc=0.95):
    """
    Return the ellipse width, height and orientation for the specified 
    covariance matrix. The *iPerc* argument specifies the percentage of 
    the distribution mass that should be covered by the ellipse.

    This function is based on the example posted on Matplotlib mailing-list:
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14153.html
    """
    U, s, _ = linalg.svd(iCovar)
    orient = atan2(U[1, 0], U[0, 0]) * 180.0/pi
    c = chi2.isf(1 - iPerc, iCovar.shape[0])
    width = 2.0 * sqrt(s[0] * c)
    height = 2.0 * sqrt(s[1] * c)

    return (width, height, orient)

def plotDistributions(iDistribs, iSamples, iLabels, iAx):
    """Plot distributions ellipses and most recent samples."""

    # Draw the distribution covariance ellipses
    for lDist in iDistribs:
        lWidth, lHeight, lOrient = getCovEllipseParams(lDist.getCurrentCovar(),
                                                    iPerc=0.95)
        lPatch = Ellipse(xy=lDist.getCurrentCenter(), 
                        width=lWidth, height=lHeight, angle=lOrient, 
                        fc=gClassColors[lDist.getClassLabel()], alpha=0.1) 

        iAx.add_patch(lPatch)

    # Draw the last sampled points
    x = [lSample[0] for lSample in iSamples]
    y = [lSample[1] for lSample in iSamples]
    alpha = 0.9 / (len(iLabels)+1)
    colors = [ColorConverter().to_rgba(COLORS[label], 
            0.1+(i+1)*alpha) for i, label in enumerate(iLabels)]
    iAx.scatter(x, y, c=colors, edgecolors='none')

def main(iArgs):
    """Run main program."""
    
    lFile = DataIO.read(iArgs.datafile)
    if iArgs.distfile == '-':
        lDistribs = Distribution.read(lFile['relation'])
    else:
        lDistribs = Distribution.read(iArgs.distfile)
    
    # Initialize figure and axis before plotting
    lFig = plt.figure(figsize=(10,10))
    lAx1 = lFig.add_subplot(111)
    plt.ion()
    plt.show()

    # allocate deques for plot samples
    if iArgs.nbsamples == -1:
        iArgs.nbsamples = len(lFile['data'])
    lSamples = deque(maxlen=iArgs.nbsamples)
    lLabels = deque(maxlen=iArgs.nbsamples)

    # find min and max over all distributions
    lMin = numpy.ones(lDistribs[0].getDims())*float('inf')
    lMax = numpy.ones(lDistribs[0].getDims())*float('-inf')
    for lDist in lDistribs:
        for (lCenter, lCovar) in zip(lDist._centers, lDist._covars):
            numpy.minimum(lMin, lCenter - 3*numpy.sqrt(numpy.diagonal(lCovar)), lMin)
            numpy.maximum(lMax, lCenter + 3*numpy.sqrt(numpy.diagonal(lCovar)), lMax)

    # enumerate class labels and set colors
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])
    for i, lLabel in enumerate(lClassLabels):
        gClassColors[lLabel] = COLORS[i]

    # create per time step plots
    for i, lDatum in enumerate(lFile['data']):

        # set time for all distributions
        for lDist in lDistribs: lDist.setTime(i)

        # set plot limits, title and legend
        lAx1.clear()
        lAx1.set_xlim(lMin[0],lMax[0])
        lAx1.set_ylim(lMin[1],lMax[1])
        lAx1.set_title("time={}".format(i))
        lPatches = []
        for i in range(len(lClassLabels)):
            lPatch = Ellipse((0,0), 1, 1, fc=COLORS[i])
            lPatches.append(lPatch)
        lAx1.legend(lPatches, lClassLabels)

        # plot distributions and samples
        lLabels.append(lClassLabels.index(lDatum[-1]))
        lSamples.append(lDatum[0:-1])
        plotDistributions(lDistribs, lSamples, lLabels, lAx1)
        lFig.canvas.draw()

        if iArgs.path:
            lFig.savefig(path+"/point_{}.png".format(i))

    plt.ioff()
    plt.show()

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Plot the data from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions.")
    parser.add_argument('--data', dest='datafile', metavar='FILE', default='-',
                        help="name of arff data file (default=stdin)")
    parser.add_argument('--dist', dest='distfile', metavar='FILE', default='-',
                        help="prefix of JSON file containing the mixture of gaussians (default=relation within the data)")
    parser.add_argument('--n', dest='nbsamples', type=int, default=-1,
                        help="number of recent samples to display on each plot")
    parser.add_argument('--save', dest='path', 
                        help='indicates where plot images should be saved')
    
    lArgs = parser.parse_args()

    main(lArgs)

#!/usr/bin/env python3
"""
Plot the data generates by the main generate script.
"""

import Distribution
import argparse, arff, numpy

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

def getCovEllipseParams(iCovar, i, j, iPerc=0.95):
    """
    Return the ellipse width, height and orientation for the specified 
    covariance matrix. The *iPerc* argument specifies the percentage of 
    the distribution mass that should be covered by the ellipse.

    This function is based on the example posted on Matplotlib mailing-list:
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14153.html
    """
    U, s, _ = linalg.svd(iCovar)
    orient = atan2(U[j, i], U[i, i]) * 180.0/pi
    c = chi2.isf(1 - iPerc, s.size)
    width = 2.0 * sqrt(s[i] * c)
    height = 2.0 * sqrt(s[j] * c)

    return (width, height, orient)

def main(iArgs):
    """Run main program."""
    
    # read arff data
    if iArgs.datafile == '-':
        import sys
        lFD = sys.stdin
    else:
        lFD = open(iArgs.datafile+'.arff')
    lFile = arff.load(lFD)

    # read mixture of gaussian file
    if iArgs.distpath[-1] != '/':
        iArgs.distpath += '/'
    lDistribs = Distribution.read(iArgs.distpath+lFile['relation'])
    
    lDims = lDistribs[0].getDims()

    # Initialize figure and axis before plotting
    lFig = plt.figure(figsize=(10,10))
    lPlot = lFig.add_subplot(111)
    plt.ion()
    plt.show()

    # allocate deques for plot samples
    if iArgs.nbsamples == -1:
        iArgs.nbsamples = len(lFile['data'])
    lSamples = deque(maxlen=iArgs.nbsamples)
    lLabels = deque(maxlen=iArgs.nbsamples)

    # check plotting indexes
    lAxes = iArgs.axes
    if lAxes[0] == lAxes[1]:
        print('\nError, dimension indexes are not distinct: {}\n'.format(lAxes))
        exit()
    for i in lAxes:
        if i >= lDims:
            print('\nError, invalid dimension indexes: {}'.format(lAxes))
            print('Indexes should between 0 and {}\n'.format(lDims-1))
            exit()

    # find min and max over all distributions
    lMin = numpy.ones(lDims)*float('inf')
    lMax = numpy.ones(lDims)*float('-inf')
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
    for lStep, lDatum in enumerate(lFile['data']):

        # extract sample and class label
        lSample = lDatum[0:lDims]
        lClass = lDatum[-1]

        # add sample and label
        lLabels.append(lClassLabels.index(lClass))
        lSamples.append(lSample)

        if lStep < iArgs.starttime: continue

        # set time for all distributions
        for lDist in lDistribs: lDist.setTime(lStep)

        # set plot limits, title and legend
        lPlot.clear()
        lDx = lMax[lAxes[0]]-lMin[lAxes[0]]
        lDy = lMax[lAxes[1]]-lMin[lAxes[1]]
        if lDx > lDy:
            lPlot.set_xlim(lMin[lAxes[0]],lMax[lAxes[0]])
            lPlot.set_ylim(lMin[lAxes[1]]-(lDx-lDy)/2,lMax[lAxes[1]]+(lDx-lDy)/2)
        else:
            lPlot.set_xlim(lMin[lAxes[0]]-(lDy-lDx)/2,lMax[lAxes[0]]+(lDy-lDx)/2)
            lPlot.set_ylim(lMin[lAxes[1]],lMax[lAxes[1]]+(lDx-lDy)/2)
        lPlot.set_title("time step = {}".format(lStep))
        lPlot.set_xlabel(lFile['attributes'][lAxes[0]][0])
        lPlot.set_ylabel(lFile['attributes'][lAxes[1]][0])
        lPatches = []
        for i in range(len(lClassLabels)):
            lPatch = Ellipse((0,0), 1, 1, fc=COLORS[i])
            lPatches.append(lPatch)
        lPlot.legend(lPatches, lClassLabels, loc=4)

        # Draw the covariance ellipses
        i = lAxes[0]; j = lAxes[1]
        for lDist in lDistribs:
            lCovar = lDist.getCurrentCovar()
            lWidth, lHeight, lOrient = getCovEllipseParams(lCovar, i, j, iPerc=0.95)
            lCenter = lDist.getCurrentCenter()
            lCenter = [ lCenter[i], lCenter[j] ]
            lPatch = Ellipse(xy=lCenter, width=lWidth, height=lHeight, angle=lOrient, 
                             fc=gClassColors[lDist.getClassLabel()], alpha=0.1) 
            lPlot.add_patch(lPatch)

        # Draw the last sampled points
        x = [lSample[i] for lSample in lSamples]
        y = [lSample[j] for lSample in lSamples]
        alpha = 0.9 / (len(lLabels)+1)
        colors = [ColorConverter().to_rgba(COLORS[label], 
                0.1+(i+1)*alpha) for i, label in enumerate(lLabels)]
        lPlot.scatter(x, y, c=colors, edgecolors='none')

        # update plot
        lFig.canvas.draw()

        if iArgs.savepath:
            if iArgs.savepath[-1] != '/':
                iArgs.savepath += '/'
            lFilename = iArgs.savepath + '{}_{}.png'.format(lFile['relation'], lStep)
            print('Saving to ', lFilename, end='\r')
            lFig.savefig(lFilename)
        else:
            print('Plotting time step', lStep, end='\r')

    plt.ioff(); plt.show()

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Plot the data from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions.")
    parser.add_argument('--data', dest='datafile', metavar='FILE', default='-',
                        help="name of arff data file (default=stdin)")
    parser.add_argument('--path', dest='distpath', metavar='FILE', default='./',
                        help="path to JSON mixture of gaussians file (default=./)")
    parser.add_argument('-n', dest='nbsamples', type=int, default=-1,
                        help="number of recent samples to display on each plot")
    parser.add_argument('--axes', dest='axes', type=int, nargs=2, default=[0, 1],
                        help="plot dimensions indexes")
    parser.add_argument('--start', dest='starttime', metavar='TIME', type=int, default=-1,
                        help="plot only this time step")
    parser.add_argument('--save', dest='savepath', 
                        help='indicates where plot images should be saved')
    
    lArgs = parser.parse_args()

    main(lArgs)

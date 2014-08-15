#!/usr/bin/env python3
"""
Plot the data generates by the main generate script.
"""

from Distribution import Distribution
import argparse, json, numpy

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

gAbort = False

def readData(iFilename, iFormat):
    """
    Read the CSV data file.

    Returns a list of tuples containing a class label and a numpy array.
    """
    try:
        if iFilename == 'stdin':
            import sys
            lFD = sys.stdin
        else:
            lFD = open(iFilename)
    except IOError:
        print('\aError, cannot open file : ', iFilename)
        exit()

    if iFormat == 'csv':
        import csv
        lFile = csv.reader(lFD)
    elif iFormat == 'arff':
        import arff
        lFile = arff.Reader(lFD)
    else: 
        print("\aError, invalid format: ", iFormat)
        exit()

    lData = []
    for lRow in lFile:
        # skip header row
        if list(lRow)[-1] == 'label': continue

        # create tuple (array, label)
        lData.append( (list(lRow)[0:-1], list(lRow)[-1]) )

    return lData

def readDistributions(iFilename):
    """
    Read the JSON description file for the mixture of gaussians.

    Returns a list of distribution objects.
    """
    try:
        lFile = open(iFilename)
    except IOError:
        print('Cannot open file : ', iFilename)
        exit()
        
    n = 1
    lDistribs = []
    for lDist in json.load(lFile):
        lDistribs.append(Distribution(lDist, n))
        n += 1

    if gAbort: exit()

    return lDistribs

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
    
    lDistribs = readDistributions(iArgs.filename)
    lData = readData(iArgs.datafile, iArgs.format)
    
    # Initialize figure and axis before plotting
    lFig = plt.figure(figsize=(10,10))
    lAx1 = lFig.add_subplot(111)
    plt.ion()
    plt.show()

    # allocate deques for plot samples
    if iArgs.nbsamples == -1:
        iArgs.nbsamples = len(lData)
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
    for i, lDatum in enumerate(lData):

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
        lLabels.append(lClassLabels.index(lDatum[1]))
        lSamples.append(lDatum[0])
        plotDistributions(lDistribs, lSamples, lLabels, lAx1)
        lFig.canvas.draw()

        if iArgs.path:
            lFig.savefig(path+"/point_{}.png".format(i))

    plt.ioff()
    plt.show()

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Plot data generated from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions")
    parser.add_argument('filename', 
                        help="name of JSON file containing the mixture of gaussians")
    parser.add_argument('--data', dest='datafile', metavar='FILE', default='stdin',
                        help="name of input data file (default=stdin)")
    parser.add_argument('--n', dest='nbsamples', type=int, default=-1,
                        help="number of recent samples to display on each plot")
    parser.add_argument('--format', dest='format', choices=['csv', 'arff'], 
                        default='csv', help="select input/output format")
    parser.add_argument('--save', dest='path', 
                        help='indicate where the figure should be saved')
    
    lArgs = parser.parse_args()

    main(lArgs)

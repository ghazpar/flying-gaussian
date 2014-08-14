#!/usr/bin/env python3
"""Random points generator where the points are sampled from
labeled distributions that can move, rotate and scale along time.
Each sample correspond to one unit of time.
"""

import argparse, json, random, numpy

from Distribution import Distribution

try:
    import matplotlib.pyplot as plt
except ImportError:
    MATPLOTLIB = False
else:
    MATPLOTLIB = True
    from matplotlib.colors import ColorConverter
    from matplotlib.patches import Ellipse    
    color_conv = ColorConverter()

from collections import deque
from itertools import repeat
from math import sqrt, cos, sin, pi, atan2, exp
from operator import attrgetter, itemgetter, truediv

from numpy import linalg
from scipy.stats import chi2, multivariate_normal

# Global constant
LAST_N_PTS = 100
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

gAbort = False

def readDistributions(iFilename):
    """Read JSON file *iFileName* and return its list of distributions."""
    try:
        file = open(iFilename)
    except IOError:
        print('Cannot open file : ', iFilename)
        exit()
        
    n = 1
    lDistList = []
    for lDist in json.load(file):
        lDistList.append(Distribution(lDist, n))
        n += 1

    if gAbort: exit()

    return lDistList

def drawCovEllipse(iCenter, iCovar, iAx, iPerc=0.95, iColor='b'):
    """Draw the ellipse associated with the multivariate normal distribution
    defined by *centroid* and *cov_matrix*. The *perc* argument specified 
    the percentage of the distribution mass that will be drawn.

    This function is based on the example posted on Matplotlib mailing-list:
    http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg14153.html
    """
    U, s, _ = linalg.svd(iCovar)
    orient = atan2(U[1, 0], U[0, 0]) * 180.0/pi

    c = chi2.isf(1 - iPerc, len(iCenter))
    width = 2.0 * sqrt(s[0] * c)
    height = 2.0 * sqrt(s[1] * c)

    ellipse = Ellipse(xy=iCenter, width=width, height=height, 
                      angle=orient, fc=iColor, alpha=0.1)
    
    return iAx.add_patch(ellipse)

def plotDistributions(iClassLabels, iDistribs, iSamples, iLabels, iAx):
    """Plot the distributions ellipses and the last sampled points."""

    lLegend = {}
    # Draw the distribution covariance ellipse
    for lDist in iDistribs:
        lLabel = lDist.getClassLabel()
        i = iClassLabels.index(lLabel)
        ref_ell = drawCovEllipse(lDist.getCurrentCenter(), lDist.getCurrentCovar(),
                                 iPerc=0.95, iAx=iAx, iColor=COLORS[i])
        lLegend[lLabel] = ref_ell

    # Draw the last sampled points
    x = list(map(itemgetter(0), iSamples))
    y = list(map(itemgetter(1), iSamples))
    alph_inc = 1.0 / len(iLabels)
    colors = [color_conv.to_rgba(COLORS[iClassLabels.index(label)], 
                (i+1)*alph_inc) for i, label in enumerate(iLabels)]
    iAx.scatter(x, y, c=colors, edgecolors='none')

    iAx.legend(lLegend.values(), lLegend.keys())

def main(iFilename, iNbSamples, iPlot, iPath, iSeed=None):
    random.seed(iSeed)
    numpy.random.seed(iSeed)
    
    lSave = iSeed is not None

    if (iPlot or lSave) and not MATPLOTLIB:
        print('Warning: the --plot or --save-fig options were activated,'\
              'but matplotlib is unavailable. ' \
              'Processing will continue without plotting.')
    
    lDistribs = readDistributions(iFilename)
    
    # Initialize figure and axis before plotting
    if (iPlot or lSave) and MATPLOTLIB:
        lFig = plt.figure(figsize=(10,10))
        lAx1 = lFig.add_subplot(111)
        if iPlot:
            plt.ion()
            plt.show()
        lSamples = deque(maxlen=LAST_N_PTS)
        lLabels = deque(maxlen=LAST_N_PTS)

        # find min and max over all distributions
        lMin = numpy.ones(lDistribs[0].getDims())*float('inf')
        lMax = numpy.ones(lDistribs[0].getDims())*float('-inf')
        for lDist in lDistribs:
            for (lCenter, lCovar) in zip(lDist._centers, lDist._covars):
                numpy.minimum(lMin, lCenter - 3*numpy.sqrt(numpy.diagonal(lCovar)), lMin)
                numpy.maximum(lMax, lCenter + 3*numpy.sqrt(numpy.diagonal(lCovar)), lMax)

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # Print CSV header
    lDims = lDistribs[0].getDims()
    print("label,", 
          ", ".join("x{}".format(i+1) for i in range(lDims)), ",",
          ", ".join("P({})".format(x) for x in lClassLabels))

    # generate the requested samples
    for i in range(iNbSamples):

        # set time for all distributions
        for _ in map(lambda x: x.setTime(i), lDistribs): pass

        # randomly select a distribution according to weights
        lWeights = [lDist.getCurrentWeight() for lDist in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)
        lSelDist = numpy.random.choice(lDistribs, p=lProbs)

        # draw a random sample from selected distribution
        lSample = multivariate_normal.rvs(lSelDist.getCurrentCenter(), 
                                          lSelDist.getCurrentCovar())

        # compute per class conditional probabilities
        lSums = {}
        # initialize sums
        for lLabel in lClassLabels:
            lSums[lLabel] = 0
        # compute per class sums
        for (lProb, lDist) in zip(lProbs, lDistribs):
            lPDF = multivariate_normal.pdf(lSample, lDist.getCurrentCenter(), 
                                                    lDist.getCurrentCovar())
            lSums[lDist.getClassLabel()] += lProb * lPDF
        # compute total sum
        lTotalSum = sum(lSums.values())

        # print data in CSV format
        print("{},".format(lSelDist.getClassLabel()),
              ", ".join("{}".format(x) for x in lSample), ",",
              ", ".join("{}".format(lSums[x]/lTotalSum) for x in lClassLabels))

        # plot distributions
        if (iPlot or lSave) and MATPLOTLIB:
            lSamples.append(lSample)
            lLabels.append(lSelDist.getClassLabel())

            # clear plot
            lAx1.clear()

            # set plot limits and title
            lAx1.set_xlim(lMin[0],lMax[0])
            lAx1.set_ylim(lMin[1],lMax[1])
            lAx1.set_title("time={}".format(i))

            plotDistributions(lClassLabels, lDistribs, lSamples, lLabels, lAx1)
            lFig.canvas.draw()

            if lSave:
                fig.savefig(path+"/point_{}.png".format(i))

    if iPlot and MATPLOTLIB:
        plt.ioff()
        plt.show()

    return lDistribs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file of classes'\
            'and return a series of randomly sampled points from those '\
            'classes.')
    parser.add_argument('filename', help='json file containing the classes')
    parser.add_argument('samples', type=int, help='number of samples')
    
    parser.add_argument('--plot', dest='plot', required=False, 
                        action='store_true', default=False,
                        help='tell if the results should be plotted')
    parser.add_argument('--save-fig', dest='save_path', required=False, 
                        metavar='PATH', 
                        help='indicate where the figure should be saved')
    parser.add_argument('--seed', type=int, default=None, required=False, 
                        metavar='SEED',
                        help='random number generator seed')
    
    args = parser.parse_args()

    main(args.filename, args.samples, args.plot, args.save_path, args.seed)

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

def plotDistributions(iTime, iRefLabels, iDistribs, iSamples, iLabels, iFig, iAxis):
    """Plot the distributions ellipses and the last sampled points."""
    iAxis.clear()

    # find min and max over all distributions
    lMin = numpy.ones(iDistribs[0].getDims())*float('inf')
    lMax = numpy.ones(iDistribs[0].getDims())*float('-inf')
    for lDist in iDistribs:
        for (lCenter, lCovar) in zip(lDist._centers, lDist._covars):
            numpy.minimum(lMin, lCenter - 3*numpy.sqrt(numpy.diagonal(lCovar)), lMin)
            numpy.maximum(lMax, lCenter + 3*numpy.sqrt(numpy.diagonal(lCovar)), lMax)

    # set limits
    iAxis.set_xlim(lMin[0],lMax[0])
    iAxis.set_ylim(lMin[1],lMax[1])

    iAxis.set_title("time={}".format(iTime))

    # Draw the last sampled points
    x = list(map(itemgetter(0), iSamples))
    y = list(map(itemgetter(1), iSamples))
    alph_inc = 1.0 / len(iLabels)
    colors = [color_conv.to_rgba(COLORS[iRefLabels.index(label)], 
                (i+1)*alph_inc) for i, label in enumerate(iLabels)]
    iAxis.scatter(x, y, c=colors, edgecolors='none')

    ellipses = []
    labels = []
    # Draw the distribution covariance ellipse
    for lDist in iDistribs:
        i = iRefLabels.index(lDist.getClassLabel())
        ref_ell = drawCovEllipse(lDist.getCurrentCenter(), lDist.getCurrentCovar(),
                                 iPerc=0.95, iAx=iAxis, iColor=COLORS[i])
        ellipses.append(ref_ell)
        labels.append(lDist.getClassLabel())

    iAxis.legend(ellipses, labels)
    iFig.canvas.draw()

def selectDistribution(iDistList, iTime):
    """Randomly select a distribution from sequence *iDistList*. 
    The selection process is biaised by the weight of each distribution
    at time *iTime*. Returns the selected distribution."""


    lWeights = [x.getDistribParams(iTime) for x in iDistributions]
    lWeights = numpy.array(lWeights) / sum(lWeights)
    return numpy.random.choice(iDistributions, p=lWeights)

def main(iFilename, iSamples, iPlot, iPath, iSeed=None):
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
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111)
        if iPlot:
            plt.ion()
            plt.show()
        points = deque(maxlen=LAST_N_PTS)
        labels = deque(maxlen=LAST_N_PTS)
        ref_labels = list(map(attrgetter('_label'), lDistribs))

    # enumerate class labels
    lLabels = set(x.getClassLabel() for x in lDistribs)

    # Print CSV header
    lDims = lDistribs[0].getDims()
    print("label,", 
          ", ".join("x{}".format(i+1) for i in range(lDims)), ",",
          ", ".join("P({})".format(x) for x in lLabels))

#    for lDist in lDistribs:
#        print(lDist._centers)
#        print(lDist._covars)

    # generate samples
    for i in range(iSamples):

        # determine current weight, center and covariance parameters
        # for all distributions
        for x in lDistribs:
            x.setTime(i)

        # randomly select a distribution according to weights
        lWeights = [x.getCurrentWeight() for x in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)
        lSelectDist = numpy.random.choice(lDistribs, p=lProbs)

        # draw sample from selected distribution
        lSample = multivariate_normal.rvs(lSelectDist.getCurrentCenter(), 
                                          lSelectDist.getCurrentCovar())

        # compute per class conditional probabilities
        lSums = {}
        # initialize sums
        for x in lLabels:
            lSums[x] = 0
        # compute per class sums
        for (lProb, lDist) in zip(lProbs, lDistribs):
            lPDF = multivariate_normal.pdf(lSample, lDist.getCurrentCenter(), 
                                                    lDist.getCurrentCovar())
            lSums[lDist.getClassLabel()] += lProb * lPDF
        # compute total sum
        lTotalSum = sum(lSums.values())

        # Print the sampled point in CSV format
        print("{},".format(lSelectDist.getClassLabel()),
              ", ".join("{}".format(x) for x in lSample), ",",
              ", ".join("{}".format(lSums[x]/lTotalSum) for x in lLabels))
# print(lSelectDist.getCurrentCenter())
# print(lSelectDist.getCurrentCovar())     

        # Plot the resulting distribution if required
        if (iPlot or lSave) and MATPLOTLIB:
            points.append(lSample)
            labels.append(lSelectDist.getClassLabel())
            plotDistributions(i, ref_labels, lDistribs, points, labels, fig, ax1)
            if lSave:
                fig.savefig(path+'/point_%i.png' % i)

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

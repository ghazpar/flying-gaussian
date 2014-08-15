#!/usr/bin/env python3
"""
Classify the data generates by the main generate script.
"""

from scipy.stats import multivariate_normal
from Distribution import Distribution
from DataIO import readData, writeData
import argparse, json, numpy

# Global constant
gAbort = False

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

def main(iArgs):
    """Run main program."""
    
    lDistribs = readDistributions(iArgs.filename)
    lData = readData(iArgs.datafile, iArgs.format)
    lDims = lDistribs[0].getDims()

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # classify each data sample
    for i, lDatum in enumerate(lData):

        # set time for all distributions
        for lDist in lDistribs: lDist.setTime(i)

        # compute per class conditional probabilities
        lSums = {}
        # initialize sums
        for x in lClassLabels: lSums[x] = 0
        # compute a priori probabilities
        lWeights = [lDist.getCurrentWeight() for lDist in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)

        # compute per class sums
        for (lProb, lDist) in zip(lProbs, lDistribs):
            lPDF = multivariate_normal.pdf(lDatum[0], lDist.getCurrentCenter(), 
                                            lDist.getCurrentCovar())
            lSums[lDist.getClassLabel()] += lProb * lPDF
        # compute total sum
        lTotalSum = sum(lSums.values())

        lData[i] = lDatum[0:-1]+[lSums[x]/lTotalSum for x in lClassLabels]+lDatum[-1:]

    # write output data file
    lHeader = {}
    lHeader['filename'] = iArgs.filename
    lHeader['attrs'] = [('x{}'.format(x), 'numeric') for x in range(lDims)]
    lHeader['attrs'] += [('P({}|X)'.format(x), 'numeric') for x in lClassLabels]
    lHeader['attrs'] += [('label', '{'+','.join(lClassLabels)+'}')]
    writeData(lHeader, lData, iArgs.format)

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Classify data generated from "
                                                 "a mixture of non-stationary "
                                                 "gaussian distributions")
    parser.add_argument('filename', 
                        help="name of JSON file containing the mixture of gaussians")
    parser.add_argument('--data',  dest='datafile', metavar='FILE', default='stdin',
                        help="name of input data file (default=stdin)")
    parser.add_argument('--format', dest='format', choices=['arff', 'csv'], 
                        default='arff', help="select input/output format (default=arff)")
    
    lArgs = parser.parse_args()

    main(lArgs)

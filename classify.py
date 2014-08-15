#!/usr/bin/env python3
"""
Classify the data generates by the main generate script.
"""

from scipy.stats import multivariate_normal
from Distribution import Distribution
import argparse, json, csv, sys, numpy

from math import sqrt, cos, sin, pi, atan2

# Global constant
gAbort = False

def readData(iFilename, iFormat):
    """
    Read the data file.

    Returns a list of tuples containing a numpy array and.
    """
    try:
        if iFilename == 'stdin':
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
        lFile = arff.reader(lFD)
    else: 
        print("\aError, invalid format: ", iFormat)
        exit()

    lData = []
    for lRow in lFile:
        # skip header row
        if lRow[-1] == 'label': continue
        print(lRow)

        # create tuple (array, label)
        lData.append( (numpy.array(lRow[0:-1]), lRow[-1]))

    print(lData)
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

def main(iArgs):
    """Run main program."""
    
    lDistribs = readDistributions(iArgs.filename)
    lData = readData(iArgs.datafile, iArgs.format)

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # Print CSV header
    lDims = lDistribs[0].getDims()
    print(''.join('x{},'.format(i+1) for i in range(lDims)),
          ''.join('P({}|X),'.format(x) for x in lClassLabels),
          'class', sep='')

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

        # print data in CSV format
        print(''.join('{},'.format(x) for x in lDatum[0]),
              ''.join('{},'.format(lSums[x]/lTotalSum) for x in lClassLabels),
              lDatum[1], sep='')

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Classify data generated from "
                                                 "a mixture of non-stationary "
                                                 "gaussian distributions")
    parser.add_argument('filename', 
                        help="name of JSON file containing the mixture of gaussians")
    parser.add_argument('--data',  dest='datafile', metavar='FILE', default='stdin',
                        help="name of input data file (default=stdin)")
    parser.add_argument('--format', dest='format', choices=['csv', 'arff'], 
                        default='csv', help="select input/output format")
    
    lArgs = parser.parse_args()

    main(lArgs)

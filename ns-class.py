#!/usr/bin/env python3
"""
Classify the data generates by the main generate script.
"""

from scipy.stats import multivariate_normal
import Distribution, DataIO
import argparse, numpy

def main(iArgs):
    """Run main program."""
    
    lFile = DataIO.read(iArgs.datafile)
    if iArgs.distfile == '-':
        lDistribs = Distribution.read(lFile['relation'])
    else:
        lDistribs = Distribution.read(iArgs.distfile)

    lDims = lDistribs[0].getDims()

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # classify each data sample
    for lStep, lDatum in enumerate(lFile['data']):

        # extract sample
        lSample = lDatum[0:lDims]
        lClass = lDatum[-1]

        # set time for all distributions
        for lDist in lDistribs: lDist.setTime(lStep)

        # compute per class conditional probabilities
        lSums = {}
        # initialize sums
        for x in lClassLabels: lSums[x] = 0
        # compute a priori probabilities
        lWeights = [lDist.getCurrentWeight() for lDist in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)

        # compute per class sums
        for (lProb, lDist) in zip(lProbs, lDistribs):
            lPDF = multivariate_normal.pdf(lSample, lDist.getCurrentCenter(), 
                                           lDist.getCurrentCovar())
            lSums[lDist.getClassLabel()] += lProb * lPDF
        # compute total sum
        lTotalSum = sum(lSums.values())

        lFile['data'][lStep] = lDatum[0:lDims]+[lSums[x]/lTotalSum for x in lClassLabels]+lDatum[-1:]

    # write output arff data
    lAttrs = lFile['attributes']
    lFile['attributes'] = lAttrs[0:lDims] + [('P({}|X)'.format(x), 'REAL') for x in lClassLabels] + lAttrs[-1:]
    DataIO.write(lFile['relation'], lFile['attributes'], lFile['data'])

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Classify the data from "
                                                 "a mixture of non-stationary "
                                                 "gaussian distributions")
    parser.add_argument('--data', dest='datafile', metavar='FILE', default='-',
                        help="name of arff data file (default=stdin)")
    parser.add_argument('--dist', dest='distfile', metavar='FILE', default='-',
                        help="prefix of JSON file containing the mixture of gaussians (default=relation within the data)")
    
    lArgs = parser.parse_args()

    main(lArgs)

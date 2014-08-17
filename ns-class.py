#!/usr/bin/env python3
"""
Classify the data generates by the main generate script.
"""

from scipy.stats import multivariate_normal
import Distribution
import argparse, arff, numpy

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
    if iArgs.path[-1] != '/':
        iArgs.path += '/'
    lDistribs = Distribution.read(iArgs.path+lFile['relation'])

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
    print(arff.dumps(lFile))

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Classify the data from "
                                                 "a mixture of non-stationary "
                                                 "gaussian distributions")
    parser.add_argument('--data', dest='datafile', metavar='FILE', default='-',
                        help="name of arff data file (default=stdin)")
    parser.add_argument('--path', dest='path', metavar='FILE', default='.',
                        help="path to JSON mixture of gaussians file (default=./)")
    
    lArgs = parser.parse_args()

    main(lArgs)

#!/usr/bin/env python3
"""
Generate data for a mixture of non-stationary gaussian distributions.
See file test.json for an example.
"""

from scipy.stats import multivariate_normal
import argparse, sys, json, arff, numpy, datetime, random
from Distribution import Distribution

def read(iFilename):
    """Read the JSON description file for the mixture of gaussians."""

    return lDistribs

def main(iArgs):
    """Run main program."""

    # seed random generators
    if iArgs.seed == None:
        iArgs.seed = int(random.random()*10e15)
    numpy.random.seed(iArgs.seed)
   
    # read dataset file in json format
    try:
        lFD = open(iArgs.filename+'.json')
        iArgs.filename += '.json'
    except:
        lFD = open(iArgs.filename)
    lInput = json.load(lFD)

    # parse distributions
    n = 1
    lDistribs = []
    for lDist in lInput['distributions']:
        lDistribs.append(Distribution(lDist, lInput['dimensions'], n))
        n += 1

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # build arff object
    lOutput = {}
    lOutput['description'] = 'This file was generated {}\n'.format(datetime.datetime.now())
    lOutput['description'] += "with the 'ns-gen.py' script\n"
    lOutput['description'] += 'using random seed: {}\n'.format(iArgs.seed)
    lOutput['description'] += 'and configuration file:\n'
    lOutput['description'] += ''.join(open(iArgs.filename).readlines())
    lOutput['relation'] = lInput['filename']
    lOutput['attributes'] = [('x{}'.format(x), 'REAL') for x in range(lInput['dimensions'])]
    lOutput['attributes'] += [('P({}|X)'.format(x), 'REAL') for x in lClassLabels]
    lOutput['attributes'] += [('class', [x for x in lClassLabels])]
    lOutput['data'] = []

    # generate the requested samples
    for lStep in range(lInput['duration']):

        # set time for all distributions
        for lDist in lDistribs: lDist.setTime(lStep)

        # randomly select a distribution according to weights
        lWeights = [lDist.getCurrentWeight() for lDist in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)
        lSelDist = numpy.random.choice(lDistribs, p=lProbs)

        # draw a random sample from selected distribution
        lCenter = lSelDist.getCurrentCenter()
        lCovar = lSelDist.getCurrentCovar()
        assert lSelDist.getCurrentWeight() != 0
        assert lCenter != None and lCovar != None
        lSample = multivariate_normal.rvs(lCenter, lCovar)

        # compute per class conditional probabilities
        lSums = {}
        # initialize sums
        for x in lClassLabels: 
            lSums[x] = 0
        # compute a priori probabilities
        lWeights = [x.getCurrentWeight() for x in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)

        # compute per class sums
        for (lProb, lDist) in zip(lProbs, lDistribs):
            # skip null distributions
            if lProb == 0: continue

            # compute pdf
            lCenter = lDist.getCurrentCenter()
            lCovar = lDist.getCurrentCovar()
            assert lCenter != None and lCovar != None
            lPDF = multivariate_normal.pdf(lSample, lCenter, lCovar)

            lSums[lDist.getClassLabel()] += lProb * lPDF
        # compute total sum
        lTotalSum = sum(lSums.values())

        lOutput['data'].append(list(lSample)+[lSums[x]/lTotalSum for x in lClassLabels]+[lSelDist.getClassLabel()])

    # write output arff data
    print(arff.dumps(lOutput))

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Generate data from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions.")
    parser.add_argument('filename', 
                        help="path to JSON mixture of gaussians file (without .json extension")
    parser.add_argument('--seed', type=int, metavar='INT', default = None,
                        help="seed for random number generators (default=None)")
        
    lArgs = parser.parse_args()

    main(lArgs)

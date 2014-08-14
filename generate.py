#!/usr/bin/env python3
"""
Generate data for a mixture of non-stationary gaussian distributions.
See file test.json for an example.
"""

from scipy.stats import multivariate_normal
from Distribution import Distribution
import argparse, json, numpy

gAbort = False

def readDistributions(iFilename):
    """Read the JSON description file for the mixture of gaussians."""
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
    
    # read distributions in json file
    lDistribs = readDistributions(iArgs.filename)

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # Print CSV header
    lDims = lDistribs[0].getDims()
    print("#class,", 
          ", ".join("x{}".format(i+1) for i in range(lDims)))

    # generate the requested samples
    for i in range(iArgs.nbsamples):

        # set time for all distributions
        for lDist in lDistribs: lDist.setTime(i)

        # randomly select a distribution according to weights
        lWeights = [lDist.getCurrentWeight() for lDist in lDistribs]
        lProbs = numpy.array(lWeights) / sum(lWeights)
        lSelDist = numpy.random.choice(lDistribs, p=lProbs)

        # draw a random sample from selected distribution
        lSample = multivariate_normal.rvs(lSelDist.getCurrentCenter(), 
                                          lSelDist.getCurrentCovar())

        # print data in CSV format
        print("{},".format(lSelDist.getClassLabel()),
              ", ".join("{}".format(x) for x in lSample))

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Generate data from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions")
    parser.add_argument('filename', help="JSON file containing the mixture "
                                          "of gaussians")
    parser.add_argument('nbsamples', type=int,
                        help="number of samples to generate")
        
    lArgs = parser.parse_args()

    main(lArgs)

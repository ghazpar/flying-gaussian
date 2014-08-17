#!/usr/bin/env python3
"""
Generate data for a mixture of non-stationary gaussian distributions.
See file test.json for an example.
"""

from scipy.stats import multivariate_normal
import Distribution, DataIO
import argparse, json, arff, numpy

def main(iArgs):
    """Run main program."""
    
    # read distributions in json file
    lDistribs = Distribution.read(iArgs.filename)
    lDims = lDistribs[0].getDims()

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    # build arff object
    lFile = {}
    lFile['description'] = "\nFlying non-stationary gaussians, 2014\n"
    for line in open(iArgs.filename+'.json'):
        lFile['description'] += line
    lFile['relation'] = iArgs.filename
    lFile['attributes'] = [('x{}'.format(x), 'REAL') for x in range(lDims)]
    lFile['attributes'] += [('class', [x for x in lClassLabels])]
    lFile['data'] = []

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

        lFile['data'].append(list(lSample)+[lSelDist.getClassLabel()])

    # write output arff data
    print(arff.dumps(lFile))

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Generate data from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions.")
    parser.add_argument('filename', 
                        help="path to JSON mixture of gaussians file (without .json extension")
    parser.add_argument('nbsamples', type=int,
                        help="number of samples to output")
        
    lArgs = parser.parse_args()

    main(lArgs)

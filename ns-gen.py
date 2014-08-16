#!/usr/bin/env python3
"""
Generate data for a mixture of non-stationary gaussian distributions.
See file test.json for an example.
"""

from scipy.stats import multivariate_normal
import Distribution
import argparse, arff, numpy

def main(iArgs):
    """Run main program."""
    
    # read distributions in json file
    lDistribs = Distribution.read(iArgs.filename)
    lDims = lDistribs[0].getDims()

    # enumerate class labels
    lClassLabels = set(x.getClassLabel() for x in lDistribs)
    lClassLabels = sorted([x for x in lClassLabels])

    lData = []
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

        lData.append(list(lSample)+[lSelDist.getClassLabel()])

    # write output arff data file
    lInfo = {}
    lInfo['description'] = "\nFlying non-stationary gaussians, 2014\n"
    lInfo['relation'] = iArgs.filename
    lInfo['attributes'] = [('x{}'.format(x), 'REAL') for x in range(lDims)]
    lInfo['attributes'] += [('class', [x for x in lClassLabels])]
    lInfo['data'] = lData
    print(arff.dumps(lInfo))

if __name__ == "__main__":

    # parse command line
    parser = argparse.ArgumentParser(description="Generate data from a "
                                                 "mixture of non-stationary "
                                                 "gaussian distributions.")
    parser.add_argument('filename', 
                        help="name of JSON file containing the mixture of gaussians")
    parser.add_argument('nbsamples', type=int,
                        help="number of samples to output")
        
    lArgs = parser.parse_args()

    main(lArgs)

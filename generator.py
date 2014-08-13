#!/usr/bin/env python3
"""Random points generator where the points are sampled from
labeled distributions that can move, rotate and scale along time.
Each sample correspond to one unit of time.
"""

import argparse
import json
import random


import numpy

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
from numpy.random import multivariate_normal
from scipy.stats import chi2

# Global constant
LAST_N_PTS = 25
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

gAbort = False

class Distribution(object):
    """ This class incapsulates a non-stationary multivariate normal distribution. 
    The object contains a list of phases that are applied sequentially, and that 
    can modify the scale, the position and the angle of the distribution.
    """
    def __init__(self, iArgs, iN):
        """ Parse file content *iArgs*.
        """

        self.label = iArgs.get("label")
        if self.label == None:
            print("Error, distribution {} is missing a `label' attribute".format(iN))
            gAbort = True

        lWeight = iArgs.get("weight")
        if lWeight == None:
            print("Error, distribution {} is missing a `weight' attribute".format(iN))
            gAbort = True
        if lWeight != None and lWeight < 0:
            print("Error, distribution {} has a negative `weight' attribute".format(iN))
            gAbort = True
        self.weights = [lWeight]

        lCenter = iArgs.get("center")
        if lCenter == None:
            print("Error, distribution {} is missing a `center' attribute".format(iN))
            gAbort = True
        self.centers = [numpy.array(lCenter)]
        self.dims = len(lCenter)

        lCovar = iArgs.get("covar")
        if lCovar == None:
            print("Error, distribution {} is missing a `covar' attribute".format(iN))
            gAbort = True
        else:
            lCovar = numpy.array(lCovar)
            if lCovar.shape != (self.dims, self.dims):
                print("Error, distribution {}".format(iN) +
                      " has an invalid covariance matrix dimensions")
                gAbort = True
        self.covars = [lCovar]
        self.rotations = [None]
        self.scales = [None]

        lStart = iArgs.get("start", 0)
        if lStart < 0:
            print("Error, distribution {}".format(iN) +
                  " has a negative `start' attribute")
            gAbort = True
        self.indices = [lStart]

        for lPhase in iArgs["phases"]:

            lDuration = lPhase.get("duration")
            if lDuration == None:
                print("Error, a phase in distribution {}".format(iN) +
                      " is missing a `duration' attribute")
                gAbort = True
            elif lDuration < 0:
                print("Error, a phase in distribution {}".format(iN) +
                      " has a negative `duration' attribute")
                gAbort = True
            self.indices.append(self.indices[-1]+lDuration)

            lWeight = lPhase.get("weight")
            if lWeight == None:
                self.weights.append(self.weights[-1])
            elif lWeight < 0:
                print("Error, a phase in distribution {}".format(iN) +
                      " has a negative `weight' attribute")
                gAbort = True
            else:
                self.weights.append(lWeight)

            lMoveto = lPhase.get("moveto")
            if lMoveto == None:
                lMoveto = lPhase.get("rmoveto")
                if lMoveto == None:
                    self.centers.append(self.centers[-1])
                else:
                    self.centers.append(self.centers[-1]+lMoveto)
            else:
                if lPhase.get("rmoveto"):
                    print("Error, a phase in distribution {}".format(i) +
                          "contains both a `moveto' and `rmoveto' attribute")
                    gAbort = True
                self.centers.append(lMoveto)
           
            lScale = lPhase.get("scale", None)
            if lScale != None and lScale < 0:
                print("Error, a phase in distribution {}".format(iN) +
                      " has a negative `scale' attribute")
                gAbort = True
            self.scales.append(lScale)

            lRotation = lPhase.get("rotate", None)
            if lRotation != None and len(lRotation) != self.dims*(self.dims-1)//2:
                print("Error, a phase in distribution {}".format(iN) +
                      " has an invalid number of rotation angles")
            self.rotations.append(lRotation)

            lCovar = numpy.copy(self.covars[-1])
            if lScale != None:
                lCovar *= lScale
            if lRotation != None:
                lMatrix = createRotationMatrix(self.dims, lRotation)
                lCovar = numpy.dot(lMatrix.T, numpy.dot(lCovar, lMatrix))
            self.covars.append(lCovar)

            lValidPhaseArgs = ("duration", "rotate", "scale", "weight", "moveto", "rmoveto")
            for lArg in lPhase:
                if not lArg in lValidPhaseArgs:
                    print("Error, unknown `{}' phase attribute in distribution {}".format(lArg, iN))
                    gAbort = True

        lValidDistribArgs = ("label", "weight", "center", "covar", "start", "phases")
        for lArg in iArgs:
            if not lArg in lValidDistribArgs:
                print("Error, unknown `{}' attribute in distribution {}".format(lArg, iN))
                gAbort = True

    def __repr__(self):
        rep = "{}, [".format(self.label)
        rep += "indices={}, weights={}, centers={}, covars={}, scales={}, rotations={}]".format(
               self.indices, self.weights, self.centers, self.covars, self.scales, self.rotations)
        return rep
        
    def getDistParams(self, iTime):
        """ Compute the distribution parameters at time *iTime*;
        Return tuple(center, covariance).
        """
        if self.getWeight(iTime) <= 0:
            return None, None

        i = list(filter(lambda x: self.indices[x]<iTime, self.indices))[-1]
        x = (iTime-self.indices[i])/(self.indices[i+1]-self.indices[i])

        lCenter = self.centers[i] + x*(self.centers[i+1]-self.centers[i])

        lCovar = numpy.copy(self.covars[i])
        if self.scales[i] != None:
            lScale = 1.0 + x*(self.scales[i]-1.0)
            lMatrix *= lScale
        if self.rotations[i] != None:
            lRotation = numpy.copy(self.rotations[i])
            lRotation *= x
            lMatrix = createRotationMatrix(self.dims, lRotation)
            lCovar = numpy.dot(lMatrix.T, numpy.dot(lCovar, lMatrix))
        return lCenter, lCovar


    def getWeight(self, iTime):
        i = list(filter(lambda x: self.indices[x]<iTime, self.indices))[-1]
        if i == len(self.indices)-1:
            return self.weights[i]
        else:
            x = (iTime-self.indices[i])/(self.indices[i+1]-self.indices[i])
            return self.weight[i] + x*(self.weights[i+1]-self.weight[i])

    def pdf(self, iPoint, iTime):
        """ Compute the probability to sample a given point at a given time."""
        if self.getWeight(iTime) <= 0: 
            return 0.

        lCenter, lCovar = self.getDistParams(iTime)
        lX = (iPoint - lCenter)
        lNum = exp(-0.5 * numpy.dot(lX.T, numpy.dot(linalg.inv(lCovar), lX)))
        lDenom = sqrt(2*pi*linalg.det(lCovar))

        return lNum/lDenom
       
    def sample(self, iTime, iN=1):
        """ Sample the distribution at the specified time."""
        if self.getWeight(iTime): 
            return None

        lCenter, lCovar = self.getDistParams(iTime)
        return multivariate_normal(lCenter, lCovar, iN)

def createRotationMatrix(iDim, iAngles):
    if len(iAngles) != iDim*(iDim-1)/2:
        raise runtime_error("invalid number of angles")
    lMatrix = numpy.identity(iDim)
    k = 0
    for i in range(iDim-1):
        for j in range(i+1, iDim):
            lTmp = numpy.identity(iDim)
            lAngle = iAngles[k]
            lSign = 1.0
            if i+j % 2 == 0:
                lSign = -1.0
            lTmp[i][i] = cos(lAngle)
            lTmp[j][j] = cos(lAngle)
            lTmp[i][j] = -sign*sin(lAngle)
            lTmp[j][i] = sign*sin(lAngle)
            lMatrix = numpy.dot(lMatrix, lTmp)
            k += 1
    return lMatrix

def read_file(iFilename):
    """Read a JSON file containing distributions and transformations, 
    and initialize the corresponding object. The function 
    return a list of initialized distributions.
    """
    try:
        file = open(iFilename)
    except IOError:
        print('Cannot open file : ', iFilename)
        exit()
        
    lDistributions = []

    n = 1
    for lDist in json.load(file):
        lDistributions.append(Distribution(lDist, n))
        n += 1

    if gAbort: exit()

    return lDistributions

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
                      angle=orient, fc=color, alpha=0.1)
    
    return ax.add_patch(ellipse)

def plotDistributions(iTime, iRefLabels, iDistList, iPoints, iLabels, iFig, iAxis):
    """Plot the distributions ellipses and the last sampled points.
    """
    iAxis.clear()
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for lDistrib in iDistList:
        if lDistrib.start <= time:
            min_x = min(min_x, lDistrib.center[0] - 4*lDistrib.scale*sqrt(lDistrib.covar[0][0]))
            max_x = max(max_x, lDistrib.center[0] + 4*lDistrib.scale*sqrt(lDistrib.covar[0][0]))
            min_y = min(min_y, lDistrib.center[1] - 4*lDistrib.scale*sqrt(lDistrib.covar[1][1]))
            max_y = max(max_y, lDistrib.center[1] + 4*lDistrib.scale*sqrt(lDistrib.covar[1][1]))
    
    iAxis.set_xlim(min_x,max_x)
    iAxis.set_ylim(min_y,max_y)

    # Draw the last sampled points
    x = list(map(itemgetter(0), iPoints))
    y = list(map(itemgetter(1), iPoints))
    alph_inc = 1.0 / len(labels)
    colors = [color_conv.to_rgba(COLORS[ref_labels.index(label)], 
                (i+1)*alph_inc) for i, label in enumerate(labels)]
    axis.scatter(x, y, c=colors, edgecolors='none')

    ellipses = []
    labels = []
    # Draw the distribution covariance ellipse
    for i, class_ in enumerate(class_list):
        present = False
        for distrib in class_.distributions:
            if time >= distrib.start_time:
                ref_ell = drawCovEllipse(distrib.centroid, distrib.matrix * distrib.scale,
                                           iPerc=0.95, iAx=axis, iColor=COLORS[i])
                if not present:
                    ellipses.append(ref_ell)
                    labels.append(class_.label)
                    present = True

    axis.legend(ellipses, labels)
    fig.canvas.draw()

def weight_choice(seq):
    """Randomly choose an element from the sequence *seq* with a 
    bias function of the weight of each element.
    """
    sorted_seq = sorted(seq, key=attrgetter("weight"), reverse=True)
    sum_weights = sum(elem.weight for elem in seq)
    u = random.random() * sum_weights
    sum_ = 0.0
    for elem in sorted_seq:
        sum_ += elem.weight
        if sum_ >= u:
            return elem

def main(filename, samples, oracle, plot, path, seed=None):
    random.seed(seed)
    numpy.random.seed(seed)
    
    save = path is not None

    if (plot or save) and not MATPLOTLIB:
        print('Warning: the --plot or --save-fig options were activated,'\
              'but matplotlib is unavailable. ' \
              'Processing will continue without plotting.')
    
    # Read file and initialize distributions
    lDistributions = read_file(filename)
    
    # Initialize figure and axis before plotting
    if (plot or save) and MATPLOTLIB:
        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111)
        if plot:
            plt.ion()
            plt.show()
        points = deque(maxlen=LAST_N_PTS)
        labels = deque(maxlen=LAST_N_PTS)
        ref_labels = list(map(attrgetter('label'), lDistributions))


    # Print CSV header
    if oracle:
        print("%s, %s, %s" % ('label',
                              ", ".join('x%i'% i for i in
                                        range(len(lDistributions[0].distributions[0].centroid))), 
                              ", ".join('%s' % class_.label for class_ in
                                        class_list)))
    else:
        print("%s, %s" % ('label',
                          ", ".join('x%i'% i for i in
                                    range(len(lDistributions[0].distributions[0].centroid))))) 
        

    for i in range(samples):
        cdistrib = weight_choice([distrib for distrib in cclass.distributions
                                  if distrib.start_time <= i])
        spoint = cdistrib.sample()[0]

        # Compute the probability for each class
        # The probability of having sampled a point from a class C_i knowing x is given by :
        # P(C_i | x) = \frac{P(C_i) p(x | C_i)}{\sum_j{P(C_j) p(x | C_j) }}
        # p(x | C_i) = \sum_k{ P(G_k) p(x | G_k)}
        # Where G_k are the gaussian distribution of class C_i
        probs = []
        classes_weight_sum = sum(class_.weight for class_ in class_list 
                                 if class_.start_time <= i)
        for class_ in class_list:
            if class_.start_time <= i:
                prob_class = class_.weight / classes_weight_sum
                prob_dist = 0.0
                dists_weight_sum = sum(dist.weight 
                                       for dist in class_.distributions
                                       if dist.start_time <= i)
                for dist in class_.distributions:
                    prob_dist += dist.weight / dists_weight_sum * dist.pdf(spoint)
                probs.append(prob_class * prob_dist)
            else:
                probs.append(0.0)
        
        # Normalize probabilities
        probs = list(map(truediv, probs, repeat(sum(probs), len(probs))))

        # Print the sampled point in CSV format
        if oracle:
            print("%s, %s, %s" % (str(cclass.label), 
                                  ", ".join("%s" % v for v in spoint), 
                                  ", ".join("%.3f" % prob for prob in probs)))
        else:
            print("%s, %s" % (str(cclass.label), 
                              ", ".join("%s" % v for v in spoint))) 

        # Plot the resulting distribution if required
        if (plot or save) and MATPLOTLIB:
            points.append(spoint)
            labels.append(class_.label)
            plot_class(i, ref_labels, class_list, points, labels, fig, ax1)
            if save:
                fig.savefig(path+'/point_%i.png' % i)
        
        # Update the classes' distributions
        for class_ in class_list:
            class_.update(i)

    if plot and MATPLOTLIB:
        plt.ioff()
        plt.show()

    return class_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read a file of classes'\
            'and return a series of randomly sampled points from those '\
            'classes.')
    parser.add_argument('filename', help='json file containing the classes')
    parser.add_argument('samples', type=int, help='number of samples')
    
    parser.add_argument('--oracle', dest='oracle', required=False, 
                        action='store_true', default=False,
                        help='append to the point its probability of ' \
                             'belonging to each class')
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
    main(args.filename, args.samples, args.oracle, args.plot, args.save_path, args.seed)

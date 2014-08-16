from math import cos, sin, radians
import numpy

class Distribution(object):
    """This class incapsulates a non-stationary multivariate normal distribution. 
    The object contains a list of phases that are applied sequentially, and that 
    can modify the scale, the position and the angle of the distribution."""
    def __init__(self, iDict, iN):
        """ Parse dictionary *iDict*; argument *iN* is the distribution number
        for error messages."""

        # process class label
        self._class = iDict.get("class")
        if self._class == None:
            print("\aError, distribution {} is missing a `label' attribute".format(iN))
            gAbort = True

        #process distribution id
        self._id = iDict.get("id")

        # process number of dimensions
        self._dims = iDict.get("dims")
        if self._dims == None:
            print("\aError, distribution {} is missing a `dims' attribute".format(iN))
            gAbort = True

        # process initial distribution weight
        lWeight = iDict.get("weight")
        if lWeight == None:
            print("\aError, distribution {} is missing a `weight' attribute".format(iN))
            gAbort = True
        if lWeight != None and lWeight < 0:
            print("\aError, distribution {} has a negative `weight' attribute".format(iN))
            gAbort = True
        self._weights = [lWeight]

        # process ditribution initial center
        lCenter = iDict.get("center")
        if lCenter == None:
            print("\aError, distribution {} is missing a `center' attribute".format(iN))
            gAbort = True
        elif len(lCenter) != self._dims:
            print("\aError, distribution {} has an invalid `center' attribute".format(iN))
            gAbort = True
        self._centers = [numpy.array(lCenter, dtype=numpy.float64)]

        # process principal components
        lDev = iDict.get("stddev")
        if lDev == None:
            print("\aError, distribution {} is missing a `stddev' attribute".format(iN))
            gAbort = True
        elif len(lDev) != self._dims:
            print("\aError, distribution {} has an invalid `stddev' attribute".format(iN))
            gAbort = True
        
        lCovar = numpy.identity(self._dims, dtype=numpy.float64)
        lCovar *= numpy.diag(lDev)
        self._covars = [lCovar]

        # process matrix rotation
        lRotation = iDict.get("rotate", None)
        if lRotation != None:
            lRotation = numpy.radians(lRotation)
            if len(lRotation) != self._dims*(self._dims-1)//2:
                print("\aError, a phase in distribution {}".format(iN) +
                      " has an invalid number of rotation angles")
                gAbort = True
            lMatrix = createRotationMatrix(self._dims, lRotation)
            self._covars[-1] = numpy.dot(lMatrix.T, numpy.dot(self._covars[-1], lMatrix))

        self._indices = [0]        
        self._rotations = []
        self._scales = []

        # process distribution phases
        for lPhase in iDict["phases"]:

            # process phase duration
            lDuration = lPhase.get("duration")
            if lDuration == None:
                print("\aError, a phase in distribution {}".format(iN) +
                      " is missing a `duration' attribute")
                gAbort = True
            elif lDuration < 0:
                print("\aError, a phase in distribution {}".format(iN) +
                      " has a negative `duration' attribute")
                gAbort = True
            self._indices.append(self._indices[-1]+lDuration)

            # process phase weight
            lWeight = lPhase.get("weight")
            if lWeight == None:
                self._weights.append(self._weights[-1])
            elif lWeight < 0:
                print("\aError, a phase in distribution {}".format(iN) +
                      " has a negative `weight' attribute")
                gAbort = True
            else:
                self._weights.append(lWeight)

            # process phase translation
            lMoveto = lPhase.get("moveto")
            if lMoveto == None:
                lMoveto = lPhase.get("rmoveto")
                if lMoveto == None:
                    self._centers.append(self._centers[-1])
                else:
                    self._centers.append(self._centers[-1]+lMoveto)
            else:
                if lPhase.get("rmoveto"):
                    print("\aError, a phase in distribution {}".format(i) +
                          "contains both a `moveto' and `rmoveto' attribute")
                    gAbort = True
                self._centers.append(lMoveto)
           
            # process phase scaling
            lScale = lPhase.get("scale", None)
            if lScale != None and lScale < 0:
                print("\aError, a phase in distribution {}".format(iN) +
                      " has a negative `scale' attribute")
                gAbort = True
            self._scales.append(lScale)

            # process phase rotation
            lRotation = lPhase.get("rotate", None)
            x = self.getDims()
            if lRotation != None:
                lRotation = numpy.radians(lRotation)
                if len(lRotation) != x*(x-1)//2:
                    print("\aError, a phase in distribution {}".format(iN) +
                          " has an invalid number of rotation angles")
                    gAbort = True
            self._rotations.append(lRotation)

            lCovar = numpy.copy(self._covars[-1])
            if lScale != None:
                lCovar *= lScale
            if lRotation != None:
                lMatrix = createRotationMatrix(self.getDims(), self._rotations[-1])
                lCovar = numpy.dot(lMatrix.T, numpy.dot(lCovar, lMatrix))
            self._covars.append(lCovar)

            # check for unknown phase attributes
            lValidPhaseAttrs = ("duration", "rotate", "scale", "weight", "moveto", "rmoveto")
            for lAttr in lPhase:
                if not lAttr in lValidPhaseAttrs:
                    print("\aError, unknown `{}' phase attribute in distribution {}".format(lAttr, iN))
                    gAbort = True

        # check for unknown distribution attributes
        lValidDistAttrs = ("class", "id", "dims", "weight", "center", "stddev", "rotate", "phases")
        for lAttr in iDict:
            if not lAttr in lValidDistAttrs:
                print("\aError, unknown `{}' attribute in distribution {}".format(lAttr, iN))
                gAbort = True

        # initialize current time
        self.setTime(0)

    def getClassLabel(self):
        """Returns the distribution class label."""
        return self._class

    def getDims(self):
        """Returns the number of dimensions of the distribution."""
        return self._dims
        
    def getCurrentCenter(self):
        """Returns the current distribution mean vector."""
        
        if self._curPhase == None:
            return None
        elif self._curPhase == -1:
            return self._centers[-1]

        i = self._curPhase; x = self._curFraction
        return self._centers[i] + x * (self._centers[i+1]-self._centers[i])

    def getCurrentCovar(self):
        """Returns the current distribution covariance matrix."""
        
        if self._curPhase == None:
            return None
        elif self._curPhase == -1:
            return self._covars[-1]

        i = self._curPhase; x = self._curFraction
        lCovar = numpy.copy(self._covars[i])
        if self._scales[i] != None:
            lScale = 1.0 + x*(self._scales[i]-1.0)
            lCovar *= lScale
        if self._rotations[i] != None:
            lRotation = numpy.array(self._rotations[i], dtype=numpy.float64)
            lRotation *= x
            lMatrix = createRotationMatrix(self.getDims(), lRotation)
            lCovar = numpy.dot(lMatrix.T, numpy.dot(lCovar, lMatrix))

        return lCovar

    def getCurrentWeight(self):
        """Returns the current distribution a priori probability."""

        if self._curPhase == None:
            return 0
        elif self._curPhase == -1:
            return self._weights[-1]

        i = self._curPhase; x = self._curFraction
        return self._weights[i] + x * (self._weights[i+1]-self._weights[i])

    def setTime(self, iTime):
        """Set the current time for the distribution."""

        assert iTime >= 0
        self._curTime = iTime

        # determine the current phase and fraction
        if iTime < self._indices[0]:
            self._curPhase = None
            self._curFraction = None
        elif iTime >= self._indices[-1]:
            self._curPhase = -1
            self._curFraction = None
        else:
            for (i, x) in enumerate(self._indices, -1):
                if iTime < x: break
            self._curPhase = i
            self._curFraction = (iTime-self._indices[i]) / (self._indices[i+1]-self._indices[i])

def createRotationMatrix(iDim, iAngles):
    """Return rotation matrix for angle *iAngles*."""
    if len(iAngles) != iDim*(iDim-1)/2:
        raise runtime_error("invalid number of angles")
    lMatrix = numpy.identity(iDim, dtype=numpy.float64)
    k = 0
    for i in range(iDim-1):
        for j in range(i+1, iDim):
            lTmp = numpy.identity(iDim, dtype=numpy.float64)
            lSign = 1.0
            if i+j % 2 == 0:
                lSign = -1.0
            lTmp[i][i] = cos(iAngles[k])
            lTmp[j][j] = cos(iAngles[k])
            lTmp[i][j] = -lSign*sin(iAngles[k])
            lTmp[j][i] = lSign*sin(iAngles[k])
            lMatrix = numpy.dot(lMatrix, lTmp)
            k += 1
    return lMatrix

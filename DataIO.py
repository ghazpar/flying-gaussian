try:
    import arff
except:
    print("\nModule liac-arff is required to read data in arff format!")
    print("Install using 'pip install liac-arff'\n")
    exit()

def read(iFilename):
    """ Read data in arff format. """

    if iFilename == '-':
        import sys
        lFD = sys.stdin
    else:
        try:
            lFD = open(iFilename)
        except:
            lFD = open(iFilename+'.arff')

    return arff.load(lFD)

def write(iRelation, iAttrs, iData):
    """ Write data in arff format. """

    lInfo = {}
    lInfo['description'] = "\nFlying non-stationary gaussians, 2014\n"
    lInfo['relation'] = iRelation
    lInfo['attributes'] = iAttrs
    lInfo['data'] = iData

    print(arff.dumps(lInfo))

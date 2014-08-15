import csv, sys

def readData(iFilename, iFormat):
    """
    Read the data file.

    Returns a list of tuples containing a numpy array and.
    """
    try:
        if iFilename == 'stdin':
            import sys
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
        lFile = arff.Reader(lFD)
    else: 
        print("\aError, invalid format: ", iFormat)
        exit()

    lData = []
    for lRow in lFile:
        # skip header row
        if list(lRow)[-1] == 'label': continue

        # create tuple (array, label)
        lData.append(list(lRow))

    return lData

def writeData(iHeader, iData, iFormat):
    """ Write output file in specified format. """

    if iFormat == 'csv':
        iData.insert(0, [x[0] for x in iHeader['attrs']])
    elif iFormat == 'arff':
        print('% Flying Gaussians, Marc Parizeau, 2014')
        print('\n@relation', iHeader['filename'])
        print()
        for n, t in iHeader['attrs']:
            print("@attribute", n, t)
        print('\n@data')
    else:
        print("\aError, invalid format: ", iFormat)
        exit()

    lFile = csv.writer(sys.stdout)
    for lRow in iData:
        lFile.writerow(lRow)

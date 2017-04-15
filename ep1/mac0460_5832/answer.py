def estimatePatternResults(src, dest):
    """
    Slides the window W through the src image and counts the number of times each
    1x3 pattern shows up with the corresponding value (True or False) for the
    center element in the dest image
    Returns a frequency table which serves as an estimator of
    P(pattern,0) and P(pattern,1)
    """

    freqtable = {}

    for i in range(src.shape[0]):
        for j in range(0, src.shape[1]-2): # clips the borders

            pattern = tuple(src[i, j:j+3]) # lists are not hashable, but tuples are
            result = dest[i, j+1]          # center pixel in the structuring element

            if not pattern in freqtable:
                freqtable[pattern] = { True : 0, False : 0 }
            freqtable[pattern][result] += 1

    return freqtable


def predictPatternOutput(pattern, freqtable):
    """
    Returns the value that minimizes MAE for this pattern considering the
    observations given by the frequency table
    """
    return freqtable[pattern][True] > freqtable[pattern][False]

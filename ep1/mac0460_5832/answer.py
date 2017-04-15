def findPatternMappings(src, dest):

    freqs = {}

    for i in range(src.shape[0]):
        for j in range(0,src.shape[1]-2):

            pattern = tuple(src[i,j:j+3]) # lists are not hashable, but tuples are
            result = dest[i,j+1]

            if pattern in freqs:
                freqs[pattern].append(result)
            else:
                freqs[pattern] = [result]

    return freqs


def mapPatternsToOutput(freqs):

    return { pattern: sum(freqs[pattern]) > (len(freqs[pattern])/2)
             for pattern in freqs }

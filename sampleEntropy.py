import numpy as np


def sampleEntropy(seq, wlen, r, shift):
    """
    Sample Entropy (python-version)

    SampEn = sampleEntropy(INPUT, M, R, TAU)

    Arguments:
        INPUT       Nx1         Input sequence.
        M           Int         Window-length (or "dimension").一般取2
        R           Double      Tolerance for "similarity". 一般取0.2*std
        TAU         Int         Spacing of valid samples (for subsampling).
                                A value of 1 corresponds to no subsampling,
                                2 takes every other value, etc.
    """

    if shift > 1:
        seq = seq[::shift]  # downsample(seq,shift) 对应Python的切片操作

    # allocate space for extracted windows
    D = np.zeros((len(seq) - wlen - 1, wlen + 1))  # zeros(length(seq)-(wlen+1), wlen+1)

    # extract windows with length wlen+1
    for pos in range(len(seq) - wlen - 1):  # pos=1:length(seq)-wlen-1
        D[pos, :] = seq[pos:pos + wlen + 1]  # D(pos,:) = seq(pos:pos+wlen)

    # initialise
    A = 0
    B = 0

    # calculate number of windows with pairwise distance of less than r, for
    # two cases:
    #   1) B = with windows = 1..wlen
    #   2) A = with windows = 1..wlen+1
    for i in range(D.shape[0]):  # i=1:size(D,1)
        # Chebyshev distance is max(abs(d_ik-d_jk))
        # D(i,i) is 0, but we should not count that.
        # Also D(i,j) is symmetrical (d(i,j)=d(j,i)), therefore we just need to
        # look at D(i+1:end). Effectively we only calculate "half" of the
        # distance matrix. Due to symmetry we can ignore the rest.
        DD = np.abs(D[i + 1:, :] - D[i, :])  # subtract current window from all future windows

        v1 = np.max(DD[:, :-1], axis=1)  # maximum along 2nd dim (case 1)
        v2 = np.maximum(v1, DD[:, -1])  # add last column (case 2)

        B = B + np.sum(v1 < r)
        A = A + np.sum(v2 < r)

    # A contains half the matches,
    # B contains half the matches. For estimating A/B this doesn't matter
    # really.
    d = -np.log(A / B)

    return d 
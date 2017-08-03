try:
    import scipy
except:
    print "Use anaconda installation for kmeans!"
    exit(0)

import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

sample_inputs = whiten(np.loadtxt("sample_inputs.csv", delimiter=",", dtype=np.float))

kmeans_result = kmeans(sample_inputs, 10)
distortion = kmeans_result[1]
print "distortion is " + str(distortion)
landmarks = kmeans_result[0]
np.savetxt("landmarks.csv", landmarks, delimiter=",")


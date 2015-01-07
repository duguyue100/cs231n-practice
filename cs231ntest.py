"""
Author : Yuhuang Hu
Date   : 2015-01-07

Perform some tests for the written code
"""

import numpy as np;
import matplotlib.pyplot as plt;

from cs231nlib.classifier import NearestNeighbor;

## Testing for Nearest Neighbor Function

nn=NearestNeighbor();

# prepare dataset
Xtr=np.random.random((30, 3));

Xtr[:,2]=(Xtr[:,2]>=0.5);

X=Xtr[:,0:2];
Y=Xtr[:,2];

# plot dataset

plt.figure(1);
plt.plot(X[:,0], X[:,1], 'ro');
plt.show();

nn.train(x=X, y=Y);

Y_predict=nn.predict(np.random.random((15,2)));

print Y_predict;

'''
Created on Jan 7, 2015

@author: Yuhuang Hu
@note: This module consists of several classifiers class.
'''

import numpy as np;

class NearestNeighbor:
    """
    @note: this class provides an implementation of NearestNeighbor classifier.
    """
    
    def __init__(self):
        pass
    
    def train(self, x, y):
        """
        @param x: N x D matrix, each row is a D-dimensional vector.
        @param y: N x 1 vector, describes corresponding labels. 
        """
        
        self.Xtr=x;
        self.Ytr=y;

    def predict(self, x):
        """
        @param x: N x D matrix, each row is a D-dimensional vector.
        @return: a 1D vector that consists of all predicted labels 
        """
        
        num_test=x.shape[0];
        
        Y_pred=np.zeros(num_test, dtype=self.Ytr.dtype);
        
        for i in xrange(num_test):
            
            distances=np.sum(np.abs(self.Xtr-x[i,:]), axis=1);
            min_index=np.argmax(distances);
            Y_pred[i]=self.Ytr[min_index];
            
        return Y_pred;
        
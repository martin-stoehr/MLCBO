# Wrapper around sklearn's GaussianProcessClassifier that accepts
# data of one class (no unfeasible point encountered)
# Classifier.predict returns value of that class
# Classifier.predict_proba returns min( 1, sum_x'( RBF(x,x') ) )

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC


class SingleClassPredicter:
    def __init__(self, cat): self.cat = cat
    
    def __call__(self, x):
        """
        Predict class for input x
        
        Parameters
        ----------
            x: array-like, shape (n_samples, *n_features)
        
        Returns
        -------
            cat: predicter's class, shape (n_samples, shape(cat))
        """
        return np.array([self.cat,]*x.shape[0])
    
class SingleClassProba:
    def __init__(self, X): self.X, self.gamma = X, 2./(X.shape[1] * X.var())
    
    def __call__(self, x):
        """
        Predict class for input x
        
        Parameters
        ----------
            x: array-like, shape (n_samples, n_features)
        
        Returns
        -------
            p: probability of being in predicter's class, shape (n_samples,)
               as given by sum of Gaussians with width = 1 / (n_features * X.var())
        """
        p = np.zeros(x.shape[0])
        for xi in self.X: p += np.exp(-self.gamma * np.square(x - xi).sum(axis=1))
        p_class = np.clip(p, 0., 1.)
        p_not = 1. - p_class
        return np.stack((p_class, p_not), axis=1)
    

class GPCFlex(GaussianProcessClassifier):
    def __init__(self, *args, **kwargs):
        self.base_gpc = GaussianProcessClassifier(*args, **kwargs)
    
    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes == 1:
            self.predict = SingleClassPredicter(classes)
            self.predict_proba = SingleClassProba(X)
        else:
            self.base_gpc.fit(X, y)
            self.predict = self.base_gpc.predict
            self.predict_proba = self.base_gpc.predict_proba
    

class SVCFlex(SVC):
    def __init__(self, *args, **kwargs):
        self.base_svm = SVC(*args, **kwargs)
    
    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes == 1:
            self.predict = SingleClassPredicter(classes)
        else:
            self.base_svm.fit(X, y)
            self.predict = self.base_svm.predict

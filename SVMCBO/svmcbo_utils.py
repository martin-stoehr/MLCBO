# Created by ricky at 15/12/2019
# Modified by Martin at 03/10/2023

import numpy as np
import scipy as spy
from joblib import Parallel, delayed
from sklearn.svm import SVC
#from pyDOE2 import lhs
import warnings

def estimateFeasibleRegion(x,y, gamma):
    svm_model = SVC(gamma=gamma)
    svm_model.fit(x, y)
    return svm_model

def evaluateEstimatedFeasibility(x, svm):
    y = svm.predict(x)
    return y

def coverage(x, sampledPoints, gamma):
    x = np.array(x)
    sampledPoints= np.array(sampledPoints)
    C = 0
    for i in np.arange(0, len(sampledPoints)):
        C = C + np.exp(-np.sum((x - sampledPoints[i, ]) ** 2) / (2 * gamma**2))
    return C

def boundness(x, svm):
    B = abs(svmSurface(x, svm))
    return B

def svmSurface(x,svm):
    x = np.array(x)
    coeffs = svm.dual_coef_[0]
    svs = svm.support_vectors_
    gamma = svm.gamma # because selected gamma='auto' in sklearn SVC
    sigma = 1/gamma
    f = 0
    for k in np.arange(0,len(coeffs)):
        f = f + coeffs[k] * np.exp(-sigma * (np.sum((x-svs[k, ])**2)))
    return f

def phase1AcquisitionFunction(x, args):
    sampledPoints = args["sampledPoints"]
    svm = args["svm"]
    gamma = args["gamma"]
    C = coverage(x, sampledPoints, gamma)
    B = boundness(x, svm)
    value = C+B
    return value

def nextPointPhase1(sampledPoints, svm, gamma, sampler, dimensions_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        startingPoints = np.array(sampler.generate(dimensions_test, 32))
    # optional parameters for minimization scipy
    additional = {'sampledPoints':sampledPoints, 'svm':svm, 'gamma':gamma}
    locOptX, locOptY = list(), list()

    ##TODO: avoid hard-coded parallelism
    results = Parallel(6)(
        delayed(spy.optimize.minimize)(
            fun=phase1AcquisitionFunction,
            args=additional,
            x0=x0,
            bounds=dimensions_test,
            method='L-BFGS-B',
            options={'maxiter':100})
        for x0 in startingPoints)

    for ix in range(len(results)):
        locOptX = locOptX + [results[ix]['x']]
        locOptY = locOptY + [results[ix]['fun']]

    ix = np.where(np.array(locOptY) == np.min(np.array(locOptY)))[0][0]

    return (np.array([locOptX[ix]]))


def acquisition_function(x, args, beta=1.96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = args["model"]
        classifier = args["classifier"]
        mu, std = model.predict(x, return_std=True)
        labels = classifier.predict(x)
        label_neg = np.where(labels == 1)[0]
        label_pos = np.where(labels == 0)[0]
        mu[label_neg] = np.max(mu)
        lcb = mu - beta * std
        return lcb

# def acquisition_function_penalty(x, args, beta=1.96):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#
#         model = args["model"]
#         mu, std = model.predict(x, return_std=True)
#         lcb = mu - beta * std
#         return lcb

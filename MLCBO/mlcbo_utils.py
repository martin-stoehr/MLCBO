# Created by ricky at 15/12/2019
# Modified by Martin at 03/10/2023

import numpy as np
import scipy as spy
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
import warnings



def mod_sigmoid(x, a=20., mu=0.68):
    y = 1. / (1. + np.exp(-a * (x - mu))) - 1. / (1. + np.exp(a*mu))
    s = 1. / (1. + np.exp(a * (mu - 1.))) - 1. / (1. + np.exp(a*mu))
    return y / s

def estimateFeasibleRegion(x, y, classifier):
    classifier.fit(x, y)
    return classifier

def evaluateEstimatedFeasibility(x, classifier):
    y = classifier.predict(x)
    return y

def coverage(x, sampledPoints, gamma):
    x = np.array(x)
    sampledPoints = np.array(sampledPoints)
    sigma2, C = 2 * gamma*gamma, 0
    for i in np.arange(0, len(sampledPoints)):
        C = C + np.exp( -np.sum((x - sampledPoints[i, ]) ** 2) / sigma2 )
    return C

def boundness(x, classifier):
    B = abs(estimate_p_nonfeasible(x, classifier))
    return B

def estimate_p_nonfeasible(x, classifier):
    """ Estimated probability of set if x's being non-feasible """
    x = np.array(x)
    if isinstance(classifier, SVC):
        coeffs = classifier.dual_coef_[0]
        svs = classifier.support_vectors_
        sigma = 1 / classifier.gamma
        f = 0
        for k in np.arange(0,len(coeffs)):
            f = f + coeffs[k] * np.exp(-sigma * (np.sum((x-svs[k, ])**2)))
        return f
    elif isinstance(classifier, GPC):
        if x.ndim < 2: x = x.reshape((1,-1))
        p_nf = classifier.predict_proba(x)[0,1]
        return mod_sigmoid(p_nf)


def phase1AcquisitionFunction(x, args):
    C = coverage(x, args['sampledPoints'], args['gamma'])
    B = boundness(x, args['classifier'])
    value = C + B
    return value

def nextPointPhase1(sampledPoints, classifier, sampler, gamma, dimensions_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        startingPoints = np.array(sampler.generate(dimensions_test, 32))
    # optional parameters for minimization scipy
    additional = {'sampledPoints':sampledPoints, 'classifier':classifier, 'gamma':gamma}
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


def acquisition_function_GPC(x, args, beta=1.96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = args["model"]
        classifier = args["classifier"]
        ## get mean and std of surrogate model
        mu, std = model.predict(x, return_std=True)
        ## get probability of non-feasibility
        #TODO: sigmoid filter or binary classifier?
        p_nf = classifier.predict_proba(x)[:,1]
        p_nf = mod_sigmoid(p_nf)
#        labels = p_nf > 0.5
#        label_neg = np.where(labels)[0]
#        mu[label_neg] = np.max(mu)
        ## mean objective = mean + P(non-feasible)*|max(mean)|
        ## total objective = mean objective - beta*std
        ## this motivates to explore regions with high uncertainty
        lcb = mu + np.abs(np.max(mu)) * p_nf - beta * std
#        lcb = mu - beta * std
        return lcb
        

def acquisition_function_SVC(x, args, beta=1.96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = args["model"]
        classifier = args["classifier"]
        mu, std = model.predict(x, return_std=True)
        labels = classifier.predict(x)
        label_neg = np.where(labels == 1)[0]
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

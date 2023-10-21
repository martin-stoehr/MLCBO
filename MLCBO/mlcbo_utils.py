# Created by ricky at 15/12/2019
# Modified by Martin at 03/10/2023

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
import warnings

NUMJOBS_OPT = 8


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

def coverage(x, sampledPoints):
    x = np.array(x)
    sampledPoints = np.array(sampledPoints)
    var = sampledPoints.var()
    sigma2, C = 2 * var * var, 0
    for i in np.arange(0, len(sampledPoints)):
        C = C + np.exp( -np.sum((x - sampledPoints[i, ]) ** 2) / sigma2 )
    return C

#def boundness(x, classifier):
#    B = abs(estimate_p_nonfeasible(x, classifier))
#    return B

def estimate_p_nonfeasible(x, classifier):
    """ Estimated probability of set if x's being non-feasible """
    x = np.array(x)
#    if isinstance(classifier, SVC):
#        coeffs = classifier.dual_coef_[0]
#        svs = classifier.support_vectors_
#        p = 0
#        for k in np.arange(0,len(coeffs)):
#            p = p + coeffs[k] * np.exp(-classifier.gamma * (np.sum((x-svs[k, ])**2)))
#        return p
#    elif isinstance(classifier, GPC):
    if x.ndim < 2: x = x.reshape((1,-1))
    p_nf = classifier.predict_proba(x)[0,1]
    return mod_sigmoid(p_nf)
#    return p_nf


def phase1AcquisitionFunction(x, args):
    C = coverage(x, args['sampledPoints'])
#    B = boundness(x, args['classifier'])
    B = estimate_p_nonfeasible(x, args['classifier'])
    value = C + B
    return value

def NextPointExploration(sampledPoints, classifier, sampler, dimensions_test, n_init_opt=24):
    """
    Generate new point based on coverage and estimated bounds.
    
    Parameters
    ----------
        sampledPoints: array-like, shape (*, n_features)
            points sampled so far
        classifier: sklearn's SVC or GaussianProcessClassifier obj
            classifier for feasible region (SVC: probabilities enables!)
        sampler: skopt's sampler obj
            sampling algorithm (e.g., lhs, sobol)
        dimension_test: array-like
            current boundaries for (quasi-)random search
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        startingPoints = np.array(sampler.generate(dimensions_test, n_init_opt))
    # optional parameters for minimization scipy
    additional = {'sampledPoints':sampledPoints, 'classifier':classifier}
    locOptX, locOptY = list(), list()

    ##TODO: avoid hard-coded parallelism
    results = Parallel(NUMJOBS_OPT)(
                delayed(minimize)(
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


def acquisition_function_smooth(x, args, beta=1.96):
    """
    Continuous acquisition function based on lower confidence bound (lcb)
    of surrogate and probability of feasibility (p_f):
        lcb = mean(surrogate) - beta * std(surrogate)
        f_acq = p_f * lcb + (1 - p_f) * max( mean(surrogate) )
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = args["model"]
        classifier = args["classifier"]
        if x.ndim < 2: x = x.reshape(1,-1)
        ## get mean and std of surrogate model
        mu, std = model.predict(x, return_std=True)
        ## get probability of non-feasibility
        p_nf = mod_sigmoid(classifier.predict_proba(x)[:,1])
#        p_nf = classifier.predict_proba(x)[:,1]
        p_f = 1. - p_nf
        ## masked_mu = mean + P(non-feasible) * |max(mean)|
        ## total objective = masked_mu - beta * std
        lcb = p_f * (mu - beta * std)
        f_acq = lcb + p_nf * 1e2#np.max(mu)
        return f_acq
        

def acquisition_function_binary(x, args, beta=1.96):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = args["model"]
        classifier = args["classifier"]
        mu, std = model.predict(x, return_std=True)
        labels = classifier.predict(x)
        mu[np.where(labels)[0]] = np.max(mu)
        lcb = mu - beta * std
        return lcb


import numpy as np
import warnings
from joblib import Parallel, delayed
from scipy.optimize import minimize
from .mlcbo_utils import NUMJOBS_OPT


def focus_search(f, args, sampler, bounds, n_restart=3, n_focus=5):
    bounds = args.pop("bounds", np.array([[0.,1.],[0.,1.]]))
    _ = args.pop("max_iter", 0)
    n_samples = args.pop("n_samples", 5000)
    cand_points, cand_acq = [], []
    for idx_start in range(n_restart):
        optimal_point = optimal_value = []
        new_bounds = bounds

        for iter_n in range(n_focus):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") 
                x = np.array(sampler.generate(new_bounds, n_samples))
            y = f(x, args)
            x_star = x[np.argmin(y)]
            y_star = np.min(y)
            optimal_point = optimal_point + [x_star]
            optimal_value = optimal_value + [y_star]
            new_bounds = []

            for i in range(len(bounds)):
                l_xi = np.min(x[:, i])
                u_xi = np.max(x[:, i])
                new_l_xi = np.max([l_xi, x_star[i] - 0.5 * (u_xi - l_xi)])
                new_u_xi = np.min([u_xi, x_star[i] + 0.5 * (u_xi - l_xi)])
                new_bounds = new_bounds + [[new_l_xi, new_u_xi]]
        
        optimal_value = np.array(optimal_value)
        optimal_point = optimal_point[np.argmin(optimal_value)]
        optimal_value = np.min(optimal_value)

        cand_points = cand_points + [optimal_point]
        cand_acq = cand_acq + [optimal_value]

    final_cand_point = cand_points[np.argmin(cand_acq)]
    final_cand_acq = np.min(cand_acq)
    return (final_cand_point, final_cand_acq)

def focus_search_parallel(f, args, sampler, n_restart=5, n_focus=10):
    bounds = args["bounds"]
    best_x = args.pop('best_x', [])
    n_samples = args.pop('n_samples', 5000)
    _ = args.pop("max_iter", 20)
    ##TODO: avoid hard-coded parallelism -> vectorize
    results = Parallel(NUMJOBS_OPT)(
                delayed(focusing)(
                    f, bounds, sampler, n_focus, args,
                    n_samples=n_samples, best_x=best_x)
                for i in range(n_restart))
    cand_xs = np.array([r[0] for r in results])
    cand_acqs = np.array([r[1] for r in results])
    classifier = args["classifier"]
    labels_cand = classifier.predict(cand_xs)
    next_x = cand_xs[np.argmin(cand_acqs)]
    ## focus gives one feasible point
    if np.count_nonzero(labels_cand==0) == 1:
        next_x = cand_xs[np.where(labels_cand == 0)][0]
    ## focus returns multiple feasible points
    elif np.count_nonzero(labels_cand==0) > 1:
        feasible = np.where(labels_cand==0)[0]
        cand_xs_pos = cand_xs[feasible]
        cand_acqs_pos = cand_acqs[feasible]
        next_x = cand_xs_pos[np.argmin(cand_acqs_pos)]
    ## focus gives no feasible point: resample
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            x = np.array(sampler.generate(new_bounds, n_samples))
        labels_cand = classifier.predict(x)
        if np.count_nonzero(labels_cand==1) == n_samples:
            next_x = x[-1]
        else:
            x = x[np.where(labels_cand == 0)]
            values = f(x, args)
            next_x = x[np.argmin(values)]
    return next_x.reshape(1,-1)

def focusing(f, bounds, sampler, n_iter, args, n_samples=5000, best_x=[]):
    optimal_point = optimal_value = []
    new_bounds = bounds
    for iter_n in range(n_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_sample = sampler.generate(new_bounds, n_samples)
            x = np.array(x_sample + best_x)
        classifier = args["classifier"]
        labels_cand = classifier.predict(x)
        if len(np.where(labels_cand == 1)[0]) == n_samples:
            print(len(np.where(labels_cand == 1)[0]))
            return x[-1], np.inf
        
        x = x[np.where(labels_cand == 0)]
        y = f(x, args) # call acquisition function
        x_star = x[np.argmin(y)]
        y_star = np.min(y)
        optimal_point = optimal_point + [x_star]
        optimal_value = optimal_value + [y_star]
        new_bounds = []
        for i in range(len(bounds)):
            l_xi = np.min(x[:, i])
            u_xi = np.max(x[:, i])
            new_l_xi = np.max([l_xi, x_star[i] - 0.5 * (u_xi - l_xi)])
            new_u_xi = np.min([u_xi, x_star[i] + 0.5 * (u_xi - l_xi)])
            new_bounds = new_bounds + [[new_l_xi, new_u_xi]]
        #print("Shrinked Bounds: {}".format(new_bounds))
    optimal_value = y_star
    optimal_point = x_star
    return optimal_point, optimal_value
    
def focus_search_local_opt(f, args, sampler):
    bounds = args.pop("bounds")
    n_iter = args.pop("max_iter", 20)
    n0_opt = args.pop('n_samples', 100)
    x0_opt = np.array(sampler.generate(bounds, n0_opt))
    ##TODO: avoid hard-coded parallelism -> vectorize
    results = Parallel(NUMJOBS_OPT)(
                    delayed(local_opt)(f, args, bounds, x0, n_iter=n_iter)
                for x0 in x0_opt)
    x_opts = np.array([r.get('x') for r in results])
    y_opts = np.array([r.get('fun') for r in results])
    next_x = x_opts[np.argmin(y_opts)]
    return next_x.reshape(1,-1)

def local_opt(f, args, bounds, x0, n_iter=20):
    res = minimize(fun=f, args=args, x0=x0, bounds=bounds,
                    method='L-BFGS-B', options={'maxiter':n_iter})
    return res


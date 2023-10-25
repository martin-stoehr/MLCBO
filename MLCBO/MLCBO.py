import numpy as np
from abc import ABC, abstractmethod
from sklearn.gaussian_process import kernels
from skopt.sampler import Sobol, Lhs
from .my_gp import *
from .my_rf import *
from .mlcbo_utils import *
from .focus_search import *
from .flex_classifiers import SVCFlex, GPCFlex


sampler_settings = {"lhs"   : {"criterion":"maximin", "iterations":100},
                    "sobol" : {}}

class ABCBO(ABC):
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init_grid=10, iter_exploration=60,
                 iter_exploitation=30, surrogate_type="GP", surrogate_kwargs={},
                 surrogate_kernel="RBF", surrogate_kernel_kwargs={}, 
                 classifier_kwargs={}, sampler="sobol", sampler_kwargs={},
                 log_models=False, noise=None, seed=42, n_focus=10, n_resample=5,
                 n_init_exploration=32, n_init_exploitation=32,
                 retrain_classifier_if_feasible=True, debug=False):
        """
        Abstract base class for (bound) Bayesian optimization with constraints from
        ML classifier (active learning framework).
        
        vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        > !TODO: move runtime parameters to subroutines! <
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        Parameters
        ----------
            f: callable
                objective function to minimize
            bounds: array-like, shape (ndim, 2), default: [(0,1), (0,1)]
                search domain for optimization
                !TODO!: turn this into a moving window (= local search radius)
                        for unbound optimization problems!
            n_init_grid: int, default: 10
                # of points in initial, uninformed sampling of search space
            iter_exploration: int, default: 60
                # of iterations in exploration phase (optimize coverage)
            iter_exploitation: int, default: 30
                # of iterations in exploitation phase (optimize acquisition function)
            surrogate_type: str, default: "GP"
                type of surrogate model (GP: Gaussian Process or RF: Random Forest)
            surrogate_kwargs: dict, default: {}
                extra keyword arguments for surrogate model
            surrogate_kernel: str, default: "RBF"
                kernel for GP surrogate model
            surrogate_kernel_kwargs: dict, default: {}
                keyword arguments for kernel in surrogate model (GP models only)
            classifier_kwargs: dict, default: {}
                keyword arguments for classifier model
            sampler: str, default: "sobol"
                algorithm for (quasi-)random sampling of search space
                "lhs": latin hyper cube
                "sobol": Sobol sequence
            sampler_kwargs: dict, default: {"criterion":"maximin", "iterations":100} for lhs, {} for sobol
                keyword arguments for sampler
            log_models: bool, default: False
                save surrogate and classifier models at every step
            noise: default: None
                whether data is noisy (add white-noise kernel if so)
            seed: int, default: 42
                random seed for (quasi-)random sampling
            n_focus: int, default: 10
                number of focusing steps (localized search around current minimum)
            n_resample: int, default: 5
                number of resampling iterations for (quasi-)random search at every step
            n_init_exploration: int, default: 32
                number of initial guesses for exploration phase (optimize coverage)
            n_init_exploitation: int, default: 32
                number of initial guesses for exploitation phase (optimize acquisition function)
            retrain_classifier_if_feasible: bool or int, default: True
                retrain the classifier also when encountering feasible points, otherwise
                only retrain if unfeasible point is encountered
                if int, retrain every <retrain_classifier_if_feasible> iterations
            debug: bool, default: False
                flag for more verbose output for each phase
        
        """
        self.classifier_kwargs = classifier_kwargs
        if surrogate_type == "GP":
            try:
                self.s_kernel = eval("kernels."+surrogate_kernel)
                self.s_kernel_kwargs = surrogate_kernel_kwargs
            except:
                raise ValueError("`surrogate_kernel` has to be one of `sklearn.gaussian_process.kernels`!")
        self.surrogate_kwargs = surrogate_kwargs
        self.gamma, self.classifier, self.surrogate = None, None, None
        self.noise = noise
        self.x_tot, self.y_tot = [], []
        self.x_feasible, self.y_feasible = [], []
        self.labels, self.n_init_grid = [], n_init_grid
        self.n_init_exploration = n_init_exploration
        self.n_init_exploitation = n_init_exploitation
        self.iter_exploration = iter_exploration
        self.iter_exploitation = iter_exploitation
        self.surrogate_type = surrogate_type.lower()
        self.bounds, self.f = np.array(bounds, dtype=float), f
        if log_models: self.surrogates, self.classifiers = list(), list()
        self.log_models = log_models
        self.c_score, self.seed = list(), seed
        sampler = sampler.lower()
        if sampler not in ["sobol", "lhs"]: raise ValueError("Unknown sampler '"+sampler+"'!")
        sampler_opt = sampler_settings[sampler]
        sampler_opt.update(sampler_kwargs)
        self.sampler = eval(sampler.capitalize() + "(**sampler_opt)")
        if retrain_classifier_if_feasible:
            if isinstance(retrain_classifier_if_feasible, bool):
                self.retrain_classifier_if_feasible = 4
            elif isinstance(retrain_classifier_if_feasible, int):
                self.retrain_classifier_if_feasible = retrain_classifier_if_feasible
            else:
                raise ValueError("'retrain_classifier_if_feasible' has to be bool or int")
        else:
            self.retrain_classifier_if_feasible = np.nan
        self.debug = debug
        

    @abstractmethod
    def init_classifier(self):
        pass

    def init_opt(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.x_tot = np.array(self.sampler.generate(self.bounds, self.n_init_grid, random_state=self.seed))
        print("START INITIALIZATION PHASE:  Sample search domain...")
        self.labels = np.zeros(self.n_init_grid)
        self.y_tot, self.y_feasible, self.x_feasible = np.nan*np.ones(self.n_init_grid), [], []
        for i, x in enumerate(self.x_tot):
            value = self.f(x)
            if np.isnan(value):
                if self.debug: print("      Evaluate at ", x, " -- Non-feasible")
                self.labels[i] = 1
            else:
                if self.debug: print("      Evaluate at ", x, " -- ", value)
                self.labels[i] = 0
                self.y_feasible.append(value)
                self.x_feasible.append(x)
            self.y_tot[i] = value
        self.y_feasible = np.array(self.y_feasible)
        self.x_feasible = np.array(self.x_feasible)
        ## gamma = 1 / (2 * sqrt(n_features) * Var(x))
        ## -> length scale = sqrt( sqrt(n_features) * Var(x) )
        ## sqrt(n_features) rescales L2 norm in higher dimensions
        self.gamma = 0.5 / (np.sqrt(self.bounds.shape[0]) * np.var(self.x_tot, axis=0))
        self.classifier = self.init_classifier(with_proba=True)
        print("END INITIALIZATION PHASE: Number of feasible points: ", self.y_feasible.size, "/", self.n_init_grid)
        print("*" * 64 + "\n")
    
    def exploration(self):
        print("*"*20 + " START EXPLORATION PHASE " + "*"*19)
        self.classifier.fit(self.x_tot, self.labels)
        score = self.classifier.score(self.x_tot, self.labels)
        if self.log_models:
            self.c_score.append(score)
            self.classifiers.append(self.classifier)
        if self.debug:
            print("Predicted labels: ",self.classifier.predict(self.x_tot))
            print("True labels:      ",self.labels)
        
        y_min = np.min(self.y_feasible)
        n_feasible = self.y_feasible.size
        print("  Iter       f(x)           Best      #feasible  Class.Score")
        for i in range(self.iter_exploration):
            # NextPointExploration return new x shape (1,n_features)
            x_new = NextPointExploration(sampledPoints=self.x_tot, classifier=self.classifier,
                                         sampler=self.sampler, dimensions_test=self.bounds,
                                         n_init_opt=self.n_init_exploration)
            self.x_tot = np.concatenate((self.x_tot, x_new))
            value = self.f(x_new[0])
            y_min = min(y_min, value)
            if np.isnan(value):
                if self.debug: print("#   function unfeasible at ",x_new)
                new_label = 1
                print("  {0:4d}   {1:>8f}       {2: 7.5e}   {3:5d}        {4:>5.3f}".format(i+1,value,y_min,n_feasible,score))
            else:
                new_label, n_feasible = 0, n_feasible + 1
                self.y_feasible = np.append(self.y_feasible, value)
                self.x_feasible = np.concatenate((self.x_feasible, x_new))
                print("  {0:4d}   {1: 7.5e}   {2: 7.5e}   {3:5d}        {4:>5.3f}".format(i+1,value,y_min,n_feasible,score))
            self.y_tot = np.append(self.y_tot, value)
            self.labels = np.append(self.labels, new_label)
            self.gamma = 0.5 / (np.sqrt(self.bounds.shape[0]) * np.var(self.x_tot, axis=0))
            self.classifier = self.init_classifier(with_proba=True)
            self.classifier.fit(self.x_tot, self.labels)
            score = self.classifier.score(self.x_tot, self.labels)
            if self.log_models:
                self.c_score.append(score)
                self.classifiers.append(self.classifier)
        print("*" * 21 + " END EXPLORATION PHASE " + "*" * 20 + "\n")
    
    def exploitation(self):
        """
        Run Bayesian optimization with on-the-fly updates of SVM estimator for feasible region.
        """
        print("*" * 19 + " START EXPLOITATION PHASE " + "*" * 19)
        warnings.filterwarnings("ignore")
        if self.surrogate_type == "gp":
            K = self.s_kernel(**self.s_kernel_kwargs)
            s_model = GaussianProcessRegressor(kernel=K, **self.surrogate_kwargs)
        elif self.surrogate_type == "rf":
            s_model = RandomForestRegressor(**self.surrogate_kwargs)
        else:
            s_model = ExtraTreesRegressor()
        
        score = self.classifier.score(self.x_tot, self.labels)
        y_min = np.min(self.y_feasible)
        n_feasible = self.y_feasible.size
        print("  Iter       f(x)           Best      #feasible  Class.Score")
        for i in range(self.iter_exploitation):
            ## setting sampling-dependent initial guess for length scale
            ## seems to improve hyperparameter optimization in surrogate model
            s_model.kernel.set_params(length_scale=np.sqrt(0.5 / self.gamma))
            s_model.fit(self.x_feasible, self.y_feasible)
            params = {"model":s_model, "classifier":self.classifier,
                      "bounds":self.bounds, "n_sampling":self.n_init_exploitation,
                      "max_iter":self.local_opt_maxsteps}
            next_x = self.exploitation_search(f=self.acquisition_func, args=params, sampler=self.sampler)
            value = self.f(next_x[0])
            y_min = min(y_min, value)
            if np.isnan(value):
                print("  {0:4d}   {1:>8f}       {2: 7.5e}   {3:5d}        {4:>5.3f}".format(i+1,value,y_min,n_feasible,score))
                new_label = 1
            else:
                n_feasible += 1
                print("  {0:4d}   {1: 7.5e}   {2: 7.5e}   {3:5d}        {4:>5.3f}".format(i+1,value,y_min,n_feasible,score))
                new_label = 0
                self.y_feasible = np.append(self.y_feasible, value)
                self.x_feasible = np.concatenate((self.x_feasible, next_x))
            self.x_tot = np.concatenate((self.x_tot, next_x))
            self.labels = np.append(self.labels, new_label)
            
            ### Update surrogate model every n iterations or if point unfeasible!
            refit_feas = (i%self.retrain_classifier_if_feasible == 0) and (len(np.unique(self.labels)) > 1)
            if np.isnan(value) or refit_feas:
                self.gamma = 0.5 / (np.sqrt(self.bounds.shape[0]) * np.var(self.x_tot, axis=0))
                self.classifier = self.init_classifier()
                self.classifier.fit(self.x_tot, self.labels)
                score = self.classifier.score(self.x_tot, self.labels)
            if self.log_models:
                self.c_score.append(score)
                self.surrogates.append(s_model)
                self.classifiers.append(self.classifier)
            self.y_tot = np.append(self.y_tot, value)
        print("*" * 20 + " END EXPLOITATION PHASE " + "*" * 20 + "\n")
        self.surrogate = s_model
    
    def post_optimize(self, n_restarts=20, n_opt=200):
        """
        Post-optimize with tighter threshold and increased number of steps on
        surrogate surface starting from current best estimate for minimum.
        Targeted sampling improves surrogate model around estimated minimum.
        
        Parameters
        ----------
            n_restarts: int, default: 20
                number of restarts
            n_opt: int, default: 200
                maximum number of steps in local optimization runs
        
        """
        print("*" * 17 + " START POST-OPTIMIZATION PHASE " + "*" * 16)
        min_idx = np.argmin(self.y_feasible)
        x_min = self.x_feasible[min_idx]
        y_min = self.y_feasible[min_idx]
        n_feasible = self.y_feasible.size
        for i in range(n_restarts):
            self.surrogate.kernel.set_params(length_scale=np.sqrt(0.5 / self.gamma))
            self.surrogate.fit(self.x_feasible, self.y_feasible)
            self.gamma = 0.5 / (np.sqrt(self.bounds.shape[0]) * np.var(self.x_tot, axis=0))
            self.classifier = self.init_classifier()
            self.classifier.fit(self.x_tot, self.labels)
            score = self.classifier.score(self.x_tot, self.labels)
            
            params = {"model":self.surrogate, "classifier":self.classifier}
            res = local_opt(self.acquisition_func, params, self.bounds, x_min, n_iter=n_opt)
            next_x = res.get("x").reshape(1,-1)
            value = self.f(next_x[0])
            if np.isnan(value):
                print("  {0:4d}   {1:>8f}       {2: 7.5e}   {3:5d}        {4:>5.3f}".format(i+1,value,y_min,n_feasible,score))
                new_label = 1
            else:
                if value < y_min: x_min, y_min = next_x, value
                n_feasible += 1
                print("  {0:4d}   {1: 7.5e}   {2: 7.5e}   {3:5d}        {4:>5.3f}".format(i+1,value,y_min,n_feasible,score))
                new_label = 0
                self.y_feasible = np.append(self.y_feasible, value)
                self.x_feasible = np.concatenate((self.x_feasible, next_x))
            self.x_tot = np.concatenate((self.x_tot, next_x))
            self.labels = np.append(self.labels, new_label)
        print("*" * 18 + " END POST-OPTIMIZATION PHASE " + "*" * 17 + "\n")
        
    def generate_result(self):
        """
        Generate result
        
        Returns
        -------
            res: dict containing
                'x':    current best estimate of location of minimum
                'fun':  value at current estimate for minimum
                'xs':   all x sampled
                'funs': all corresponding function values
                'xs_feasible':   all x found in feasible region
                'funs_feasible': corresponding function values
        
        """
        min_idx = np.argmin(self.y_feasible)
        x = self.x_feasible[min_idx]
        fun = self.y_feasible[min_idx]
        res = {"x":x, "fun":fun, "xs":self.x_tot, "funs":self.y_tot, "xs_feasible":self.x_feasible,
               "funs_feasible":self.y_feasible}
        return res

    def gap_metric(self, optimum_value=0):
        """
        Generate gap metric along optimization process defined by
            |current min(f) - init min(f)| / |reference min(f) - init min(f)|
        where init min(f) is the minimum from the initial (quasi-)random sampling
        
        Returns
        -------
            gap_metric: array-like
                gap_metric as defined above for each iteration
        
        """
        init_opt = current_opt = np.nanmin(self.y_tot[0:self.n_init_grid])
        gap_metric = []
        for i in range(0, len(self.y_tot)):
            current_opt = min(current_opt, self.y_tot[i])
            gap = abs(current_opt - init_opt) / abs(optimum_value - init_opt)
            gap_metric.append(gap)
        return gap_metric




class GPCBO(ABCBO):
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init_grid=10, iter_exploration=60,
                 iter_exploitation=30, surrogate_type="GP", surrogate_kwargs={},
                 surrogate_kernel="RBF", surrogate_kernel_kwargs={},
                 classifier_kwargs={}, classifier_kernel="RBF",
                 classifier_kernel_kwargs={}, sampler="sobol", sampler_kwargs={},
                 log_models=False, noise=None, seed=42, n_init_exploration=32,
                 n_init_exploitation=32, retrain_classifier_if_feasible=True,
                 smooth_acquisition=True, local_opt=True, local_opt_maxsteps=20,
                 debug=False):
        """
        Bayesian Optimization with constraints from Gaussian process classifier.
        
        Parameters
        ----------
            f: callable
                objective function to minimize
            bounds: array-like, shape (ndim, 2), default: [(0,1), (0,1)]
                search domain for optimization
                !TODO!: turn this into a moving window (= local search radius)
                        for unbound optimization problems!
            n_init_grid: int, default: 10
                # of points in initial, uninformed sampling of search space
            iter_exploration: int, default: 60
                # of iterations in exploration phase (optimize coverage)
            iter_exploitation: int, default: 30
                # of iterations in exploitation phase (optimize acquisition function)
            surrogate_type: str, default: "GP"
                type of surrogate model (GP: Gaussian Process or RF: Random Forest)
            surrogate_kwargs: dict, default: {}
                extra keyword arguments for surrogate model
            surrogate_kernel: str, default: "RBF"
                kernel for GP surrogate model
            surrogate_kernel_kwargs: dict, default: {}
                keyword arguments for kernel in surrogate model (GP models only)
            classifier_kwargs: dict, default: {}
                keyword arguments for classifier model
            classifier_kernel: str, default: "RBF"
                kernel for GP classifier
            classifier_kernel_kwargs: dict, default: {}
                keyword arguments for kernel in GP classifier
            sampler: str, default: "sobol"
                algorithm for (quasi-)random sampling of search space
                "lhs": latin hyper cube
                "sobol": Sobol sequence
            sampler_kwargs: dict, default: {"criterion":"maximin", "iterations":100} for lhs, {} for sobol
                keyword arguments for sampler
            log_models: bool, default: False
                save surrogate and classifier models at every step
            noise: default: None
                whether data is noisy (add white-noise kernel if so)
            seed: int, default: 42
                random seed for (quasi-)random sampling
            n_focus: int, default: 10
                number of focusing steps (localized search around current minimum)
            n_resample: int, default: 5
                number of resampling iterations for (quasi-)random search at every step
            n_init_exploration: int, default: 32
                number of initial guesses for exploration phase (optimize coverage)
            n_init_exploitation: int, default: 32
                number of initial guesses for exploitation phase (optimize acquisition function)
            retrain_classifier_if_feasible: bool or int, default: True
                retrain the classifier also when encountering feasible points, otherwise
                only retrain if unfeasible point is encountered
                if int, retrain every <retrain_classifier_if_feasible> iterations
            smooth_acquisition: bool, default: True
                whether to smoothly interpolate acquisition function at estimated
                boundaries (i.e., use probability of feasibility before decision function)
            local_opt: bool, default: True
                flag for performing exploitation with multi-start local optimization
                otherwise quasi-random search with focusing
            local_opt_maxsteps: int, default: 20
                maximum number of steps in local optimization during exploitation
            debug: bool, default: False
                flag for more verbose output for each phase
        
        """
        super(GPCBO, self).__init__(f, bounds=bounds, n_init_grid=n_init_grid,
                iter_exploration=iter_exploration, iter_exploitation=iter_exploitation,
                surrogate_type=surrogate_type, surrogate_kernel=surrogate_kernel,
                surrogate_kernel_kwargs=surrogate_kernel_kwargs,
                classifier_kwargs=classifier_kwargs, sampler=sampler,
                sampler_kwargs=sampler_kwargs, log_models=log_models,
                noise=noise, seed=seed, n_init_exploration=n_init_exploration,
                n_init_exploitation=n_init_exploitation,
                retrain_classifier_if_feasible=retrain_classifier_if_feasible,
                debug=debug)
        try:
            self.c_kernel = eval("kernels."+classifier_kernel)
            self.c_kernel_opt = classifier_kernel_kwargs
        except:
            raise ValueError("`classifier_kernel` has to be one of `sklearn.gaussian_process.kernels`!")
        if smooth_acquisition or local_opt:
            self.acquisition_func = acquisition_function_smooth
        else:
            self.acquisition_func = acquisition_function_binary
        if local_opt:
            self.exploitation_search = focus_search_local_opt
            self.local_opt_maxsteps = local_opt_maxsteps
        else:
            self.exploitation_search = focus_search_parallel
            

    def init_classifier(self, **kwargs):
        return GPCFlex(kernel=self.c_kernel(length_scale=1./np.sqrt(self.bounds.shape[0]), **self.c_kernel_opt),
                       ## let sklearn's hyperparameter optimization figure out the length scale
                       ## sampling-dependent initial guess seems to hinder optimization
                       ## (lack of connection between input and output variance for binary classifier!)
                       #length_scale=np.sqrt(0.5 / self.gamma), **self.c_kernel_opt),
                       **self.classifier_kwargs)
    

class SVMCBO(ABCBO):
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init_grid=10, iter_exploration=60,
                 iter_exploitation=30, surrogate_type="GP", surrogate_kwargs={},
                 surrogate_kernel="RBF", surrogate_kernel_kwargs={},
                 classifier_kwargs={}, sampler="sobol", sampler_kwargs={},
                 log_models=False, noise=None, seed=42, n_init_exploration=32,
                 n_init_exploitation=1000, retrain_classifier_if_feasible=True,
                 debug=False):
        """
        Bayesian optimization with constraints from support vector machine classifier.
        
        Parameters
        ----------
            f: callable
                objective function to minimize
            bounds: array-like, shape (ndim, 2), default: [(0,1), (0,1)]
                search domain for optimization
                !TODO!: turn this into a moving window (= local search radius)
                        for unbound optimization problems!
            n_init_grid: int, default: 10
                # of points in initial, uninformed sampling of search space
            iter_exploration: int, default: 80
                # of iterations in exploration phase (optimize coverage)
            iter_exploitation: int, default: 30
                # of iterations in exploitation phase (optimize acquisition function)
            surrogate_type: str, default: "GP"
                type of surrogate model (GP: Gaussian Process or RF: Random Forest)
            surrogate_kwargs: dict, default: {}
                extra keyword arguments for surrogate model
            surrogate_kernel: str, default: "RBF"
                kernel for GP surrogate model
            surrogate_kernel_kwargs: dict, default: {}
                keyword arguments for kernel in surrogate model (GP models only)
            classifier_kwargs: dict, default: {}
                keyword arguments for classifier model
            sampler: str, default: "sobol"
                algorithm for (quasi-)random sampling of search space
                "lhs": latin hyper cube
                "sobol": Sobol sequence
            sampler_kwargs: dict, default: {"criterion":"maximin", "iterations":100} for lhs, {} for sobol
                keyword arguments for sampler
            log_models: bool, default: False
                save surrogate and classifier models at every step
            noise: default: None
                whether data is noisy (add white-noise kernel if so)
            seed: int, default: 42
                random seed for (quasi-)random sampling
            n_focus: int, default: 10
                number of focusing steps (localized search around current minimum)
            n_resample: int, default: 5
                number of resampling iterations for (quasi-)random search at every step
            n_init_exploration: int, default: 32
                number of initial guesses for exploration phase (optimize coverage)
            n_init_exploitation: int, default: 32
                number of initial guesses for exploitation phase (optimize acquisition function)
            retrain_classifier_if_feasible: bool or int, default: True
                retrain the classifier also when encountering feasible points, otherwise
                only retrain if unfeasible point is encountered
                if int, retrain every <retrain_classifier_if_feasible> iterations
            debug: bool, default: False
                flag for more verbose output for each phase
        
        """
        super(SVMCBO, self).__init__(f, bounds=bounds, n_init_grid=n_init_grid,
                iter_exploration=iter_exploration, iter_exploitation=iter_exploitation,
                surrogate_type=surrogate_type, surrogate_kernel=surrogate_kernel,
                surrogate_kernel_kwargs=surrogate_kernel_kwargs,
                classifier_kwargs=classifier_kwargs, sampler=sampler,
                sampler_kwargs=sampler_kwargs, log_models=log_models,
                noise=noise, seed=seed, n_init_exploration=n_init_exploration,
                n_init_exploitation=n_init_exploitation,
                retrain_classifier_if_feasible=retrain_classifier_if_feasible,
                debug=debug)
        self.local_opt_maxsteps = 1
        self.acquisition_func = acquisition_function_binary
        self.exploitation_search = focus_search_parallel
    
    def init_classifier(self, with_proba=False):
        return SVCFlex(gamma=self.gamma, probability=with_proba, **self.classifier_kwargs)
    



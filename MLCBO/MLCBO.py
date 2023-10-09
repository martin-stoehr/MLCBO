import numpy as np
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from skopt.sampler import Sobol, Lhs
from .my_gp import *
from .my_rf import *
from .mlcbo_utils import *
from .focus_search import *


sampler_settings = {"lhs"   : {'criterion':'maximin', 'iterations':100},
                    "sobol" : {}}

class ABCBO(ABC):
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init=10, iter_phase1=60,
                 iter_phase2=30, surrogate_type="GP", surrogate_kwargs={},
                 surrogate_kernel="RBF", surrogate_kernel_kwargs={}, 
                 classifier_kwargs={}, sampler='lhs', sampler_kwargs={},
                 log_models=False, noise=None, seed=42):
        self.classifier_kwargs = classifier_kwargs
        if surrogate_type == "GP":
            try:
                self.s_kernel = eval("kernels."+surrogate_kernel+"(**surrogate_kernel_kwargs)")
            except:
                raise ValueError("`surrogate_kernel` has to be one of `sklearn.gaussian_process.kernels`!")
        self.surrogate_kwargs = surrogate_kwargs
        self.gamma, self.classifier, self.surrogate = None, None, None
        self.noise = noise
        self.x_tot, self.y_tot = [], []
        self.x_feasible, self.y_feasible = [], []
        self.labels, self.n_init = [], n_init
        self.iter_phase1 = iter_phase1
        self.iter_phase2 = iter_phase2
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
        

    @abstractmethod
    def init_classifier(self):
        pass

    def init_opt(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = np.array(self.sampler.generate(self.bounds, self.n_init, random_state=self.seed))
        self.labels = np.zeros((x.shape[0],))
        self.y_feasible = list()
        i = 0
        while i < x.shape[0]:
            print("*** Iteration ", i)
            print("==> Points to evaluate: ", x[i,])
            function_evaluation = self.f(x[i,])

            if np.isnan(function_evaluation):
                print("---------- NO FUNCTION EVALUATION ----------")
                self.labels[i] = 1
            else:
                print("==> Function evaluation:")
                print(function_evaluation)
                self.labels[i] = 0
                self.y_feasible = self.y_feasible + [function_evaluation]

            self.y_tot = self.y_tot + [function_evaluation]
            i = i + 1
            print(" --> Number of feasible points:", len(self.y_feasible))
        self.x_tot = x.tolist()
        self.gamma = 1 / (self.bounds.shape[0] * np.var(self.x_tot))
        self.classifier = self.init_classifier()

    def phase1(self):
        self.classifier.fit(self.x_tot, self.labels)
        score = self.classifier.score(self.x_tot, self.labels)
        self.c_score.append(score)
        print('*'*70)
        print("> Classifier score (init): {}".format(score))
        print('*'*70)
        if self.log_models: self.classifiers.append(self.classifier)
        print(self.classifier.predict(self.x_tot))
        print(self.labels)
        
        print("Start phase 1!!!")
        for i in np.arange(0, self.iter_phase1):
            print('*' * 70)
            print("Iteration", i, "in phase 1... (feasible points:", len(self.y_feasible), ")")
            print("[Current best value: ", np.min(self.y_feasible),"]")
            print("Computing the new point...")
            x_new = nextPointPhase1(sampledPoints=self.x_tot, classifier=self.classifier,
                                    gamma=np.var(self.x_tot), sampler=self.sampler,
                                    dimensions_test=self.bounds)
            print("Updating the design...")
            self.x_tot = self.x_tot + x_new.tolist()
            print("==> Points to evaluate: ", x_new)
            function_evaluation = self.f(x_new[0])
            print("-------------------")
            print(function_evaluation)
            print("-------------------")
            if np.isnan(function_evaluation):
                print("==> Function evaluation: ", function_evaluation)
                new_label = 1
            else:
                new_label = 0
                self.y_feasible = self.y_feasible + [function_evaluation]

            self.y_tot = self.y_tot + [function_evaluation]
            self.labels = np.concatenate((self.labels, np.array([new_label])))

            print("Updating the estimated feasible region...")
            self.gamma = 1 / (self.bounds.shape[0] * np.var(self.x_tot))
            self.classifier = self.init_classifier()
            self.classifier.fit(self.x_tot, self.labels)
            if self.log_models:
                score = self.classifier.score(self.x_tot, self.labels)
                print("> Classifier score: {}".format(score))
                print('*' * 70)
                self.c_score.append(score)
                self.classifiers.append(self.classifier)
        self.x_feasible = np.array(self.x_tot)[np.where(self.labels==0)[0],].tolist()
    
    def phase2(self):
        """
        Run Bayesian optimization with on-the-fly updates of SVM estimator for feasible region.
        """
        warnings.filterwarnings("ignore")
        if self.surrogate_type == "gp":
            s_model = GaussianProcessRegressor(kernel=self.s_kernel, **self.surrogate_kwargs)
        elif self.surrogate_type == "rf":
            s_model = RandomForestRegressor(**self.surrogate_kwargs)
        else:
            s_model = ExtraTreesRegressor()
        for iter_bo in range(self.iter_phase2):
            ymin_arg = np.argmin(self.y_feasible)
            ymin_i = self.y_feasible[ymin_arg]
            xmin_i = self.x_feasible[ymin_arg]
            print('*' * 70)
            print("> Iteration {} in phase 2... (feasible points: {})".format(iter_bo, len(self.y_feasible)))
            print("[Current best value: ", ymin_i,"]")
            x_, y_ = self.x_feasible, self.y_feasible
            s_model.fit(x_, y_)
            params = {'model':s_model, 'classifier':self.classifier,
                      'bounds':self.bounds, 'n_sampling':10000, 'best_x':[xmin_i]}
            next_x = focus_search_parallel(f=self.acquisition_func, args=params, sampler=self.sampler)
            value = self.f(next_x)
            print("f({}) = {}".format(next_x, value))
            new_label = 1 if np.isnan(value) else 0
            ### Add classification label new point
            self.x_tot = self.x_tot + [next_x.tolist()]
            self.labels = np.concatenate((self.labels, np.array([new_label])))
            
            ### Update surrogate model in case unfeasible point sampled!
            if np.isnan(value):
                print("**** Retraining bounds! ***")
                self.gamma = 1 / (self.bounds.shape[0] * np.var(self.x_tot))
                self.classifier = self.init_classifier()
                self.classifier.fit(self.x_tot, self.labels)
            else:
                self.y_feasible = self.y_feasible + [value]
                self.x_feasible = self.x_feasible + [next_x.tolist()]
            if self.log_models:
                score = self.classifier.score(self.x_tot, self.labels)
                print("> Classifier score: {}".format(score))
                self.c_score.append(score)
                self.surrogates.append(s_model)
                self.classifiers.append(self.classifier)
            print('*' * 70)
            self.y_tot = self.y_tot + [value]
        self.surrogate = s_model

    ## Generate results of experiment
    def generate_result(self):
        x = self.x_feasible[np.argmin(self.y_feasible)]
        fun = np.min(self.y_feasible)
        return {"x": x, "fun": fun, "xs": self.x_tot, "funs": self.y_tot, "xs_feasible": self.x_feasible,
                "funs_feasible": self.y_feasible}

    ## Generate gap metric for check the progress of the optimizer into optimization process
    def gap_metric(self, optimum_value=0):
        y_tmp = [9999999 if np.isnan(x) else x for x in self.y_tot]
        current_opt = np.min(y_tmp[0:self.n_init])
        init_opt = np.min(y_tmp[0:self.n_init])
        gap_metric = []
        for i in range(0, len(y_tmp)):
            if current_opt > y_tmp[i]:
                current_opt = y_tmp[i]
            gap = abs(current_opt - init_opt) / abs(optimum_value - init_opt)
            gap_metric = gap_metric + [gap]
        return gap_metric




class GPCBO(ABCBO):
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init=10, iter_phase1=60,
                 iter_phase2=30, surrogate_type="GP", surrogate_kwargs={},
                 surrogate_kernel="RBF", surrogate_kernel_kwargs={},
                 classifier_kwargs={}, classifier_kernel="RBF",
                 classifier_kernel_kwargs={}, sampler='lhs', sampler_kwargs={},
                 log_models=False, noise=None, seed=42):
        super(GPCBO, self).__init__(f, bounds=bounds, n_init=n_init,
                iter_phase1=iter_phase1, iter_phase2=iter_phase2,
                surrogate_type=surrogate_type, surrogate_kernel=surrogate_kernel,
                surrogate_kernel_kwargs=surrogate_kernel_kwargs,
                classifier_kwargs=classifier_kwargs, sampler=sampler,
                sampler_kwargs=sampler_kwargs, log_models=log_models,
                noise=noise, seed=seed)
        try:
            self.c_kernel = eval("kernels."+classifier_kernel)
            self.c_kernel_opt = classifier_kernel_kwargs
        except:
            raise ValueError("`classifier_kernel` has to be one of `sklearn.gaussian_process.kernels`!")
        self.acquisition_func = acquisition_function_GPC

    def init_classifier(self):
        return GPC(kernel=self.c_kernel(**self.c_kernel_opt),#self.gamma, 
                   **self.classifier_kwargs)
    

class SVMCBO(ABCBO):
    def __init__(self, f, bounds=[[0, 1], [0, 1]], n_init=10, iter_phase1=60,
                 iter_phase2=30, surrogate_type="GP", surrogate_kwargs={},
                 surrogate_kernel="RBF", surrogate_kernel_kwargs={},
                 classifier_kwargs={}, sampler='lhs', sampler_kwargs={},
                 log_models=False, noise=None, seed=42):
        super(SVMCBO, self).__init__(f, bounds=bounds, n_init=n_init,
                iter_phase1=iter_phase1, iter_phase2=iter_phase2,
                surrogate_type=surrogate_type, surrogate_kernel=surrogate_kernel,
                surrogate_kernel_kwargs=surrogate_kernel_kwargs,
                classifier_kwargs=classifier_kwargs, sampler=sampler,
                sampler_kwargs=sampler_kwargs, log_models=log_models,
                noise=noise, seed=seed)
        self.acquisition_func = acquisition_function_SVC
    
    def init_classifier(self):
        return SVC(gamma=self.gamma, **self.classifier_kwargs)
    



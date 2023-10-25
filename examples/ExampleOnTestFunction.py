import matplotlib.pyplot as plt
from MLCBO import SVMCBO, GPCBO
from MLCBO.mlcbo_utils import estimate_p_nonfeasible
from Optimization_Test_Functions.test_function_suite import *
from time import time


CBO = GPCBO

test_functions = [rosenbrock_fun_c_2, gomez_levi_c,
                  mishra_bird_c, drop_wave_c]

def boundary_rosenbrock_fun_c_2(x):
    b1 = abs((x[0] - 1)**3 - x[1] + 1) <= 1e-2
    b2 = abs(x[0] + x[1] - 2.) <= 1e-2
    b3 = x[0] < 1.000001
    return (b1 or b2) and b3

def boundary_mishra_bird_c(x):
    return abs((x[0]+5)**2 + (x[1]+5)**2 - 25) <= 1e-2

def boundary_gomez_levi_c(x):
    return abs(2*np.sin(2*np.pi*x[1])**2-np.sin(4*np.pi*x[0]) - 1.5) <= 1e-2

def boundary_drop_wave_c(x):
    return abs(.1 - (1 + np.cos(12.* np.sqrt(x[0]**2 + x[1]**2))) / (0.5*(x[0]**2 + x[1]**2) + 2.)) <= 1e-2

global_min = {"rosenbrock_fun_c_2":0.,
              "gomez_levi_c":0.0000000005,
              "mishra_bird_c":-106.7645367,
              "drop_wave_c":0.}

f_names = {"rosenbrock_fun_c_2":"Rosenbrock",
           "gomez_levi_c":"GomezLevi",
           "mishra_bird_c":"MishraBird",
           "drop_wave_c":"DropWave"}

options = {'iter_exploration':60, 'iter_exploitation':60,
           'n_init_grid':32, 'n_init_exploitation':64,
           'sampler':'sobol', 'local_opt':True, 'debug':False}

surrogate_kernels = ["RBF", "Matern"]
classifier_types = ["SVM"] if isinstance(CBO, SVMCBO) else ["RBF", "Matern"]

x_opt = {k:{} for k in f_names.values()}
y_opt = {k:{} for k in f_names.values()}
t_tot = {k:{} for k in f_names.values()}
gap_metric = {k:{} for k in f_names.values()}

for test_f in test_functions:
    fun_name = test_f.__name__
    optimum_for_gap = global_min[fun_name]
    for skernel in surrogate_kernels:
        for class_type in classifier_types:
            ## CBO with RBF surrogate
            skernel_opts = {'nu':5/2} if skernel == "Matern" else {}
            if isinstance(CBO, SVMCBO):
                cbo_opt = CBO(f=test_f, surrogate_type="GP", surrogate_kernel=skernel,
                              surrogate_kernel_kwargs=skernel_opts, **options)
            else:
                ckernel_opts = {'nu':5/2} if class_type == "Matern" else {}
                cbo_opt = CBO(f=test_f, surrogate_type="GP", surrogate_kernel=skernel,
                              surrogate_kernel_kwargs=skernel_opts, 
                              classifier_kernel=class_type,
                              classifier_kernel_kwargs=ckernel_opts, **options)
            t0 = time()
            ### RUN OPTIMIZATION: Initialise, explore, focus, post-opt
            cbo_opt.init_opt()
            cbo_opt.exploration()
            cbo_opt.exploitation()
            if class_type != "SVM": cbo_opt.post_optimize()
            t = time() - t0
            res = cbo_opt.generate_result()
            x_fin = scale_to_domain(res.get('x'), fun_name)
            y_fin = res.get('fun')
            gap_metric[f_names[fun_name]][skernel+","+class_type] = cbo_opt.gap_metric(optimum_value=optimum_for_gap)
            x_opt[f_names[fun_name]][skernel+","+class_type] = x_fin
            y_opt[f_names[fun_name]][skernel+","+class_type] = y_fin
            t_tot[f_names[fun_name]][skernel+","+class_type] = t
            
            bounds = get_bounds(fun_name)
            xg = np.arange(bounds[0,0], bounds[0,1]+0.01, 0.01)
            yg = np.arange(bounds[1,0], bounds[1,1]+0.01, 0.01)
            X, Y = np.meshgrid(xg, yg)
            constr, grid = [], []
            p_feas, n_feas = np.zeros_like(X.T), np.zeros_like(X.T)
            surrogate = cbo_opt.surrogate
            classifier = cbo_opt.classifier
            for i, x_i in enumerate(xg):
                for j, y_j in enumerate(yg):
                    p2 = np.array([x_i,y_j])
                    p = scale_from_domain(p2, fun_name)
                    grid.append(p)
                    if eval("boundary_"+fun_name+"(p2)"): constr.append(p2)
                    n_feas[i,j] = np.nan if classifier.predict([p]) else 1
                    p_feas[i,j] = 1 - estimate_p_nonfeasible(p, classifier)
            grid = np.array(grid)
            f_func = surrogate.predict(grid, return_std=False)
            f_func -= np.min(f_func) - 1e-12
            f_func = np.array(f_func).reshape((len(xg), len(yg)))
            f_func *= n_feas
            f_acqu = cbo_opt.acquisition_func(grid, {"model":surrogate, "classifier":classifier})
            f_acqu -= np.min(f_acqu) - 1e-12
            f_acqu = np.array(f_acqu).reshape((len(xg), len(yg)))

            ax1 = plt.subplot2grid((2,2), (0,0))
            if fun_name == "rosenbrock_fun_c_2":
                ax1.pcolormesh(X, Y, f_func.T, norm='log')
            else:
                ax1.pcolormesh(X, Y, f_func.T)
            ax1.plot(*get_minimum(fun_name), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
            ax1.plot(x_fin[0], x_fin[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="ML-CBO min")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_title("surrogate model")

            ax2 = plt.subplot2grid((2,2), (0,1))
            ax2.pcolormesh(X, Y, n_feas.T, cmap='Greys', vmin=0, vmax=1.75)   
            for p_c in constr: ax2.plot(*p_c, ls='', marker='.', c='k', ms=1)
            ax2.plot(*get_minimum(fun_name), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
            ax2.plot(x_fin[0], x_fin[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="ML-CBO min")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_title("estimated feasible region")

            ax3 = plt.subplot2grid((2,2), (1,0))
            if fun_name == "rosenbrock_fun_c_2":
                ax3.pcolormesh(X, Y, f_acqu.T, norm='log')
            else:
                ax3.pcolormesh(X, Y, f_acqu.T)
            ax3.plot(*get_minimum(fun_name), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
            ax3.plot(x_fin[0], x_fin[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="ML-CBO min")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_title("acquisition function")

            ax4 = plt.subplot2grid((2,2), (1,1))
            ax4.pcolormesh(X, Y, p_feas.T)
            for p_c in constr: ax4.plot(*p_c, ls='', marker='.', c='w', ms=1)
            ax4.plot(*get_minimum(fun_name), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
            ax4.plot(x_fin[0], x_fin[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="ML-CBO min")
            ax4.set_xlabel("x")
            ax4.set_ylabel("y")
            ax4.set_title("P(feasible)")

            fig = plt.gcf()
            fig.subplots_adjust(left=0.075, top=0.94, right=0.99, bottom=0.09, hspace=0.4, wspace=0.25)
            fig.savefig(f_names[fun_name]+"_"+class_type+"-CBO-"+skernel+"_analysis.png", dpi=600)
            plt.close()
        
    
    lstyles = ["-", "--", ":", "-."]
    cols = ["tab:blue", "darkorange", "tab:green"]
    marks = ['o', 'd', 'v', '<', '>', '^', '*']
    i = 0
    for skernel in surrogate_kernels:
        for class_type in classifier_types:
            clab = "GP("+class_type+")" if class_type != "SVM" else class_type
            plt.plot(gap_metric[f_names[fun_name]][skernel+","+class_type], c=cols[i%3], ls=lstyles[i%4], label=clab+"-CBO("+skernel+")")
            i += 1
    plt.ylim(0.0,1.1)
    plt.ylabel("Gap Metric")
    plt.xlabel("Iteration")
    plt.title("Comparison gap metric on {} test function".format(fun_name))
    plt.legend(loc="best")
    plt.savefig(class_type+"-CBO_"+f_names[fun_name]+"-metric.png", dpi=600)
    plt.close()
    
    bounds = get_bounds(fun_name)
    xg = np.arange(bounds[0,0], bounds[0,1]+0.01, 0.01)
    yg = np.arange(bounds[1,0], bounds[1,1]+0.01, 0.01)
    X, Y = np.meshgrid(xg, yg)
    Z = np.zeros_like(X.T)
    for i, x_i in enumerate(xg):
        for j, y_j in enumerate(yg):
            p2 = np.array([x_i,y_j])
            p = scale_from_domain(p2, fun_name)
            Z[i,j] = test_f(p)
    
    if fun_name == "rosenbrock_fun_c_2":
        plt.pcolormesh(X, Y, Z.T, norm='log')
        plt.contour(X, Y, Z.T, levels=18, linewidths=0.5, colors='0.5', norm='log')
    else:
        plt.pcolormesh(X, Y, Z.T)
        plt.contour(X, Y, Z.T, levels=18, linewidths=0.5, colors='0.5')
    plt.plot(*get_minimum(fun_name), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
    i = 0
    for skernel in surrogate_kernels:
        for class_type in classifier_types:
            clab = "GP("+class_type+")" if class_type != "SVM" else class_type
            x1, x2 = x_opt[f_names[fun_name]][skernel+","+class_type]
            plt.plot(x1, x2, c=cols[i%3], ls='', marker=marks[i%7], mec='k', mew=0.5, label=clab+"-CBO("+skernel+")")
            i += 1
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='best')
    plt.savefig(f_names[fun_name]+"_"+class_type+"-CBO_surface.png", dpi=600)
    plt.close()
    
for test_f in test_functions:
    fun_name == test_f.__name__
    print("\n\n### " + fun_name + " ###")
    for skernel in surrogate_kernels:
        for class_type in classifier_types:
            clab = "GP("+class_type+")" if class_type != "SVM" else class_type
            print("\n"+clab+"-CBO("+skernel+")")
            x = x_opt[f_names[fun_name]][skernel+","+class_type]
            y = y_opt[f_names[fun_name]][skernel+","+class_type]
            t = t_tot[f_names[fun_name]][skernel+","+class_type]
            print(f"Optimum point: {x}")
            print(f"Optimal value: {y}")
            print(f"Total time:    {t}")


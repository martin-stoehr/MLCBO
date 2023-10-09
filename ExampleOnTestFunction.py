import matplotlib.pyplot as plt
from SVMCBO import SVMCBO, GPCBO
from SVMCBO.svmcbo_utils import estimate_p_nonfeasible
from Optimization_Test_Functions.test_function_suite import *
from time import time


CBO = SVMCBO


def boundary_rosenbrock_fun_c_2(x):
    b1 = abs((x[0] - 1)**3 - x[1] + 1) <= 1e-2
    b2 = abs(x[0] + x[1] - 2.) <= 1e-2
    b3 = x[0] < 1.000001
    return (b1 or b2) and b3

def boundary_mishra_bird_c(x):
    return abs((x[0]+5)**2 + (x[1]+5)**2 - 25) <= 1e-2

def boundary_gomez_levi_c(x):
    return abs(2*np.sin(2*np.pi*x[1])**2-np.sin(4*np.pi*x[0]) - 1.5) <= 1e-2


# Example optimization Constrained Rosenbrock test function ############################################
test_f = rosenbrock_fun_c_2
optimum_for_gap = 0.
options = {'iter_phase1':20, 'iter_phase2':50,
           'n_init':32, 'sampler':'sobol'}

## Experiment with SVM-CBO
exp_cbo_gpr = CBO(f=test_f, surrogate_type="GP", **options)
class_type = "SVM" if isinstance(exp_cbo_gpr, SVMCBO) else "GP"
exp_cbo_gpr.init_opt()
t0 = time()
exp_cbo_gpr.phase1()
exp_cbo_gpr.phase2()
t1_GPR = time() - t0
res1 = exp_cbo_gpr.generate_result()
x1_1 = scale_to_domain(res1.get('x'), test_f.__name__)
y1_1 = res1.get('fun')
gap_metric_cbo_GPR = exp_cbo_gpr.gap_metric(optimum_value=optimum_for_gap)
print("Classes: ",exp_cbo_gpr.classifier.classes_)
print("")

## Experiment with SVM-CBO Matern-3/2 kernel
exp_cbo = CBO(f=test_f, surrogate_type="GP", surrogate_kernel="Matern", 
                    surrogate_kernel_kwargs={'nu':5/2}, **options)
exp_cbo.init_opt()
t0 = time()
exp_cbo.phase1()
exp_cbo.phase2()
t1_GPM = time() - t0
res2 = exp_cbo.generate_result()
x2_1 = scale_to_domain(res2.get('x'), test_f.__name__)
y2_1 = res2.get('fun')
gap_metric_cbo_GPM = exp_cbo.gap_metric(optimum_value=optimum_for_gap)

## Experiment with SVM-CBO_RF
exp_cbo_rf = CBO(f=test_f, surrogate_type="RF", **options)
exp_cbo_rf.init_opt()
t0 = time()
exp_cbo_rf.phase1()
exp_cbo_rf.phase2()
t1_RF = time() - t0
res3 = exp_cbo_rf.generate_result()
x3_1 = scale_to_domain(res3.get('x'), test_f.__name__)
y3_1 = res1.get('fun')
gap_metric_cbo_rf = exp_cbo_rf.gap_metric(optimum_value=optimum_for_gap)

## Comparison gap metric between SVMCBO and SVMCBO_RF

f1 = test_f.__name__

plt.plot(gap_metric_cbo_GPR, label="SVM-CBO GP(RBF)")
plt.plot(gap_metric_cbo_GPM, ls='--', label="SVM-CBO GP(Matern-5/2)")
plt.plot(gap_metric_cbo_rf, ls=':', label="SVM-CBO RF")
plt.ylim(0.0,1.1)
plt.ylabel("Gap Metric")
plt.xlabel("Iteration")
plt.title("Comparison gap metric on {} test function".format(f1))
plt.legend(loc="best")
plt.savefig(class_type+"-CBO_Rosenbrock-metric.png", dpi=600)
plt.close()

bounds = get_bounds(test_f.__name__)
x = np.linspace(bounds[0,0], bounds[0,1], 750)
y = np.linspace(bounds[1,0], bounds[1,1], 750)
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X.T)
f_func, std_fn = np.zeros_like(X.T), np.zeros_like(X.T)
p_feas, n_feas = np.zeros_like(X.T), np.zeros_like(X.T)
constr = []
for i, x_i in enumerate(x):
    for j, y_j in enumerate(y):
        p2 = np.array([x_i,y_j])
        p = scale_from_domain(p2, f1)
        Z[i,j] = test_f(p)
        if eval("boundary_"+test_f.__name__+"(p2)"): constr.append(p2)
        surrogate = exp_cbo_gpr.surrogate
        classifier = exp_cbo_gpr.classifier
        mu, std = surrogate.predict([p], return_std=True)
        label = (classifier.predict([p]) > 0.68)
        p_nf = estimate_p_nonfeasible(p, classifier)
        f_func[i,j] = mu
        std_fn[i,j] = std
        n_feas[i,j] = np.nan if label else 1
        p_feas[i,j] = 1 - p_nf

sm_err = np.abs(f_func - Z)
f_func -= np.min(f_func) - 1e-12

print("MIN: ",np.nanmin(Z))
plt.pcolormesh(X, Y, Z.T, norm='log')
plt.contour(X, Y, Z.T, levels=18, linewidths=0.5, colors='0.5', norm='log')
plt.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
plt.plot(x1_1[0], x1_1[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
plt.plot(x2_1[0], x2_1[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
plt.plot(x3_1[0], x3_1[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='best')
plt.savefig(class_type+"-CBO_Rosenbrock-surface.png", dpi=600)
plt.close()

ax1 = plt.subplot2grid((2,2), (0,0))
ax1.pcolormesh(X, Y, f_func.T)#, norm='log')
ax1.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax1.plot(x1_1[0], x1_1[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax1.plot(x2_1[0], x2_1[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax1.plot(x3_1[0], x3_1[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("surrogate model")

ax2 = plt.subplot2grid((2,2), (0,1))
ax2.pcolormesh(X, Y, n_feas.T, cmap='Greys', vmin=0, vmax=1.75)
for p_c in constr: ax2.plot(*p_c, ls='', marker='.', c='k', ms=1)
ax2.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax2.plot(x1_1[0], x1_1[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax2.plot(x2_1[0], x2_1[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax2.plot(x3_1[0], x3_1[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("estimated feasible region")

ax3 = plt.subplot2grid((2,2), (1,0))
ax3.pcolormesh(X, Y, sm_err.T, norm='log')
#ax3.pcolormesh(X, Y, std_fn.T)
ax3.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax3.plot(x1_1[0], x1_1[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax3.plot(x2_1[0], x2_1[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax3.plot(x3_1[0], x3_1[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("|surrogate - func|")

ax4 = plt.subplot2grid((2,2), (1,1))
ax4.pcolormesh(X, Y, p_feas.T)
for p_c in constr: ax4.plot(*p_c, ls='', marker='.', c='w', ms=1)
ax4.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax4.plot(x1_1[0], x1_1[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax4.plot(x2_1[0], x2_1[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax4.plot(x3_1[0], x3_1[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_title("P(feasible)")

fig = plt.gcf()
fig.savefig(class_type+"-CBO_Rosenbrock-analysis.png", dpi=600)
plt.close()


###############################################################
## Example optimization Constrained Gomez-Levi test function ##
###############################################################

test_f = gomez_levi_c
optimum_for_gap = 0.0000000005
options = {'iter_phase1':20, 'iter_phase2':50,
           'n_init':32, 'sampler':'sobol'}


## Experiment with SVM-CBO: RBF kernel
exp_cbo_gpr = CBO(f=test_f, surrogate_type="GP", **options)
exp_cbo_gpr.init_opt()
t0 = time()
exp_cbo_gpr.phase1()
exp_cbo_gpr.phase2()
t2_GPR = time() - t0
res1 = exp_cbo_gpr.generate_result()
x1_2 = scale_to_domain(res1.get('x'), test_f.__name__)
y1_2 = res1.get('fun')
gap_metric_cbo_gpr = exp_cbo_gpr.gap_metric(optimum_value=optimum_for_gap)

## Experiment with SVM-CBO: Matern-5/2 kernel
exp_cbo = CBO(f=test_f, surrogate_type="GP", surrogate_kernel="Matern",
                    surrogate_kernel_kwargs={'nu':5/2}, **options)
exp_cbo.init_opt()
t0 = time()
exp_cbo.phase1()
exp_cbo.phase2()
t2_GPM = time() - t0
res2 = exp_cbo.generate_result()
x2_2 = scale_to_domain(res2.get('x'), test_f.__name__)
y2_2 = res2.get('fun')
gap_metric_cbo_gpm = exp_cbo.gap_metric(optimum_value=optimum_for_gap)


## Experiment with SVM-CBO_RF
exp_cbo_rf = CBO(f=test_f, surrogate_type="RF", **options)
exp_cbo_rf.init_opt()
t0 = time()
exp_cbo_rf.phase1()
exp_cbo_rf.phase2()
t2_RF = time() - t0
res3 = exp_cbo_rf.generate_result()
x3_2 = scale_to_domain(res3.get('x'), test_f.__name__)
y3_2 = res3.get('fun')
gap_metric_cbo_rf = exp_cbo_rf.gap_metric(optimum_value=optimum_for_gap)

## Comparison gap metric between SVMCBO and SVMCBO_RF
f2 = test_f.__name__

plt.plot(gap_metric_cbo_gpr, label="SVM-CBO GP(RBF)")
plt.plot(gap_metric_cbo_gpm, ls='--', label="SVM-CBO GP(Matern-5/2)")
plt.plot(gap_metric_cbo_rf, ls=':', label="SVM-CBO RF")
plt.ylim(0.0,1.1)
plt.ylabel("Gap Metric")
plt.xlabel("Iteration")
plt.title("Comparison gap metric on {} test function".format(test_f.__name__))
plt.legend(loc="best")
plt.savefig(class_type+"-CBO_GomezLevi-metric.png", dpi=600)
plt.close()

bounds = get_bounds(test_f.__name__)
x = np.linspace(bounds[0,0], bounds[0,1], 750)
y = np.linspace(bounds[1,0], bounds[1,1], 750)
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X.T)
f_func, std_fn = np.zeros_like(X.T), np.zeros_like(X.T)
p_feas, n_feas = np.zeros_like(X.T), np.zeros_like(X.T)
constr = []
for i, x_i in enumerate(x):
    for j, y_j in enumerate(y):
        p2 = np.array([x_i,y_j])
        p = scale_from_domain(p2, f2)
        Z[i,j] = test_f(p)
        if eval("boundary_"+test_f.__name__+"(p2)"): constr.append(p2)
        surrogate = exp_cbo_gpr.surrogate
        classifier = exp_cbo_gpr.classifier
        mu, std = surrogate.predict([p], return_std=True)
        p_nf = estimate_p_nonfeasible(p, classifier)
        label = (classifier.predict([p]) > 0.68)
        f_func[i,j] = mu
        std_fn[i,j] = std
        n_feas[i,j] = np.nan if label else 1
        p_feas[i,j] = 1 - p_nf

sm_err = np.abs(f_func - Z)
f_func -= np.min(f_func) - 1e-12

print("MIN: ",np.nanmin(Z))
plt.pcolormesh(X, Y, Z.T)
plt.contour(X, Y, Z.T, levels=14, linewidths=0.5, colors='0.5')
plt.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
plt.plot(x1_2[0], x1_2[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
plt.plot(x2_2[0], x2_2[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
plt.plot(x3_2[0], x3_2[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='best')
plt.savefig(class_type+"-CBO_GomezLevi-surface.png", dpi=600)
plt.close()


ax1 = plt.subplot2grid((2,2), (0,0))
ax1.pcolormesh(X, Y, f_func.T)
ax1.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax1.plot(x1_2[0], x1_2[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax1.plot(x2_2[0], x2_2[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax1.plot(x3_2[0], x3_2[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("surrogate model")

ax2 = plt.subplot2grid((2,2), (0,1))
ax2.pcolormesh(X, Y, n_feas.T, cmap='Greys', vmin=0, vmax=1.75)
for p_c in constr: ax2.plot(*p_c, ls='', marker='.', c='k', ms=1)
ax2.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax2.plot(x1_2[0], x1_2[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax2.plot(x2_2[0], x2_2[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax2.plot(x3_2[0], x3_2[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("estimated feasible region")

ax3 = plt.subplot2grid((2,2), (1,0))
ax3.pcolormesh(X, Y, sm_err.T)
#ax3.pcolormesh(X, Y, std_fn.T)
ax3.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax3.plot(x1_2[0], x1_2[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax3.plot(x2_2[0], x2_2[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax3.plot(x3_2[0], x3_2[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("|surrogate - func|")
#ax3.set_title("std(surrogate)")

ax4 = plt.subplot2grid((2,2), (1,1))
ax4.pcolormesh(X, Y, p_feas.T)
for p_c in constr: ax4.plot(*p_c, ls='', marker='.', c='w', ms=1)
ax4.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax4.plot(x1_2[0], x1_2[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax4.plot(x2_2[0], x2_2[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax4.plot(x3_2[0], x3_2[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_title("P(feasible)")

fig = plt.gcf()
fig.savefig(class_type+"-CBO_GomezLevi-analysis.png", dpi=600)
plt.close()


###############################################################
## Example optimization Constrained Gomez-Levi test function ##
###############################################################

test_f = mishra_bird_c
optimum_for_gap = -106.7645367
options = {'iter_phase1':20, 'iter_phase2':50,
           'n_init':32, 'sampler':'sobol'}

## Experiment with SVM-CBO: RBF kernel
exp_cbo_gpr = CBO(f=test_f, surrogate_type="GP", **options)
exp_cbo_gpr.init_opt()
t0 = time()
exp_cbo_gpr.phase1()
exp_cbo_gpr.phase2()
t3_GPR = time() - t0
res1 = exp_cbo_gpr.generate_result()
x1_3 = scale_to_domain(res1.get('x'), test_f.__name__)
y1_3 = res1.get('fun')
gap_metric_cbo_gpr = exp_cbo_gpr.gap_metric(optimum_value=optimum_for_gap)

## Experiment with SVM-CBO: Matern-5/2 kernel
exp_cbo = CBO(f=test_f, surrogate_type="GP", surrogate_kernel="Matern",
                    surrogate_kernel_kwargs={'nu':5/2}, **options)
exp_cbo.init_opt()
t0 = time()
exp_cbo.phase1()
exp_cbo.phase2()
t3_GPM = time() - t0
res2 = exp_cbo.generate_result()
x2_3 = scale_to_domain(res2.get('x'), test_f.__name__)
y2_3 = res2.get('fun')
gap_metric_cbo_gpm = exp_cbo.gap_metric(optimum_value=optimum_for_gap)

## Experiment with SVM-CBO_RF
exp_cbo_rf = CBO(f=test_f, surrogate_type="RF", **options)
exp_cbo_rf.init_opt()
t0 = time()
exp_cbo_rf.phase1()
exp_cbo_rf.phase2()
t3_RF = time() - t0
res3 = exp_cbo_rf.generate_result()
x3_3 = scale_to_domain(res3.get('x'), test_f.__name__)
y3_3 = res3.get('fun')
gap_metric_cbo_rf = exp_cbo_rf.gap_metric(optimum_value=optimum_for_gap)

## Comparison gap metric between SVMCBO and SVMCBO_RF
f3 = test_f.__name__

plt.plot(gap_metric_cbo_gpr, label="SVM-CBO GP(RBF)")
plt.plot(gap_metric_cbo_gpm, ls='--', label="SVM-CBO GP(Matern-5/2)")
plt.plot(gap_metric_cbo_rf, ls=':', label="SVM-CBO RF")
plt.ylim(0.0,1.1)
plt.ylabel("Gap Metric")
plt.xlabel("Iteration")
plt.title("Comparison gap metric on {} test function".format(test_f.__name__))
plt.legend(loc="best")
plt.savefig(class_type+"-CBO_MishraBird-metric.png", dpi=600)
plt.close()


bounds = get_bounds(test_f.__name__)
x = np.linspace(bounds[0,0], bounds[0,1], 760)
y = np.linspace(bounds[1,0], bounds[1,1], 494)
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X.T)
f_func, std_fn = np.zeros_like(X.T), np.zeros_like(X.T)
p_feas, n_feas = np.zeros_like(X.T), np.zeros_like(X.T)
constr = []
for i, x_i in enumerate(x):
    for j, y_j in enumerate(y):
        p2 = np.array([x_i,y_j])
        p = scale_from_domain(p2, f3)
        Z[i,j] = test_f(p)
        if eval("boundary_"+test_f.__name__+"(p2)"): constr.append(p2)
        surrogate = exp_cbo_gpr.surrogate
        classifier = exp_cbo_gpr.classifier
        mu, std = surrogate.predict([p], return_std=True)
        p_nf = estimate_p_nonfeasible(p, classifier)
        label = (classifier.predict([p]) > 0.68)
        f_func[i,j] = mu
        std_fn[i,j] = std
        n_feas[i,j] = np.nan if label else 1
        p_feas[i,j] = 1 - p_nf

sm_err = np.abs(f_func - Z)
f_func -= np.min(f_func) - 1e-12


print("MIN: ",np.nanmin(Z))
plt.pcolormesh(X, Y, Z.T)
plt.contour(X, Y, Z.T, levels=18, linewidths=0.5, colors='0.5')
plt.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
plt.plot(x1_3[0], x1_3[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
plt.plot(x2_3[0], x2_3[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
plt.plot(x3_3[0], x3_3[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='best')
plt.savefig(class_type+"-CBO_MishraBird-surface.png", dpi=600)
plt.close()


ax1 = plt.subplot2grid((2,2), (0,0))
ax1.pcolormesh(X, Y, f_func.T)
ax1.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax1.plot(x1_3[0], x1_3[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax1.plot(x2_3[0], x2_3[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax1.plot(x3_3[0], x3_3[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("surrogate model")

ax2 = plt.subplot2grid((2,2), (0,1))
ax2.pcolormesh(X, Y, n_feas.T, cmap='Greys', vmin=0, vmax=1.75)
for p_c in constr: ax2.plot(*p_c, ls='', marker='.', c='k', ms=1)
ax2.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax2.plot(x1_3[0], x1_3[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax2.plot(x2_3[0], x2_3[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax2.plot(x3_3[0], x3_3[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("estimated feasible region")

ax3 = plt.subplot2grid((2,2), (1,0))
#ax3.pcolormesh(X, Y, std_fn.T)
ax3.pcolormesh(X, Y, sm_err.T)
ax3.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax3.plot(x1_3[0], x1_3[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax3.plot(x2_3[0], x2_3[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax3.plot(x3_3[0], x3_3[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("|surrogate - func|")
#ax3.set_title("std(surrogate)")

ax4 = plt.subplot2grid((2,2), (1,1))
ax4.pcolormesh(X, Y, p_feas.T)
for p_c in constr: ax4.plot(*p_c, ls='', marker='.', c='w', ms=1)
ax4.plot(*get_minimum(test_f.__name__), ls='', marker='s', mec='k', mfc='hotpink', mew=1.25, label="global min")
ax4.plot(x1_3[0], x1_3[1], ls='', marker='o', c='tab:red', mec='k', mew=0.5, label="GP(RBF)")
ax4.plot(x2_3[0], x2_3[1], ls='', marker='d', c='tab:red', mec='k', mew=0.5, label="GP(Matern-5/2)")
ax4.plot(x3_3[0], x3_3[1], ls='', marker='^', c='tab:red', mec='k', mew=0.5, label="RF")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_title("P(feasible)")

fig = plt.gcf()
fig.savefig(class_type+"-CBO_MishraBird-analysis.png", dpi=600)
plt.close()



print("\n\n### " + f1 + "###")
print("\nGaussian Process (RBF)")
print(f"Optimum point: {x1_1}")
print(f"Optimal value: {y1_1}")
print(f"Total time:    {t1_GPR}\n")

print("Gaussian Process (Matern-5/2)")
print(f"Optimum point: {x2_1}")
print(f"Optimal value: {y2_1}")
print(f"Total time:    {t1_GPM}\n")

print("Random Forests")
print(f"Optimum point: {x3_1}")
print(f"Optimal value: {y3_1}")
print(f"Total time:    {t1_RF}\n")


print("\n### " + f2 + "###")
print("\nGaussian Process (RBF)")
print(f"Optimum point: {x1_2}")
print(f"Optimal value: {y1_2}")
print(f"Total time:    {t2_GPR}\n")

print("Gaussian Process (Matern-5/2)")
print(f"Optimum point: {x2_2}")
print(f"Optimal value: {y2_2}")
print(f"Total time:    {t2_GPM}\n")

print("Random Forests")
print(f"Optimum point: {x3_2}")
print(f"Optimal value: {y3_2}")
print(f"Total time:    {t2_RF}\n")


print("\n### " + f3 + "###")
print("\nGaussian Process (RBF)")
print(f"Optimum point: {x1_3}")
print(f"Optimal value: {y1_3}")
print(f"Total time:    {t3_GPR}\n")

print("Gaussian Process (Matern-5/2)")
print(f"Optimum point: {x2_3}")
print(f"Optimal value: {y2_3}")
print(f"Total time:    {t3_GPM}\n")

print("Random Forests")
print(f"Optimum point: {x3_3}")
print(f"Optimal value: {y3_3}")
print(f"Total time:    {t3_RF}\n")



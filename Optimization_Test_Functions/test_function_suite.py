import numpy as np
import subprocess

### To define a new test function please add the domain bounds inside of the definition of the function

def rosenbrock(x):
    maxx = np.array([1.5, 2.5])
    minn = np.array([-1.5, -0.5])
    x = x * (maxx-minn) + minn

    x1 = x[0]
    x2 = x[1]
    return((1-x1)**2) + 100 * (x2-x1**2)**2

def rosenbrock_fun_c_1(x):
    maxx = np.array([1.5, 2.5])
    minn = np.array([-1.5, -0.5])
    x = x * (maxx-minn) + minn

    x1 = x[0]
    x2 = x[1]
    if ( x1**2 + x2**2 ) <= 2.:
        return((1-x1)**2) + 100 * (x2-x1**2)**2
    else:
        return np.nan

def rosenbrock_fun_c_2(x):
    maxx = np.array([1.5, 2.5])
    minn = np.array([-1.5, -0.5])
    x = x * (maxx-minn) + minn

    x1 = x[0]
    x2 = x[1]
    c = ((x1 - 1)**3 - x2 <= -1.) and (x1 + x2 <= 2.)
    if c:
        return ((1-x1)**2) + 100 * (x2-x1**2)**2
    else:
        return np.nan

def branin(x):

    x1 = x[0]
    x2 = x[1]
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    term1 = x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y = (term1**2 + term2 - 44.81) / 51.95

    return y

def branin_c(x, a=1, b=5.1/(4*np.pi**2) ,c=5/np.pi):
    # ----- Parameters of the elliptic feasible region #1 -------------
    x1min = 0
    x1max = 1
    cx1 = (x1min + x1max) / 3
    cx2 = (x1min + x1max) / 4
    a = 0.45
    b = 0.27
    alpha = np.pi / 4
    # -------------------------------------------------------------------

    # ----- Params of the elliptic feas. region #2 (for disonnected case)
    c2x1 = 5 * (x1min + x1max) / 6
    c2x2 = 7 * (x1min + x1max) / 8
    a2 = 0.25
    b2 = 0.1
    alpha2 = 3 * np.pi / 4
    # -------------------------------------------------------------------

    x1 = x[0]
    x2 = x[1]
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    term1 = x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y = (term1**2 + term2 - 44.81) / 51.95

    check_1 = ((((x1-cx1)*np.cos(alpha)) + ((x2-cx2)*np.sin(alpha)))**2)/(a**2) + \
              ((((x2-cx2)*np.cos(alpha)) - ((x1-cx1)*np.sin(alpha)))**2)/(b**2)

    check_2 = ((((x1-c2x1)*np.cos(alpha2)) + ((x2-c2x2)*np.sin(alpha2)))**2)/(a2**2) + \
              ((((x2-c2x2)*np.cos(alpha2)) - ((x1-c2x1)*np.sin(alpha2)))**2)/(b2**2)

    if check_1 < 1 or check_2 < 1:
        return y
    else:
        return np.nan


def branin_c_disc(x, a=1, b=5.1/(4*np.pi**2) ,c=5/np.pi, disc_factor=30):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    # ----- Parameters of the elliptic feasible region #1 -------------
    x1min = 0
    x1max = 1
    cx1 = (x1min + x1max) / 3
    cx2 = (x1min + x1max) / 4
    a = 0.45
    b = 0.27
    alpha = np.pi / 4
    # -------------------------------------------------------------------

    # ----- Params of the elliptic feas. region #2 (for disonnected case)
    c2x1 = 5 * (x1min + x1max) / 6
    c2x2 = 7 * (x1min + x1max) / 8
    a2 = 0.25
    b2 = 0.1
    alpha2 = 3 * np.pi / 4
    # -------------------------------------------------------------------

    x1 = x[0]
    x2 = x[1]
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    term1 = x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y = (term1**2 + term2 - 44.81) / 51.95

    check_1 = ((((x1-cx1)*np.cos(alpha)) + ((x2-cx2)*np.sin(alpha)))**2)/(a**2) + \
              ((((x2-cx2)*np.cos(alpha)) - ((x1-cx1)*np.sin(alpha)))**2)/(b**2)

    check_2 = ((((x1-c2x1)*np.cos(alpha2)) + ((x2-c2x2)*np.sin(alpha2)))**2)/(a2**2) + \
              ((((x2-c2x2)*np.cos(alpha2)) - ((x1-c2x1)*np.sin(alpha2)))**2)/(b2**2)

    if check_1 < 1 or check_2 < 1:
        return y
    else:
        return np.nan

def michalewicz(x, m=10):
    maxx = np.array([np.pi, np.pi])
    minn = np.array([0, 0])
    x = x * (maxx-minn) + minn
    y = 0
    for i in np.arange(len(x)):
        y = y + np.sin(x[i]) * np.sin(((i+1)*x[i]**2)/(np.pi))**(2*m)
    return(-y)

def michalewicz_c(x, m=10):
    # maxx = np.array([np.pi, np.pi])
    # minn = np.array([0, 0])
    maxx = np.repeat(np.pi, len(x)) #np.array([np.pi, np.pi])
    minn = np.repeat(0.0, len(x)) #np.array([0, 0])
    x = x * (maxx-minn) + minn
    if x[0]**3 + x[1]**3 > 15:# and x[0]**3 + x[1]**3 < 30:  #25
        return(np.nan)
    y = 0
    for i in np.arange(len(x)):
        y = y + np.sin(x[i]) * np.sin(((i+1)*x[i]**2)/(np.pi))**(2*m)
    return -y

def dejong5(x):
    maxx = np.array([50., 50.])
    minn = np.array([-50., -50.])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    mat = "-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32;" \
          "-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32"
    A = np.matrix(mat)

    sumterm1 = np.arange(0,25)
    sumterm2 = np.power((x1 - A[0, 0:26]),6)
    sumterm3 = np.power((x2 - A[1, 0:26]),6)
    sum = np.sum(1 / (sumterm1 + sumterm2 + sumterm3))

    y = 1 / (0.002 + sum)
    return y

def dejong5_c(x):
    maxx = np.array([50., 50.])
    minn = np.array([-50., -50.])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    if x1**3-x2**3 > 20:
        return np.nan
    mat = "-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32;" \
          "-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32"
    A = np.matrix(mat)

    sumterm1 = np.arange(0,25)
    sumterm2 = np.power((x1 - A[0, 0:26]),6)
    sumterm3 = np.power((x2 - A[1, 0:26]),6)
    sum = np.sum(1 / (sumterm1 + sumterm2 + sumterm3))

    y = 1 / (0.002 + sum)
    return y

def mishra_bird(x):
    maxx = np.array([0., 0.])
    minn = np.array([-10., -6.5])
    #minn = np.array([-20., -13])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    term1 = np.sin(x2)*np.exp((1-np.cos(x1))**2)
    term2 = np.cos(x1)*np.exp((1-np.sin(x2))**2)
    term3 = (x1-x2)**2
    return term1 + term2 + term3 + 106.7645367

def mishra_bird_c(x):
    maxx = np.array([0., 0.])
    minn = np.array([-10., -6.5])
    #minn = np.array([-20., -13])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    term1 = np.sin(x2)*np.exp((1-np.cos(x1))**2)
    term2 = np.cos(x1)*np.exp((1-np.sin(x2))**2)
    term3 = (x1-x2)**2
    y = term1 + term2 + term3 + 106.7645367
    if (x1+5)**2 + (x2+5)**2 < 25:
        return y
    else:
        return np.nan

def mishra_bird_c_disc(x, disc_factor=50):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    #print(x)
    maxx = np.array([0., 0.])
    minn = np.array([-10., -6.5])
    #minn = np.array([-20., -13])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    term1 = np.sin(x2)*np.exp((1-np.cos(x1))**2)
    term2 = np.cos(x1)*np.exp((1-np.sin(x2))**2)
    term3 = (x1-x2)**2
    y = term1 + term2 + term3 + 106.7645367
    if (x1+5)**2 + (x2+5)**2 < 25:
        return y
    else:
        return np.nan

#Corrected form of constraints here: https://dl.acm.org/doi/abs/10.5555/2946645.3053442
# Robert B. Gramacy, Genetha A. Gray, S´ebastien Le Digabel, Herbert K. H. Lee, Pritam
# Ranjan, Garth Wells, and Stefan M. Wild. Modeling an augmented Lagrangian for
# blackbox constrained optimization. Technometrics, 58(1):1–11, 2016.
def test54(x):
  #maxx = np.array([1., 0.75])
  maxx = np.array([1., 1.])
  minn = np.array([0., 0.])
  x = x * (maxx - minn) + minn
  x1 = x[0]
  x2 = x[1]
  y = x1+x2
  c1 = 0.5*np.sin(2*np.pi*(x1**2-2*x2))+x1+2*x2-1.5
  c2 = -x1**2-x2**2+1.5
  if c1>=0 and c2>=0:
    return y
  else:
    return np.nan

def test54_disc(x, disc_factor = 30):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    #maxx = np.array([1., 0.75])
    maxx = np.array([1., 1.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]

    y = x1+x2
    c1 = 0.5*np.sin(2*np.pi*(x1**2-2*x2))+x1+2*x2-1.5
    c2 = -x1**2-x2**2+1.5
    if c1>=0 and c2>=0:
        return y
    else:
        return np.nan

def alpine2(x):
    maxx = np.array([10., 10.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = (np.sin(x1) * np.sqrt(x1)) * (np.sin(x2) * np.sqrt(x2))
    return -y

def alpine2_c(x):
    maxx = np.array([10., 10.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = (np.sin(x1) * np.sqrt(x1)) * (np.sin(x2) * np.sqrt(x2))
    c1 = (x1-5)**2 + (x2-5)**2 - 0.25
    if c1 < 17:
        return -y
    else:
        return np.nan

def alpine2_c_disc(x, disc_factor = 30):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    maxx = np.array([10., 10.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = (np.sin(x1) * np.sqrt(x1)) * (np.sin(x2) * np.sqrt(x2))
    c1 = (x1-5)**2 + (x2-5)**2 - 0.25
    if c1 < 17:
        return -y
    else:
        return np.nan

# To use this test function you have to install the Emmental-GKLS generator
# The generator can be used only if asked to the authors
# For the authors please reference to this work:
# Sergeyev, Y. D., Kvasov, D. E., & Mukhametzhanov, M. S. (2017, June). Emmental-type GKLS-based multiextremal smooth test problems with non-linear constraints. In International Conference on Learning and Intelligent Optimization (pp. 383-388). Springer, Cham. 
# Link to the paper: https://link.springer.com/chapter/10.1007/978-3-319-69404-7_35

def CGen_function(x, seed=1, hardness='simple'):

    maxx = np.repeat(1, len(x))
    minn = np.repeat(-1, len(x))
    x = x * (maxx-minn) + minn
    dim_test = len(x)
    dict_hardness = {'simple' : {2 : {'r' : 0.90, 'p' : 0.2 },
                                 3 : {'r' : 0.66, 'p' : 0.2 },
                                 4 : {'r' : 0.66, 'p' : 0.2 },
                                 5 : {'r' : 0.66, 'p' : 0.3 }},
                     'hard' : {2 : {'r' : 0.90, 'p' : 0.1 },
                               3 : {'r' : 0.90, 'p' : 0.2 },
                               4 : {'r' : 0.90, 'p' : 0.2 },
                               5 : {'r' : 0.66, 'p' : 0.2 }}}

    r = dict_hardness.get(hardness).get(dim_test).get('r')
    p = dict_hardness.get(hardness).get(dim_test).get('p')
    out = ""
    for x_el in x:
        out = out + str(x_el) + " "
    input_generator = f"CGen_function.exe 0 d {seed} 30 -1.00 {r} {p} 20 2 10 5 y {out}"

    p1 = subprocess.Popen(["cmd", "/C", input_generator],
                          stdout=subprocess.PIPE)
    line = p1.stdout.readlines()
    p1.kill()
    c = float(str(line[-2]).split(":")[-1].split("\\")[0])

    if c == 0:
        y = np.nan
    else:
        y = float(str(line[-1]).split(":")[-1].split("\\")[0])
    return y


def CGen_function_disc(x, seed=1, hardness='simple', disc_factor=30):

    ## Discrization of the function
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    maxx = np.repeat(1, len(x))
    minn = np.repeat(-1, len(x))
    x = x * (maxx-minn) + minn
    dim_test = len(x)
    dict_hardness = {'simple' : {2 : {'r' : 0.90, 'p' : 0.2 },
                                 3 : {'r' : 0.66, 'p' : 0.2 },
                                 4 : {'r' : 0.66, 'p' : 0.2 },
                                 5 : {'r' : 0.66, 'p' : 0.3 }},
                     'hard' : {2 : {'r' : 0.90, 'p' : 0.1 },
                               3 : {'r' : 0.90, 'p' : 0.2 },
                               4 : {'r' : 0.90, 'p' : 0.2 },
                               5 : {'r' : 0.66, 'p' : 0.2 }}}

    r = dict_hardness.get(hardness).get(dim_test).get('r')
    p = dict_hardness.get(hardness).get(dim_test).get('p')
    out = ""
    for x_el in x:
        out = out + str(x_el) + " "
    input_generator = f"CGen_function.exe 0 d {seed} 30 -1.00 {r} {p} 20 2 10 5 y {out}"

    p1 = subprocess.Popen(["cmd", "/C", input_generator],
                          stdout=subprocess.PIPE)
    line = p1.stdout.readlines()
    p1.kill()
    c = float(str(line[-2]).split(":")[-1].split("\\")[0])

    if c == 0:
        y = np.nan
    else:
        y = float(str(line[-1]).split(":")[-1].split("\\")[0])
    return y


def gomez_levi(x):
    maxx = np.array([ 1.0,  1.0])
    minn = np.array([-1.0, -1.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    return 4*x1**2 - 2.1*x1**4 + x1**6 / 3 + x1*x2 - 4*x2**2 + 4*x2**4 + 1.0316284526

def gomez_levi_c(x):
    maxx = np.array([ 1.0,  1.0])
    minn = np.array([-1.0, -1.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    c = 2 * np.sin(2*np.pi*x2)**2 - np.sin(4*np.pi*x1) <= 1.5
    if c:
        return 4*x1**2 - 2.1*x1**4 + x1**6 / 3 + x1*x2 - 4*x2**2 + 4*x2**4 + 1.0316284526
    else:
        return np.nan

def six_hump_camel(x):
    maxx = np.array([ 3.0,  2.0])
    minn = np.array([-3.0, -2.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    return x1*x1*(4 - 2.1*x1*x1 + x1**4/3) + x1*x2 + 4*x2*x2*(x2*x2 - 1)

def drop_wave(x):
    maxx = np.array([ 2.0,  2.0])
    minn = np.array([-2.0, -2.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    return 1. - (1 + np.cos(12.* np.sqrt(x1*x1 + x2*x2))) / (0.5*(x1*x1 + x2*x2) + 2.)
    
def drop_wave_c(x, lim=0.9):
    maxx = np.array([ 2.0,  2.0])
    minn = np.array([-2.0, -2.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = 1. - (1 + np.cos(12.* np.sqrt(x1*x1 + x2*x2))) / (0.5*(x1*x1 + x2*x2) + 2.)
    if y < lim: return y
    return np.nan

def ackley(x, a=20., b=.2, c=2*np.pi, dim=2):
    maxx = np.array([ 32.768,]*dim)
    minn = np.array([-32.768,]*dim)
    x = x * (maxx-minn) + minn
    rms = np.sqrt(np.mean(x*x))
    mcos = np.mean(np.cos(c*x))
    y = a + np.exp(1) - a*np.exp(-b*rms) - np.exp(mcos)
    return y

def ackley_ellipse(x, a=20., b=.2, c=2*np.pi, lengths=None, dim=2):
    if lengths is None: lengths = np.random.random(dim)*0.8 + 0.2
    in_ellipse = np.sum((x/lengths)**2) < 1
    if in_ellipse: return ackley(x, a=a, b=b, c=c, dim=dim)
    return np.nan

def hartmann_3d(x):
    maxx = np.array([1., 1., 1.])
    minn = np.array([0., 0., 0.])
    x = x * (maxx-minn) + minn
    alpha = np.array([1., 1.2, 3., 3.2])
    A = np.array([[3.0, 10., 30.],
                  [0.1, 10., 35.],
                  [3.0, 10., 30.],
                  [0.1, 10., 35.]])
    P = np.array([[3689., 1170., 2673.],
                  [4699., 4387., 7470.],
                  [1091., 8732., 5547.],
                  [ 381., 5743., 8828.]])*1e-4
    y = 3.86278
    for i in range(4):
        expon = 0.
        for j,xj in enumerate(x): expon -= A[i,j]*(xj - P[i,j])**2
        y -= alpha[i] * np.exp(expon)
    return y

def powell_singular(x):
    maxx = np.array([ 5.,  5.,  5.,  5.])
    minn = np.array([-4., -4., -4., -4.])
    x = x * (maxx-minn) + minn
    y = (x[0] + 10.*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10.*(x[0] - x[3])**4
    return y

def colville(x):
    maxx = np.array([ 10.,  10.,  10.,  10.])
    minn = np.array([-10., -10., -10., -10.])
    x = x * (maxx-minn) + minn
    y  = 100.*(x[0]**2 - x[1])**2 + (x[0] - 1.)**2 + (x[2] - 1.)**2 + 90.*(x[2]**2 - x[3])**2
    y += 10.1*( (x[1] - 1.)**2 + (x[3] - 1.)**2 ) + 19.8*(x[1] - 1.)*(x[3] - 1.)
    return y

def shekel(x):
    maxx = np.array([10., 10., 10., 10.])
    minn = np.array([0.0, 0.0, 0.0, 0.0])
    x = x * (maxx-minn) + minn
    bt = np.array([1., 2., 2., 4., 4., 6., 3., 7., 5., 5.])/10.
    C = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.0],
                  [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6],
                  [4., 1., 8., 6., 3., 2., 5., 8., 6., 7.0],
                  [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6]])
    y = 10.5364
    for i in range(10):
        for j,xj in enumerate(x): y -= 1. / ( (xj - C[j,i])**2 + bt[i] )
    return y

def hartmann_6d(x):
    maxx = np.array([1.,]*6)
    minn = np.array([0.,]*6)
    x = x * (maxx-minn) + minn
    alpha = np.array([1., 1.2, 3., 3.2])
    A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                  [0.05, 10., 17.0, 0.1, 8.0, 14.],
                  [3.00, 3.5, 1.70, 10., 17., 8.0],
                  [17.0, 8.0, 0.05, 10., 0.1, 14.]])
    P = np.array([[1312., 1696., 5569.,  124., 8283., 5886.],
                  [2329., 4135., 8307., 3736., 1004., 9991.],
                  [2348., 1451., 3522., 2883., 3047., 6650.],
                  [4047., 8828., 8732., 5743., 1091.,  381.]])*1e-4
    y = 3.32237
    for i in range(4):
        expon = 0.
        for j,xj in enumerate(x): expon -= A[i,j]*(xj - P[i,j])**2
        y -= alpha[i] * np.exp(expon)
    return y

def tension_compression(x):
    """ From Kumar et al. Swarm and Evolutionary Computation 56, 100693 (2020). """
    maxx = np.array([2.00, 1.30, 15.0])
    minn = np.array([0.05, 0.25, 2.00])
    x = x * (maxx-minn) + minn
    x1, x2, x3 = x
    c1 = x2**3 * x3 / (71785*x1**4) < 1.
    c2 = x2*(4*x2 - x1) / (12566*x1**3(x2 - x1)) + 1. / (5108*x1**2) > 1.
    c3 = 140.45*x1 / (x2*x2*x3) < 1.
    c4 = (x1 + x2)/1.5 > 1.
    if any((c1,c2,c3,c4)): return np.nan
    return x1*x1*x2*(2. + x3)


def gas_compressor(x):
    """ From Kumar et al. Swarm and Evolutionary Computation 56, 100693 (2020). """
    maxx = np.array([50., 10., 50., 60.])
    minn = np.array([20., 1.0, 20., 0.1])
    x = x * (maxx-minn) + minn
    x1, x2, x3, x4 = x
    c = (x4 + 1.)/x2**2 > 1
    if c: return np.nan
    return 8.61e5 * np.sqrt(x1) * x2 / x3**(2/3) / np.sqrt(x4) + 3.69e4 * x3 + (7.72e8 * x2**0.219 - 765.43e6) / x1



search_domain = {
    'rosenbrock':          np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    'rosenbrock_fun_c_1':  np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    'rosenbrock_fun_c_2':  np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    'michalewicz':         np.array([[0., np.pi], [0., np.pi]]),
    'michalewicz_c':       np.array([[0., np.pi], [0., np.pi]]),
    'mishra_bird':         np.array([[-10.,  0.], [-6.5,  0.]]),
    'mishra_bird_c':       np.array([[-10.,  0.], [-6.5,  0.]]),
    'mishra_bird_c_disc':  np.array([[-10.,  0.], [-6.5,  0.]]),
    'alpine2':             np.array([[  0., 10.], [  0., 10.]]),
    'alpine2_c':           np.array([[  0., 10.], [  0., 10.]]),
    'alpine2_c_disc':      np.array([[  0., 10.], [  0., 10.]]),
    'gomez_levi':          np.array([[ -1.,  1.], [ -1.,  1.]]),
    'gomez_levi_c':        np.array([[ -1.,  1.], [ -1.,  1.]]),
    'six_hump_camel':      np.array([[ -3.,  3.], [ -2.,  2.]]),
    'drop_wave':           np.array([[ -2.,  2.], [ -2.,  2.]]),
    'drop_wave_c':         np.array([[ -2.,  2.], [ -2.,  2.]]),
    'ackley':              np.array([[-32.768, 32.768], [-32.768, 32.768]]),
    'ackley_ellipse':      np.array([[-32.768, 32.768], [-32.768, 32.768]]),
    'hartmann_3d':         np.array([[0., 1.],]*3),
    'powell_singular':     np.array([[-4., 5.],]*4),
    'colville':            np.array([[-10.,10.],]*4),
    'shekel':              np.array([[0., 10.],]*4),
    'tension_compression': np.array([[0.05, 2.], [0.25, 1.30], [2., 15.0]]),
    'hartmann_6d':         np.array([[0., 1.],]*6),
    'gas_compressor':      np.array([[20., 50.], [1.0, 10.], [20., 50.], [0.1, 60.]]),

            }

def get_dim(func_name):
    if func_name in ["hartmann_3d", "tension_compression_c"]:
        return 3
    elif func_name in ["shekel", "colville", "powell_singular", "gas_compressor"]:
        return 4
    elif func_name == "hartmann_6d":
        return 6
    return 2

def get_bounds(func_name):
    return search_domain.get(func_name, [[0.,1.],]*get_dim(func_name))

def scale_to_domain(x, func_name):
    bounds = get_bounds(func_name)
    return x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

def scale_from_domain(x, func_name):
    bounds = get_bounds(func_name)
    return (x - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

x_minimum = {
    'rosenbrock'          : np.array([1.,1.]),
    'rosenbrock_fun_c_1'  : np.array([1.,1.]),
    'rosenbrock_fun_c_2'  : np.array([1.,1.]),
    'mishra_bird'         : np.array([-3.1302468,-1.5821422]),
    'mishra_bird_c'       : np.array([-3.1302468,-1.5821422]),
    'mishra_bird_c_disc'  : np.array([-3.1302468,-1.5821422]),
    'gomez_levi'          : np.array([0.08984201, -0.7126564]),
    'gomez_levi_c'        : np.array([0.08984201, -0.7126564]),
    'drop_wave'           : np.array([0., 0.]),
    'drop_wave_c'         : np.array([0., 0.]),
    'ackley'              : np.array([0., 0.]),
    'ackley_ellipse'      : np.array([0., 0.]),
    'hartmann_3d'         : np.array([0.114614, 0.555649, 0.852547]),
    'powell_singular'     : np.array([0., 0., 0., 0.]),
    'colville'            : np.array([1., 1., 1., 1.]),
    'shekel'              : np.array([4., 4., 4., 4.]),
    'hartmann_6d'         : np.array([0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657300]),
          }

def get_minimum(func_name): return x_minimum.get(func_name, [0., 0.])

def get_minimum_value(func_name): return 0.


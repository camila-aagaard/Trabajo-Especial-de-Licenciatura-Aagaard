import numpy as np
from sympy import Matrix, diag, zeros, eye, simplify, factor, cancel, roots, solve, symbols
from sympy.abc import z
z = symbols('z', complex=True)

#definimos funciones auxiliares necesarias para la factorización:
def coef(A):
#dada un matriz con columnas linealmente dependientes nos da los coeficientes de la dependencia lineal y el número de la columna que depende de las demás
    n = A.shape[0]
    coef = np.array(A.nullspace()[0]).T[0]

    for i in reversed(range(n)):
        if coef[i] != 0:

            if i != 0:
                coef = [-x/coef[i] for x in coef[:i]]
                return coef, i
            
            else:
                return [coef[0]], 0

def find_U_inv(n, k, x, c):
#nos da la matriz Ui^−1 para cada iteración
    U = zeros(n,n)
    if k != 0:
        U[:k, k] = [-y/(z - x) for y in c]
        U[k, k] = 1/(z - x)
    else:
        U[k, k] = 1/(z - x)
    
    for i in [x for x in range(n) if x != k]:
            U[i, i] = 1

    return U

def find_V(n, k, x, c, lst):
#nos da la matriz Vi para cada iteración
    V = zeros(n,n)
    if k != 0:
        V[k, k] = 1 - (x/z)
        for j in range(k):
            V[j, k] = c[j]*z**(lst[k]-lst[j])
    else:
        V[k, k] = 1 - (x/z)

    for i in [x for x in range(n) if x != k]:
        V[i, i] = 1

    return V
    
def perm(lst):
#dados los índices de factorización en cada iteración nos da los índices de la próxima iteración y la matriz de permutación asociada
    sort = sorted(lst, reverse = True)
    n = len(lst)
    perm = np.eye(n)

    for i in range(n):
        if lst[i] != sort[i]:
            j = next(j for j in reversed(range(i, n)) if lst[j] == sort[i])
            perm[i, i] = 0
            perm[j, j] = 0
            perm[i, j] = 1
            perm[j, i] = 1
            lst[i], lst[j] = lst[j], lst[i]

    return Matrix(perm), lst


#finalmente presentamos el algoritmo para la factorización:
def fact(F, r = 1, type = "left"):
#dado un polinomio matricial F el algoritmo devuelve los factores F_+, Lambda (diagonal) y F_- de la factorización de F con respecto a un círculo de radio r a elección (centrado en el origen) y es posible realizar tanto la factorización a izquierda (type = "left") como a derecha (type = "right")
    n = F.shape[0]
    FM = F
    Fm = eye(n)
    L = eye(n)
    lst = [0] * n
    m = len([z0 for z0 in roots(FM.det(), z, multiple=True) if abs(z0) < r])

    if type == "left":
        for i in reversed(range(m)):
            if FM.det() != 0:
                x = [z0 for z0 in solve(FM.det(), z) if abs(z0) < r][0]
                C = FM.subs(z, x)
            else:
                C = FM.subs(z, 0)
        
            c, k = coef(C)
            U_= find_U_inv(n, k, x, c)
            V = find_V(n, k, x, c, lst)
            
            lst[k] += 1
            P, lst = perm(lst)
            L = diag([z ** k for k in lst], unpack = True)
            
            FM = factor(cancel(FM @ U_ @ P))
            Fm = factor(cancel(P @ V @ Fm))
        
    elif type == "right":
        for i in reversed(range(m)):
            if FM.det() != 0:
                x = [z0 for z0 in solve(FM.det(), z) if abs(z0) < r][0]
                C = FM.subs(z, x)
            else:
                C = FM.subs(z, 0)
            
            c, k = coef(C.T)
            U_= find_U_inv(n, k, x, c).T
            V = find_V(n, k, x, c, lst).T
            
            lst[k] += 1
            P, lst = perm(lst)
            L = diag([z ** k for k in lst], unpack = True)

            FM = factor(cancel(P @ U_ @ FM))
            Fm = factor(cancel(Fm @ V @ P))

    return simplify(FM), L, simplify(Fm)




#a modo de ejemplo:
F = simplify(z**2 * Matrix(([1, (z + 1/z)/2], [(z + 1/z)/2, (((z + 1/z)/2)**2) + 1])))

fact = fact(F, r=1, type = "left")

#verificación:
simplify(fact[0]@fact[1]@fact[2] - F)
import numpy as np
import numpy.random as rnd
import scipy.signal as sp
import scipy.linalg as la
import cvxpy as cv

class dtss(sp.ltisys.StateSpaceDiscrete):
    def append_dt(self,P):
        return dtss(P.A,P.B,P.C,P.D,dt=self.dt)
        
    def __add__(self,other):
        P = super().__add__(other)
        return self.append_dt(P)
    
    def __mul__(self,other):
        P = super().__mul__(other)
        return self.append_dt(P)

    def __rmul__(self,other):
        P = super().__rmul__(other)
        return self.append_dt(P)
 
    
    def __neg__(self):
        P = super().__neg__()
        return self.append_dt(P)

def extract_matrices(P):
    return P.A, P.B, P.C, P.D

def h2norm(P):
    A,B,C,D = extract_matrices(P)
    X = la.solve_discrete_lyapunov(A,B@B.T)
    gam = np.trace(C@X@C.T + D@D.T)
    return np.sqrt(gam)
    
    
def hinfnorm(P):
    A,B,C,D = extract_matrices(P)
    nX,nU = B.shape
    X = cv.Variable((nX,nX),PSD=True)
    gam = cv.Variable(nonneg=True)

    H = cv.vstack([cv.hstack([A.T*X*A-X + C.T@C, A.T*X*B + C.T@D]),
                   cv.hstack([B.T*X*A + D.T@C, B.T*X*B + D.T@D - gam * np.eye(nU)])])
    prob = cv.Problem(cv.Minimize(gam),[H << 0])
    prob.solve()
    return np.sqrt(gam.value)

def controllabilityMatrix(A,B):
    n = len(A)
    M = np.copy(B)
    Columns = []
    for _ in range(n):
        Columns.append(M)
        M = A @ M
    return np.hstack(Columns)

def observabilityMatrix(A,C):
    return controllabilityMatrix(A.T,C.T).T

def minreal(P,tol=1e-12):
    A,B,C,D = extract_matrices(P)
    Con = controllabilityMatrix(A,B)
    Obs = observabilityMatrix(A,C)

   
   
    U,S,Vt = la.svd(Obs@Con)

    nX = sum(S > tol)

    nY,nU = D.shape

    Sig = np.diag(np.sqrt(S[:nX]))

    Umin = U[:,:nX]
    Vtmin = Vt[:nX]
    ObsMin = Umin@Sig
    ConMin = Sig@Vtmin

   
    Cmin = ObsMin[:nY]
    Bmin = ConMin[:,:nU]

    
    SigInv = np.diag(1/np.sqrt(S[:nX]))


    H = Obs@A@Con

    Amin = SigInv @ Umin.T @ H @ Vtmin.T @ SigInv

   
    return dtss(Amin,Bmin,Cmin,D,dt=P.dt)

specrad = lambda A : np.max(np.abs(la.eigvals(A)))

def drss(nX,nU,nY,maxeig = .9,dt=1):
    """
    Required Parameters:
    nX - State Dimension
    nU - Input Dimension
    nY - Output Dimension

    Optional Parameters
    maxeig - Maximum spectral radius (default: 0.9)
    dt - Time Step (default: 1)
    """
    A = rnd.randn(nX,nX)
    u = rnd.rand() * maxeig
    A = A * u / specrad(A)
    B = rnd.randn(nX,nU)
    C = rnd.randn(nY,nX)
    D = rnd.randn(nY,nU)
    return dtss(A,B,C,D,dt=dt)

def similarityTransform(P,T):
    Tinv = la.inv(T)
    A,B,C,D = extract_matrices(P)

    Anew = la.solve(T,A@T)
    Bnew = la.solve(T,B)
    Cnew = C@T
    Dnew = D

    return dtss(Anew,Bnew,Cnew,Dnew,dt=P.dt)

def controllabilityGramian(A,B):
    return la.solve_discrete_lyapunov(A,B@B.T)

def observabilityGramian(A,C):
    return controllabilityGramian(A.T,C.T)

def balancedRealization(P):
    Pmin = minreal(P)
    A,B,C,D = extract_matrices(Pmin)

    Xc = controllabilityGramian(A,B)
    Yo = observabilityGramian(A,C)

    Ux,Sx,Vtx = la.svd(Xc)

    Xrt = Ux @ np.diag(np.sqrt(Sx)) @ Vtx

    M = Xrt @ Yo @ Xrt

    U,S,Vt = la.svd(M)

    T = Xrt @ U @ np.diag(S**(-1/4))

    return similarityTransform(Pmin,T)

def inv(P):
    A,B,C,D = extract_matrices(P)
    Dinv = la.inv(D)
    Ainv = A-B@Dinv@C
    Binv = B@Dinv
    Cinv = -Dinv@C

    return dtss(Ainv,Binv,Cinv,Dinv,dt=P.dt)

def solve_discrete_are(A,B,Q,R,S=None):
    if S is None:
        return la.solve_discrete_are(A,B,Q,R)
    else:
        G = la.solve(R,S.T)
        Anew = A - B@G
        Qnew = Q - G.T@R@G
        return la.solve_discrete_are(Anew,B,Qnew,R)

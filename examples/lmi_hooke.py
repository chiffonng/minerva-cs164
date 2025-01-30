import numpy as np
import cvxpy as cvx

# Parameters

rho = 7500  # Bar density in kg / m^3
E = 190e9  # Young's modulus in Pascals
theta = np.pi / 4  # Angle in radians
l1 = 0.4  # length of bar 1 in meters
l2 = 0.4 * np.sqrt(2)  # length of bar 2 in meters
mmax = 100  # Maximum mass

# Applied force
F = np.array([[2000], [-500]])

# Define the stiffness matrix

uvec = lambda theta: np.array([[np.cos(theta)], [np.sin(theta)]])
K = lambda v: E * (
    v[0] / l1**2 * np.array([[1, 0], [0, 0]])
    + v[1] / l2**2 * uvec(theta) @ uvec(theta).T
)
v = cvx.Variable(2, nonneg=True)  # Volumes (must be nonnegative)
t = cvx.Variable((1, 1), nonneg=True)  # Slack variable

# LMI constraint
# Note that we need to scale by E to avoid numerical problems. This doesn't affect the LMI, since if A is PSD, then so is kA for any positive k.
constraint = [1 / E * cvx.bmat([[K(v), F], [F.T, t]]) >> 0]

# Overall mass constraint
constraint += [(rho * cvx.sum(v) <= mmax)]
prob = cvx.Problem(cvx.Minimize(t), constraint)
prob.solve(solver=cvx.SCS)
print(v.value)

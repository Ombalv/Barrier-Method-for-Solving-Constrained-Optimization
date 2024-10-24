# -*- coding: utf-8 -*-
# Part 1

import jax
import jax.numpy as jnp

# compute the logarithmic barrier for the barrier method
def compute_f_obj(f,f_ineq,t):
  # original objective function
  # f_ineq: lst of inequality constraints functions
  # t: time
  # return objective function f_obj=t*f_0+phi of centering step
  def sum_of_functions(x):
    result = 0
    for ineq in f_ineq:
        result += jnp.log(-ineq(x))
    return -result

  def new_function(x):
      return  t * f(x) + sum_of_functions(x)

  return new_function

  def phi_func(x):
      result = 0
      for ineq in f_ineq:
          result += jnp.log(-ineq(x))
      return result
  return lambda x: t * f(x) - phi_func(x)

# compute the Hessian(second derivative) of certain function 'f(x)'
def Hessian(f):
  # f: objective function
  return jax.hessian(f)
  # return jax.jacfwd(jax.grad(f))

def backtracking_line_search(f,g,x,Dx,alpha=0.1,beta=0.5): # f_ineq
  # f: the objective function
  # g: gradient of f
  # x: initial point
  # alpha, beta: line search parameters
  # Dx: descent direction = -jax.grad(f)(x)
  # initialize counter and max number of iterations and T
  counter = 0
  max_iteration = 100
  t=1.0 # Step Size
  # must ensure the x+T*Dx is feasible
  # For barrier method, jnp.isnan(f(x+T*Dx)) is sufficient because log(negative)=nan
  # In general, the following is likely better (written for one constraint):
  # while f_ineq(x+T*Dx)>0:
  #   T=beta * T
  while jnp.isnan(f(x+t*Dx)):
    t=beta*t
  # run while loop until Armijo-Goldstein inequality is met
  while f(x+t*Dx) > f(x) + alpha * t * jnp.dot(g(x), Dx):
    counter += 1
    t = beta * t
    # if counter > max_iteration:
    #   break
  return t

def solve_KKT_system(Q,g,x,A):
  # Q: Hessian of f(x)
  # g: gradient of f(x)
  # Ax=b is equality constraint
  # p=len(A)
  p = A.shape[0]
  # construct KKT matrix
  # print("Q shape", Q(x).shape)
  # print("x", x)
  # print("Q(x)", Q(x))
  # print("A.T shape", jnp.transpose(A).shape)
  # print("A shape", A.shape)
  # print("jnp.zeros shape", jnp.zeros((p,p)).shape)
  KKT_matrix=jnp.block([
      [Q(x),jnp.transpose(A)],
      [A,jnp.zeros((p,p))]
      ])
  # constrcut the KKT vector
  # b = jnp.zeros((1,p))
  b = jnp.zeros(p)
  KKT_vector = jnp.concatenate([-g(x),b],axis=0)
  solution= jax.scipy.linalg.solve(KKT_matrix, KKT_vector)
  x_nt, _ = solution[:-p], solution[-p:]
  return x_nt

def newtons_method_unconstrained(f,x,eps,alpha=0.1,beta=0.5):
  # f: objective function
  # x: initial point
  # eps: suitable stopping criterion
  # alpha, beta: backtracking parameters
  # A: constrained equality
  counter=0
  max_iterations = 100
  Q = Hessian(f)
  g = jax.grad(f)
  for _ in range(max_iterations):
    x_nt = -jnp.linalg.solve(Q(x),g(x))
    # print(x_nt)
    # print(-jnp.linalg.solve(Q,g))
    # h_square = jnp.dot(-jnp.transpose(g),x_nt)
    h_square = jnp.dot(g(x), jnp.linalg.solve(Q(x), g(x)))
    # h_square2 = jnp.dot(jnp.dot(jnp.transpose(g),jnp.linalg.inv(Q)),g)
    if 0.5 * h_square <= eps:
      break

    t = backtracking_line_search(f,g,x,x_nt,alpha,beta)
    x = x + t * x_nt
  return x

def newtons_method_eq_constrained(f,x,eps,alpha,beta,A):
  # f: objective function
  # x: initial point
  # eps: suitable stopping criterion
  # alpha, beta: backtracking parameters
  # A: constrained equality
  counter=0
  max_iterations = 100
  # print(inspect.getsource(f))
  Q = Hessian(f)
  # print(inspect.getsource(Q))
  g = jax.grad(f)
  for _ in range(max_iterations):
    x_nt = solve_KKT_system(Q,g,x,A)
    # print(x_nt)
    h_square = jnp.dot(x_nt, jnp.dot(Q(x),x_nt))

    if (0.5 * h_square <= eps):
      break

    t = backtracking_line_search(f,g,x,x_nt,alpha,beta)
    x = x + t * x_nt #[:len(x)]
  return x

def barrier_method(f,x,mu=2.0,eps=1e-6,alpha=0.1,beta=0.5,A=None,f_ineq=None):
  # f: objective function
  # x: initial point
  # mu: multiplier
  # eps: stopping criterion
  # alpha, beta: backtracking parameters
  # Ax=b: equality constraint
  # f_ineq: list of inequality constraints functions

  # neither inequality nor equality constraints
  if (f_ineq == None and A == None): # unconstrained optimization problem, using unconstrained newton's method
    return newtons_method_unconstrained(f,x,eps,alpha,beta)

  # no equality constraints
  elif (f_ineq != None and A == None):
    # turn inequality constrained optimization problem into unconstrained optimization problem(f_obj=t*f+phi), then solve it with unconstrained newton's method
    t = 1.0
    counter = 0
    max_iteration = 100
    m = len(f_ineq) # number of inequality constraints
    # phi = compute_phi(f_ineq)
    # f_obj = lambda x: t * f(x) + phi(x)
    while True:
      f_obj = compute_f_obj(f,f_ineq,t)
      # f_obj = lambda x: f_obj(x)
      # print(Hessian(f_obj)(x))
      x = newtons_method_unconstrained(f_obj,x,eps,alpha,beta)
      if (m / eps < t):
        break
      t = mu * t
    return x

  # no inequality constraints
  elif (f_ineq == None and A != None): # equality constrained optimization problem, using equality constrained newton's method
    return newtons_method_eq_constrained(f,x,eps,alpha,beta,A)

  # optimization problem with equality and inequality constraints
  elif (f_ineq != None and A != None):
    t = 1.0
    counter = 0
    max_iteration = 100
    m = len(f_ineq) # number of inequality constraints
    # phi = compute_phi(f_ineq)
    # f_obj = lambda x: t * f(x) + phi(x)

    while True:
      f_obj = compute_f_obj(f,f_ineq,t)
      # f_obj = lambda x: f_obj(x)
      # print(Hessian(f_obj)(x))
      x = newtons_method_eq_constrained(f_obj,x,eps,alpha,beta,A)
      if (m / eps < t):
        break
      t = mu * t
    return x


"""testcase with both equality constraints and inequality constraints"""
# objective function
def objective_function(x):
    return jnp.sum(x**2)

f_ineq = [lambda x:-x[0], lambda x:-x[1], lambda x:x[0]-4, lambda x:x[1]-4]

# equality constraint
A = jnp.array([[1.0, 1.0]])
b = jnp.array([1.0])

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([0.9, 0.1])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq)
print("Optimal Solution", solution)
print("Optimal Value", objective_function(solution))


"""testcase with neither equality constraints nor inequality constraints"""
def objective_function(x):
  return jnp.sum(x**2)

x_initial = jnp.array([11.0,10.0])
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
f_ineq = None
A = None
optimal_solution = barrier_method(objective_function,x_initial,mu,eps,alpha,beta,A,f_ineq)
print("Optimal solution:", optimal_solution)
print("Objective value at optimal solution:", objective_function(optimal_solution))


"""testcase with no inequality constraints but with equality constraints"""
def objective_function(x):
    return jnp.sum(x**2)

x_initial = jnp.array([0.3,0.7])
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
f_ineq = None
A = jnp.array([[1.0, 1.0]])
b = jnp.array([1.0])
optimal_solution = barrier_method(objective_function,x_initial,mu,eps,alpha,beta,A,f_ineq)
print("Optimal solution:", optimal_solution)
print("Objective value at optimal solution:", objective_function(optimal_solution))


"""testcase with no equality constraints but with inequality constraints"""
def objective_function(x):
    return jnp.sum(x**2)

f_ineq = [lambda x:-x[0]-1, lambda x:-x[1]-1, lambda x:x[0]-4, lambda x:x[1]-4]
A = None

initial_x = jnp.array([1.0, 2.0])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq)
print("Optimal Solution", solution)
print("Optimal Value", objective_function(solution))
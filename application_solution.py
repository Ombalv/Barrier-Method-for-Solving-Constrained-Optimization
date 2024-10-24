# -*- coding: utf-8 -*-
# Part 2
import jax
import numpy as np
import jax.numpy as jnp
from final_part_1 import barrier_method

# Objective Function of Optimization Problem
def objective_function(x):
  f = (x[0] - 2)**2 + 3*(x[1] - 1)**2 + jnp.exp(2*x[0] - 3*x[1])
  return f

"""# Tiling B (First Tiling)

Green Individual 1 (left bottom corner)
"""

# 1
# Compute the green tetromino on the left bottom corner
# split it into two part, vertical three squares and the one on the right to get the minimum

# vertical three squares
f_ineq_g1_vertical3 = [lambda x:-x[0], lambda x:-x[1], lambda x:x[0]-1, lambda x:x[1]-3]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([0.5, 0.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g1_vertical3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g1_vertical3)
print("Optimal Solution_g1_vertical3", solution_g1_vertical3)
print("Optimal Value_g1_vertical3", objective_function(solution_g1_vertical3))

# right 1 square
f_ineq_g1_right1 = [lambda x:-x[0]+1, lambda x:-x[1]+1, lambda x:x[0]-2, lambda x:x[1]-2]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([1.5, 1.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g1_right1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g1_right1)
print("Optimal Solution_g1_right1", solution_g1_right1)
print("Optimal Value_g1_right1", objective_function(solution_g1_right1))

"""Green Individual 2 (middle bottom)"""

# 2
# Compute the green tetromino on the middle of the bottom
# split it into two part, horizontal three squares and the one on the top to get the minimum

# horizontal three squares
f_ineq_g2_horizontal3 = [lambda x:-x[0]+1, lambda x:-x[1], lambda x:x[0]-4, lambda x:x[1]-1]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.0, 0.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g2_horizontal3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g2_horizontal3)
print("Optimal Solution_g2_horizontal3", solution_g2_horizontal3)
print("Optimal Value_g2_horizontal3", objective_function(solution_g2_horizontal3))

# top 1 square
f_ineq_g2_top1 = [lambda x:-x[0]+2, lambda x:-x[1]+1, lambda x:x[0]-3, lambda x:x[1]-2]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.5, 1.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g2_top1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g2_top1)
print("Optimal Solution_g2_top1", solution_g2_top1)
print("Optimal Value_g2_top1", objective_function(solution_g2_top1))

"""Red Individual 1 (left top corner)"""

# 3
# Compute the red tetromino on the left top corner
# split it into two part, top 2 squares and the bottom 2 squares to get the minimum

# top 2 squares
f_ineq_r1_top2 = [lambda x:-x[0], lambda x:-x[1]+3, lambda x:x[0]-2, lambda x:x[1]-4]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([1.0, 3.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_r1_top2 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_r1_top2)
print("Optimal Solution_r1_top2", solution_r1_top2)
print("Optimal Value_r1_top2", objective_function(solution_r1_top2))


# bottom 2 squares
f_ineq_r1_bottom2 = [lambda x:-x[0]+1, lambda x:-x[1]+2, lambda x:x[0]-3, lambda x:x[1]-3]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.0, 2.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_r1_bottom2 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_r1_bottom2)
print("Optimal Solution_r1_bottom2", solution_r1_bottom2)
print("Optimal Value_r1_bottom2", objective_function(solution_r1_bottom2))

"""Yellow Individual 1 (right top)"""

# 4
# Compute the yellow tetromino on the right top corner
# split it into two part, vertical 3 squares and the left 1 square to get the minimum

# vertical 3 squares
f_ineq_y1_vertical3 = [lambda x:-x[0]+3, lambda x:-x[1]+1, lambda x:x[0]-4, lambda x:x[1]-4]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([3.5, 2.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_y1_vertical3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_y1_vertical3)
print("Optimal Solution_y1_vertical3", solution_y1_vertical3)
print("Optimal Value_y1_vertical3", objective_function(solution_y1_vertical3))


# left 1 square
f_ineq_y1_left1 = [lambda x:-x[0]+2, lambda x:-x[1]+3, lambda x:x[0]-3, lambda x:x[1]-4]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.5, 3.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_y1_left1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_y1_left1)
print("Optimal Solution_y1_left1", solution_y1_left1)
print("Optimal Value_y1_left1", objective_function(solution_y1_left1))

"""Blue Individual (right corner)"""

# 5
# Compute the blue tetromino on the right corner

f_ineq_b1 = [lambda x:-x[0]+4, lambda x:-x[1], lambda x:x[0]-5, lambda x:x[1]-4]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([4.5, 2.0])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_b1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_b1)
print("Optimal Solution_b1", solution_b1)
print("Optimal Value_b1", objective_function(solution_b1))

# Optimal Solution_g1_vertical3 [0.9999997 1.1260298]
# Optimal Value_g1_vertical3 1.2997105 SAME
# Optimal Solution_g1_right1 [1.5102979 1.2448512]
# Optimal Value_g1_right1 0.9093666 SAME

# Optimal Solution_g2_horizontal3 [1.3125888 0.9999999]
# Optimal Value_g2_horizontal3 1.1599458 SAME
# Optimal Solution_g2_top1 [2.0000002 1.4042181]
# Optimal Value_g2_top1 1.2986128 SAME

# Optimal Solution_r1_top2 [1.9932497 3.0000002]
# Optimal Value_r1_top2 12.006696 SAME
# Optimal Solution_r1_bottom2 [1.8911426 2.0000002]
# Optimal Value_r1_bottom2 3.1207087 SAME

# Optimal Solution_y1_vertical3 [3.0000002 1.8307571]
# Optimal Value_y1_vertical3 4.7319865 SAME
# Optimal Solution_y1_left1 [2.0000215 3.0000002]
# Optimal Value_y1_left1 12.006742 SAME

# Optimal Solution_b1 [4.0000005 2.3384478]
# Optimal Value_b1 12.051224

# 0.9093666 + 1.1599458 + 3.1207087 + 4.7319865 + 12.051224



"""# Tiling A (Second Tiling)"""

# Objective Function of Optimization Problem
def objective_function(x):
  f = (x[0] - 2)**2 + 3*(x[1] - 1)**2 + jnp.exp(2*x[0] - 3*x[1])
  return f

"""Yellow Individual 1 (left bottom)"""

# 1
# Compute the yellow tetromino on the left bottom corner
# split it into two part, vertical 3 squares and the right 1 square to get the minimum

# vertical 3 squares
f_ineq_y1_vertical3 = [lambda x:-x[0], lambda x:-x[1], lambda x:x[0]-1, lambda x:x[1]-3]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([0.5, 1.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_y1_vertical3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_y1_vertical3)
print("Optimal Solution_y1_vertical3", solution_y1_vertical3)
print("Optimal Value_y1_vertical3", objective_function(solution_y1_vertical3))


# right 1 square
f_ineq_y1_right1 = [lambda x:-x[0]+1, lambda x:-x[1], lambda x:x[0]-2, lambda x:x[1]-1]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([1.5, 0.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_y1_right1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_y1_right1)
print("Optimal Solution_y1_right1", solution_y1_right1)
print("Optimal Value_y1_right1", objective_function(solution_y1_right1))

"""Yellow Individual 2 (right bottom)"""

# 2
# Compute the yellow tetromino on the right bottom corner
# split it into two part, vertical 3 squares and the left 1 square to get the minimum

# vertical 3 squares
f_ineq_y2_vertical3 = [lambda x:-x[0]+4, lambda x:-x[1], lambda x:x[0]-5, lambda x:x[1]-3]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([4.5, 1.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_y2_vertical3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_y2_vertical3)
print("Optimal Solution_y2_vertical3", solution_y2_vertical3)
print("Optimal Value_y2_vertical3", objective_function(solution_y2_vertical3))


# left 1 square
f_ineq_y2_left1 = [lambda x:-x[0]+3, lambda x:-x[1], lambda x:x[0]-4, lambda x:x[1]-1]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([3.5, 0.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_y2_left1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_y2_left1)
print("Optimal Solution_y2_left1", solution_y2_left1)
print("Optimal Value_y2_left1", objective_function(solution_y2_left1))

"""Green Individual 1 (middle bottom)"""

# 3
# Compute the green tetromino on the middle bottom corner
# split it into two part, horizontal 3 squares and the bottom 1 square to get the minimum

# horizontal 3 squares
f_ineq_g1_horizontal3 = [lambda x:-x[0]+1, lambda x:-x[1]+1, lambda x:x[0]-4, lambda x:x[1]-2]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.5, 1.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g1_horizontal3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g1_horizontal3)
print("Optimal Solution_g1_horizontal3", solution_g1_horizontal3)
print("Optimal Value_g1_horizontal3", objective_function(solution_g1_horizontal3))


# bottom 1 square
f_ineq_g1_bottom1 = [lambda x:-x[0]+2, lambda x:-x[1], lambda x:x[0]-3, lambda x:x[1]-1]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.5, 0.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g1_bottom1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g1_bottom1)
print("Optimal Solution_g1_bottom1", solution_g1_bottom1)
print("Optimal Value_g1_bottom1", objective_function(solution_g1_bottom1))

"""Green Individual 2 (right top)"""

# 4
# Compute the green tetromino on the right top corner
# split it into two part, horizontal 3 squares and the bottom 1 square to get the minimum

# horizontal 3 squares
f_ineq_g2_horizontal3 = [lambda x:-x[0]+2, lambda x:-x[1]+3, lambda x:x[0]-5, lambda x:x[1]-4]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([3.5, 3.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g2_horizontal3 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g2_horizontal3)
print("Optimal Solution_g2_horizontal3", solution_g2_horizontal3)
print("Optimal Value_g2_horizontal3", objective_function(solution_g2_horizontal3))


# bottom 1 square
f_ineq_g2_bottom1 = [lambda x:-x[0]+3, lambda x:-x[1]+2, lambda x:x[0]-4, lambda x:x[1]-3]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([3.5, 2.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_g2_bottom1 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_g2_bottom1)
print("Optimal Solution_g2_bottom1", solution_g2_bottom1)
print("Optimal Value_g2_bottom1", objective_function(solution_g2_bottom1))

"""Red Individual (left top)"""

# 5
# Compute the red tetromino on the left top corner
# split it into two part, top 2 squares and the bottom 2 squares to get the minimum

# top 2 squares
f_ineq_r1_top2 = [lambda x:-x[0], lambda x:-x[1]+3, lambda x:x[0]-2, lambda x:x[1]-4]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([1.0, 3.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_r1_top2 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_r1_top2)
print("Optimal Solution_r1_top2", solution_r1_top2)
print("Optimal Value_r1_top2", objective_function(solution_r1_top2))


# bottom 2 squares
f_ineq_r1_bottom2 = [lambda x:-x[0]+1, lambda x:-x[1]+2, lambda x:x[0]-3, lambda x:x[1]-3]

# equality constraint
A = None

# Set initial guess, initial t, multiplier mu, and tolerance epsilon
initial_x = jnp.array([2.0, 2.5])
initial_t = 1.0
mu = 2.0
eps = 1e-6
alpha = 0.1
beta = 0.5
solution_r1_bottom2 = barrier_method(objective_function, initial_x, mu, eps, alpha, beta, A, f_ineq_r1_bottom2)
print("Optimal Solution_r1_bottom2", solution_r1_bottom2)
print("Optimal Value_r1_bottom2", objective_function(solution_r1_bottom2))

# Optimal Solution_y1_vertical3 [0.9999997 1.1260298]
# Optimal Value_y1_vertical3 1.2997105
# Optimal Solution_y1_right1 [1.3125887 0.9999999]
# Optimal Value_y1_right1 1.1599458

# Optimal Solution_y2_vertical3 [4.0000005 2.3384476]
# Optimal Value_y2_vertical3 12.051225
# Optimal Solution_y2_left1 [3.0000002 0.9999999]
# Optimal Value_y2_left1 21.085556

# Optimal Solution_g1_horizontal3 [1.510298  1.2448512]
# Optimal Value_g1_horizontal3 0.9093666
# Optimal Solution_g1_bottom1 [2.0000002  0.99999994]
# Optimal Value_g1_bottom1 2.7182837

# Optimal Solution_g2_horizontal3 [2.0000207 3.0000002]
# Optimal Value_g2_horizontal3 12.006742
# Optimal Solution_g2_bottom1 [3.0000002 2.0000002]
# Optimal Value_g2_bottom1 5.0000014

# Optimal Solution_r1_top2 [1.9932497 3.0000002]
# Optimal Value_r1_top2 12.006696
# Optimal Solution_r1_bottom2 [1.8911426 2.0000002]
# Optimal Value_r1_bottom2 3.1207087

# 1.1599458 + 12.051225 + 0.9093666 + 5.0000014 + 3.1207087
# Check f_impl, meta information, x_dot equations
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, diag, solve, cross

def quadrotor_model_auto() -> AcadosModel:
    """
    Create a quadrotor model for acados MPC
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    - Position: (x, y, z)
    - Velocity: (vx, vy, vz) 
    - Attitude: (qw, qx, qy, qz)
    - Angular rates: (p, q, r)
    
    Control vector: [f, tau_phi, tau_theta, tau_psi]
    - f: total thrust
    - tau_phi, tau_theta, tau_psi: torques around body axes
    """
    
    # --- Parameters and Initial Values ---
    L = 0.0397 # (m) Distance from COM to rotor axis
    M = 0.027 # (kg) Total mass
    G = 9.81 # (m/s^2) Gravity
    I = diag(SX([1.4e-5, 1.4e-5, 2.17e-5])) # kg * m^2

    # --- State Vector ---
    r = SX.sym('r', 3)
    v = SX.sym('v', 3)
    q = SX.sym('q', 4)
    w = SX.sym('w', 3)
    x = vertcat(r, v, q, w)

    # --- Control Vector ---
    u_1 = SX.sym('u_1')
    tau = SX.sym('tau', 3)
    u = vertcat(u_1, tau)

    # --- Kinematics Related ---
    # Using Quaternion Kinematics
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    a_R_b = vertcat(horzcat(1 - 2*(qy * qy + qz * qz),      2*(qx * qy - qw * qz),    2*(qx * qz + qw * qy)),
                    horzcat(  2*(qx * qy + qw * qz),    1 - 2*(qx * qx + qz * qz),      2*(qy * qz - qw * qx)),
                    horzcat(  2*(qx * qz - qw * qy),      2*(qy * qz + qw * qx),    1 - 2*(qx * qx + qy * qy)))

    w_p, w_q, w_r = w[0], w[1], w[2]
    Omega = vertcat(horzcat(  0, -w_p,  -w_q, -w_r),
                    horzcat(w_p,    0,   w_r, -w_q),
                    horzcat(w_q, -w_r,     0,  w_p),
                    horzcat(w_r,  w_q,  -w_p,    0))

    # --- Dynamics ---
    r_dot = v # dp = v
    v_dot = vertcat(0, 0, -G) + (u_1 / M) * (a_R_b @ vertcat(0, 0, 1)) # Take M out
    q_dot = 0.5 * (Omega @ q)
    w_dot = solve(I, (tau - cross(w, I @ w)))

    x_dot = vertcat(r_dot, v_dot, q_dot, w_dot)

    # --- Build Model ---
    model = AcadosModel()
    model.name = 'quadrotor'

    model.x = x
    model.u = u
    model.xdot = SX.sym('xdot', x.size1())

    model.f_expl_expr = x_dot
    model.f_impl_expr = model.xdot - x_dot

    return model

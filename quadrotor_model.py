# Check f_impl, meta information, x_dot equations
from acados_template import AcadosModel
from casadi import SX, vertcat, horzcat, diag, cross

from constants import constants

def quadrotor_model_auto() -> AcadosModel:
    """
    Create a quadrotor model for acados MPC
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    - r: Position: (x, y, z)
    - v: Velocity: (vx, vy, vz) 
    - q: Attitude: (qw, qx, qy, qz)
    - w: Angular rates: (p, q, r)
    
    Control vector: [f, tau_phi, tau_theta, tau_psi]
    - f: total thrust
    - tau_phi, tau_theta, tau_psi: torques around body axes
    """
    
    # --- Parameters and Initial Values ---
    const = constants()

    INERTIA = diag(SX([const.I_11, const.I_22, const.I_33])) # kg * m^2
    INERTIA_INV = diag(SX([1.0 / const.I_11, 1.0 / const.I_22, 1.0 / const.I_33]))

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

    # --- Kinematics ---
    # Orientation Kinematics (Unit Quaternion)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # World/inertial frame A, Body frame B
    a_R_b = vertcat(
        horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
        horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
        horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
    )

    # Body Angular Velocity
    w_p, w_q, w_r = w[0], w[1], w[2]
    Omega = vertcat(
        horzcat(  0, -w_p,  -w_q, -w_r),
        horzcat(w_p,    0,   w_r, -w_q),
        horzcat(w_q, -w_r,     0,  w_p),
        horzcat(w_r,  w_q,  -w_p,    0)
    )


    # --- Rigid Body Dynamics ---
    r_dot = v 
    v_dot = vertcat(0, 0, -const.g) + (1 / const.mass) * (a_R_b @ vertcat(0, 0, u_1))
    q_dot = 0.5 * (Omega @ q)
    w_dot = INERTIA_INV @ (tau - cross(w, INERTIA @ w))
    x_dot = vertcat(r_dot, v_dot, q_dot, w_dot)

    xdot_sym = SX.sym('xdot', 13)

    # --- Build Dynamics Model ---
    model = AcadosModel()
    model.name = 'quadrotor'

    model.x = x 
    model.xdot = xdot_sym
    model.u = u
    model.f_expl_expr = x_dot
    model.f_impl_expr = xdot_sym - x_dot

    return model

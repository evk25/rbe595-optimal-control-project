import numpy as np
import math

from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
from quadrotor_model import quadrotor_model_auto
import casadi as ca

from trajectories import *
from constants import constants

class MPC:
    """Model predictive controller for the crazyflie quadrotor"""
    def __init__(self, target_height):
        # TUNABLE PARAMS
        self.HORIZON = 1 # second
        self.STEPS = 10 # look ahead distance 
        self.dt = self.HORIZON / self.STEPS

        self.c = constants()


        # Create Reference linear trajectory (origin to [1, 1, target height])
        # p_traj = linear_trajectory(
        #     'quintic',
        #     (0.0, 10.0),
        #     self.dt,
        #     ([0, 0, 0,], [1, 1, target_height]),
        #     ([0, 0, 0], [0, 0, 0]),
        #     ([0, 0, 0], [0, 0, 0])
        # )

        p_traj = circular_trajectory(
            t=(0.0, 10.0),
            dt=self.dt,
            center=(0.0, 0.0, target_height),
            radius=1.0,
            omega=2.0 * np.pi / 10.0,
        )

        self.sim_steps = len(p_traj['t']) - 1
        Nsim = self.sim_steps + 1



        # Reference trajectory Z_des(t_k) = [r_des(t_k), phi_des(t_k)]
        # We will track inertial position and a constant yaw (0) first
        q_ref = np.tile([1.0, 0.0, 0.0, 0.0], (Nsim, 1))
        w_ref = np.zeros((Nsim, 3))

        self.trajectory = np.hstack([p_traj['p'], p_traj['v'], q_ref, w_ref])

        # Logging for plots
        self.simX1 = np.zeros_like(self.trajectory)


        Nsim = self.sim_steps + 1
        self.log = {
            "t":        np.arange(Nsim) * self.dt,
            "r_ref":    np.zeros((Nsim, 3)),
            "r_meas":   np.zeros((Nsim, 3)),
            "yaw_ref":  np.zeros((Nsim, )),
            "yaw_meas": np.zeros((Nsim, )),
            "u":        np.zeros((Nsim, 4)),   # [thrust, tau_x, tau_y, tau_z]
            "e_r":      np.zeros((Nsim, 3)),
            "e_yaw":    np.zeros((Nsim, )),
        }

        #print("Shape of simX1", self.simX1.shape)

        # Create NLOP in Acados
        self.ocp = self.create_ocp()
        self.ocp_solver = AcadosOcpSolver(self.ocp)

    def solve(self, step, current_state):
        self.ocp_solver.constraints_set(0, 'lbx', current_state)
        self.ocp_solver.constraints_set(0, 'ubx', current_state)
        self.ocp_solver.set(0, 'x', current_state)

        # Build refernces for the horizon steps
        traj = self.trajectory
        traj_len = traj.shape[0] - 1
        hover_thrust = self.c.mass * self.c.g

        q_curr = current_state[6:10]

        for k in range(self.STEPS):
            idx = min(step + k + 1, traj_len)
            yref = np.zeros(17)
            yref[0:13] = traj[idx, 0:13]
            if np.dot(q_curr, yref[6:10]) < 0.0:
                yref[6:10] = -yref[6:10]
            yref[13] = hover_thrust
            self.ocp_solver.cost_set(k, "yref", yref)


        # Terminal Costs
        idx_T = min(step + self.STEPS, traj_len)
        yref_T = traj[idx_T, 0:13].copy() # r, v, q, w
        if np.dot(q_curr, yref_T[6:10]) < 0.0:
            yref_T[6:10] = -yref_T[6:10]
        self.ocp_solver.cost_set(self.STEPS, "yref", yref_T)


        status = self.ocp_solver.solve()
        if status != 0:
            print("Solve Status from acados: ", status)

        u0 = self.ocp_solver.get(0, "u") # return first control values
        return {
            'thrust': float(u0[0]),
            'moment_x': float(u0[1]),
            'moment_y': float(u0[2]), 
            'moment_z': float(u0[3])
        }
    
    def create_ocp(self):         
        ocp = AcadosOcp()
        model = quadrotor_model_auto()
        ocp.model = model

        # MPC Gain tuning
        w_pos = 20 # was 10
        w_vel = 2 # was 2.0
        w_att = 4 # was 4.0
        w_omega = 0.5
        w_u=1e-3 # minimize chatter (some small number)

        n_x = model.x.size()[0] # R in 12

        # Set up QP From the dynamics model:
        x = model.x
        u = model.u

        r = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13] 

        # Cost Function formulation
        y = ca.vertcat(r, v, q, w, u)
        y_e = ca.vertcat(r, v, q, w)

        W = np.diag([
            w_pos, w_pos, w_pos,
            w_vel, w_vel, w_vel,
            w_att, w_att, w_att, w_att,
            w_omega, w_omega, w_omega,
            w_u, w_u, w_u, w_u
        ])
        
        W_e = np.diag([w_pos, w_pos, w_pos,
                       w_vel, w_vel, w_vel,
                       w_att, w_att, w_att, w_att,
                       w_omega, w_omega, w_omega])
        
        ocp.cost.yref = np.zeros(W.shape[0])
        ocp.cost.yref[6] = 1.0
        ocp.cost.yref[13] = self.c.mass * self.c.g # hover thrust

        ocp.cost.yref_e = np.zeros(W_e.shape[0])
        ocp.cost.yref_e[6] = 1.0

        
        # Constraints (with params from the model)             
        u1_min, u1_max = 0.0, 4.0*self.c.mu # thrust
        tau_x_max = np.sqrt(2.0) * self.c.robot['arm'] * self.c.mu # body rates
        tau_y_max = np.sqrt(2.0) * self.c.robot['arm'] * self.c.mu
        tau_z_max = 2.0 * (self.c.robot['km'] / self.c.robot['kf']) * self.c.mu

        roll_max = math.pi 
        pitch_max = math.pi

        rotor_min, rotor_max = 0.0, self.c.omega_max 

        # Add constraints to QP
        ocp.constraints.lbu = np.array([
            u1_min, 
            -tau_x_max, 
            -tau_y_max, 
            -tau_z_max])
        ocp.constraints.ubu = np.array([
            u1_max,  
            tau_x_max,  
            tau_y_max,  
            tau_z_max])
        
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ocp.constraints.x0 = np.zeros(n_x) # Initial state is 0

        # --- Solver / Integrator Options ---
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr = y
        ocp.model.cost_y_expr_e = y_e

        ocp.dims.N = self.STEPS # [10, 20]
        ocp.solver_options.tf = self.STEPS * self.dt # (s) [0.05, 0.10]
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        # ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # ocp.solver_options.sim_method_num_stages = 4
        # ocp.solver_options.sim_method_num_steps  = 3
        # ocp.solver_options.qp_solver_cond_N = 0

        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4

        return ocp
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
from quadrotor_model import quadrotor_model_auto
import casadi as ca
import matplotlib.pyplot as plt

# Quick Generic Helpers for some logging stuff
def yaw_from_quat_np(qw, qx, qy, qz):
    return math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class CrazyflieController:
    def __init__(self, urdf_path="./cf2x.urdf"):
        # Physical parameters from URDF
        self.mass = 0.027 # 0.027  # kg
        self.arm_length = 0.0397  # m
        self.kf = 3.16e-10  # thrust coefficient
        self.km = 7.94e-12  # moment coefficient
        self.max_rpm = 21000  # typical max RPM for CF2.0
        omega_max = self.max_rpm * 2*np.pi / 60.0  # rad/s
        self.max_thrust_per_motor = self.kf * (omega_max ** 2)
        
        # Initialize PyBullet
        self.setup_pybullet(urdf_path)
    
    def quaternion2rotation_matrix(self, quaternion):
        """
        quaternion: [x,y,z,w] (shape used in Bullet Physics) to rotation matrix 
        """
        n = np.dot(quaternion,quaternion)
        if n<1e-12:
            return np.identity(3)
        q =np.array([quaternion[3],quaternion[0],quaternion[1],quaternion[2]])
        q *= math.sqrt(2.0/n)
        q = np.outer(q,q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

    def setup_pybullet(self, urdf_path):
        """Initialize PyBullet simulation"""
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        dt = 1/10.0
        p.setTimeStep(dt)
        p.setPhysicsEngineParameter(numSubSteps=10)
        # Load the Crazyflie
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        plane = p.loadURDF("plane.urdf")
        self.cf_id = p.loadURDF(urdf_path, start_pos, start_orientation, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # Get motor link indices (prop links)
        self.motor_links = []
        for i in range(4):
            link_name = f"prop{i}_link"
            for j in range(p.getNumJoints(self.cf_id)):
                link_info = p.getJointInfo(self.cf_id, j)
                if link_info[12].decode('utf-8') == link_name:
                    self.motor_links.append(j)
                    break
        
        # p.changeDynamics(bodyUniqueId=self.cf_id, linkIndex=-1, lateralFriction=0.0)
        print(f"PyBullet Dynamics: {p.getDynamicsInfo(self.cf_id, -1)}")
        print(f"Found motor links at indices: {self.motor_links}")
        
    def mpc_to_motor_forces(self, thrust, moment_x, moment_y, moment_z):
        """
        Convert MPC outputs (thrust, moments) to individual motor forces
        
        Args:
            thrust: Total thrust command [N]
            moment_x: Roll moment [N⋅m]
            moment_y: Pitch moment [N⋅m] 
            moment_z: Yaw moment [N⋅m]
            
        Returns:
            motor_forces: Array of 4 motor forces [N]
        """
        c = self.km / self.kf
        control_vector = np.array([thrust, moment_x, moment_y, moment_z])

        mat = np.array([[1, 1, 1, 1], 
                        [self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2)], 
                        [-self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2)],
                        [c, -c, -c, c]])

        # [thrust, roll_moment, pitch_moment, yaw_moment] = mat * [F0, F1, F2, F3]
        motor_forces = np.linalg.inv(mat) @ control_vector

        # print("Control vector: ", control_vector)
        
        return motor_forces

    def apply_motor_forces(self, motor_forces, cur_traj):
        """
        Apply thrust forces at the four rotors.
        """
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        #print("curr traj 6-10", cur_traj[6:10])
        quat = cur_traj[6:10]
        #print("cur_traj", cur_traj)
        des_quat = [quat[1], quat[2], quat[3], quat[0]]
        # print(des_quat)
        quat_norm_sq = des_quat[0]*des_quat[0] + des_quat[1]*des_quat[1] + des_quat[2]*des_quat[2] + des_quat[3]*des_quat[3]
    
        quat_norm = math.sqrt(quat_norm_sq)
        des_quat[0] = des_quat[0] / quat_norm
        des_quat[1] = des_quat[1] / quat_norm
        des_quat[2] = des_quat[2] / quat_norm
        des_quat[3] = des_quat[3] / quat_norm

        #print("After normalizing", des_quat)
        R_meas = self.quaternion2rotation_matrix(des_quat)
        #R_meas = self.quaternion2rotation_matrix(orn)

        orn = np.array(orn)
        quat_norm_sq_orn = orn[0]*orn[0] + orn[1]*orn[1] + orn[2]*orn[2] + orn[3]*orn[3]
    
        quat_norm_orn = math.sqrt(quat_norm_sq_orn)
        orn[0] = orn[0] / quat_norm_orn
        orn[1] = orn[1] / quat_norm_orn
        orn[2] = orn[2] / quat_norm_orn
        orn[3] = orn[3] / quat_norm_orn
        # R_meas = self.quaternion2rotation_matrix(orn)

        # print("Motor Forces in body frame: ", motor_forces)
        motor_forces = np.array([f_i * R_meas[:,2] for f_i in motor_forces])
        #print("Forces applied at each motor: ",  motor_forces)

        p.applyExternalForce(self.cf_id,-1,motor_forces[0],[self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)
        p.applyExternalForce(self.cf_id,-1,motor_forces[1],[self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)
        p.applyExternalForce(self.cf_id,-1,motor_forces[2],[-self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)
        p.applyExternalForce(self.cf_id,-1,motor_forces[3],[-self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)

    def get_state(self):
        """Get current state of the Crazyflie for MPC feedback"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        vel, ang_vel = p.getBaseVelocity(self.cf_id)
        euler = p.getEulerFromQuaternion(orn)
        
        state = {
            'position': np.array(pos),
            'velocity': np.array(vel),
            'orientation': np.array([orn[3], orn[0], orn[1], orn[2]]),  # [roll, pitch, yaw]            
            'angular_velocity': np.array(ang_vel)
        }
        return state
    
    def step_simulation(self, mpc_output, cur_traj):# dt=1/240):
        """
        One simulation step with MPC control
        
        Args:
            mpc_output: Dict with keys 'thrust', 'moment_x', 'moment_y', 'moment_z'
            dt: Time step
        """
        # Convert MPC output to motor forces
        motor_forces = self.mpc_to_motor_forces(
            mpc_output['thrust'],
            mpc_output['moment_x'], 
            mpc_output['moment_y'],
            mpc_output['moment_z']
        )
        
        # Apply forces
        self.apply_motor_forces(motor_forces, cur_traj)
        
        # Step simulation
        p.stepSimulation()
        # time.sleep(dt)
        
        return self.get_state()

# Your implementation of the MPC class start here!! 
class MPC:
    """ MPC for the drone"""
    def __init__(self, target_height):
        # TUNABLE PARAMS
        self.dt = 0.1 # (s)
        self.N = 10
        self.T = self.N * self.dt # Horizon [s]

        # Trajectory
        params = {
            't' : [0, 10],
            'q' : [[0, 0, 0], [1, 1, target_height]],
            'v' : [[0, 0, 0], [0, 0, 0]],
            'a' : [[0, 0, 0], [0, 0, 0]],
            'dt': self.dt
        }

        #build reference trajectory
        traj = self.make_trajectory('quintic', params)
        self.sim_steps = len(traj['t'])-1
        q_w = np.concat([np.ones((self.sim_steps+1,1)), np.zeros((self.sim_steps+1,6))],axis=1) # hold quat and angular still
        self.trajectory1 = np.concat([traj['q'], traj['v'], q_w], axis=1)

        # Logging for plots
        self.simX1 = np.zeros_like(self.trajectory1)

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
        # Acaodos
        self.ocp = self.create_ocp()
        self.ocp_solver = AcadosOcpSolver(self.ocp)

    def solve(self, step, current_state):
        self.ocp_solver.constraints_set(0, 'lbx', current_state)
        self.ocp_solver.constraints_set(0, 'ubx', current_state)
        self.ocp_solver.set(0, 'x', current_state)

        # Build refernces for the horizon steps
        traj = self.trajectory1
        traj_len = traj.shape[0] - 1
        #print("traj len", traj_len)
        for k in range(self.N):
            idx = min(step + k + 1, traj_len - 1)
            q = traj[idx, 0:3]
            v = traj[idx, 3:6]
            qw,qx,qy,qz = traj[idx, 6:10]
            w = traj[idx, 10:13]        # p,q,r  (often zeros in your build)   # desired quaternion
            yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

            yref = np.zeros(14)
            yref[0:3]  = q
            yref[3:6]  = v
            yref[6]    = yaw
            yref[7:10] = w 
            self.ocp_solver.cost_set(k, "yref", yref)


        # Terminal Costs
        idx_T = min(step + self.N, traj_len)      
        r_T   = traj[idx_T, 0:3]
        v_T   = traj[idx_T, 3:6]
        qw,qx,qy,qz = traj[idx_T, 6:10]
        yaw_T = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        yref_T = np.zeros(7)
        yref_T[0:3] = r_T
        yref_T[3:6] = v_T
        yref_T[6]   = yaw_T

        self.ocp_solver.cost_set(self.N, "yref", yref_T)

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

        # --- TUNABLE-PARAMS ---
        w_pos = 20 # was 10
        w_vel = 2 # was 2.0
        w_yaw = 4 # was 4.0
        w_omega = 0.5
        w_u=1e-3 # minimize chatter (some small number)

        nx = model.x.size()[0]
        #nu = model.u.size()[0]

        # --- Dynamics Model ---
        x = model.x
        u = model.u

        r = x[0:3]
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        yaw = np.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        # costs
        y = ca.vertcat(r, v, yaw, w, u)
        y_e = ca.vertcat(r, v, yaw)

        W = np.diag([w_pos, w_pos, w_pos,
                     w_vel, w_vel, w_vel,
                     w_yaw,
                     w_omega, w_omega, w_omega,
                     w_u, w_u, w_u, w_u])
        
        W_e = np.diag([w_pos, w_pos, w_pos,
                       w_vel, w_vel, w_vel,
                       w_yaw])
        
        ocp.cost.yref = np.zeros(W.shape[0])
        ocp.cost.yref_e = np.zeros(W_e.shape[0])

        # --- Constraints --- (with params from the model)
        mu = 0.73575                 # max thrust per rotor [N]
        L  = 0.0397                  # arm length [m]
        kf = 3.16e-10                # thrust coeff
        km = 7.94e-12                # moment coeff
        g  = km / kf                 

        u1_min, u1_max   = 0.0, 4.0*mu                 # total thrust
        tau_xy_max       = np.sqrt(2.0) * L * mu       # roll/pitch torque bound
        tau_z_max        = 2.0 * g * mu 

        ocp.constraints.lbu   = np.array([u1_min, -tau_xy_max, -tau_xy_max, -tau_z_max])
        ocp.constraints.ubu   = np.array([u1_max,  tau_xy_max,  tau_xy_max,  tau_z_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ocp.constraints.x0 = np.zeros(nx) # Default values overwrite with solve()

        # --- Solver / Integrator Options ---
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr = y
        ocp.model.cost_y_expr_e = y_e

        ocp.dims.N = self.N # [10, 20]
        ocp.solver_options.tf = self.N * self.dt # (s) [0.05, 0.10]
        ocp.solver_options.integrator_type = "ERK" # = "RK4"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps  = 3
        ocp.solver_options.qp_solver_cond_N = 0

        return ocp

    def make_trajectory(self, traj_type, params):
        """
        Multi-dimensional cubic/quintic trajectory generator.

        Parameters
        ----------
        traj_type : str
            'cubic' or 'quintic'
        params : dict
            't' : [t0, tf]
            'q' : [[q0_x, q0_y, q0_z], [qf_x, qf_y, qf_z]]   (initial and final positions)
            'v' : [[v0_x, v0_y, v0_z], [vf_x, vf_y, vf_z]]   (initial and final velocities)
            'a' : [[a0_x, a0_y, a0_z], [af_x, af_y, af_z]]   (only for quintic)
            'dt': time step

        Returns
        -------
        traj : dict
            't' : (N,) time array
            'q' : (N, D) positions
            'v' : (N, D) velocities
            'a' : (N, D) accelerations
        """
        t0, tf = params['t']
        q0, qf = np.array(params['q'][0]), np.array(params['q'][1])
        v0, vf = np.array(params['v'][0]), np.array(params['v'][1])
        dt = params['dt']
        D = len(q0)   # number of dimensions (1D, 2D, 3D, ...)

        if traj_type == 'quintic':
            a0, af = np.array(params['a'][0]), np.array(params['a'][1])

        times = np.arange(t0, tf+dt, dt)
        N = len(times)

        q_traj = np.zeros((N, D))
        v_traj = np.zeros((N, D))
        a_traj = np.zeros((N, D))

        # Solve trajectory for each dimension independently
        for d in range(D):
            if traj_type == 'cubic':
                A = np.array([
                    [1, t0, t0**2, t0**3],
                    [0, 1, 2*t0, 3*t0**2],
                    [1, tf, tf**2, tf**3],
                    [0, 1, 2*tf, 3*tf**2]
                ])
                b = np.array([q0[d], v0[d], qf[d], vf[d]])
                coeffs = np.linalg.solve(A, b)

                for i, t in enumerate(times):
                    q_traj[i, d] = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3
                    v_traj[i, d] = coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2
                    a_traj[i, d] = 2*coeffs[2] + 6*coeffs[3]*t

            elif traj_type == 'quintic':
                A = np.array([
                    [1, t0, t0**2, t0**3, t0**4, t0**5],
                    [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                    [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                    [1, tf, tf**2, tf**3, tf**4, tf**5],
                    [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                    [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
                ])
                b = np.array([q0[d], v0[d], a0[d], qf[d], vf[d], af[d]])
                coeffs = np.linalg.solve(A, b)

                for i, t in enumerate(times):
                    q_traj[i, d] = (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 +
                                    coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)
                    v_traj[i, d] = (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 +
                                    4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
                    a_traj[i, d] = (2*coeffs[2] + 6*coeffs[3]*t +
                                    12*coeffs[4]*t**2 + 20*coeffs[5]*t**3)

        return {'t': times, 'q': q_traj, 'v': v_traj, 'a': a_traj}

# Main simulation loop
if __name__ == "__main__":
    # Initialize controller and MPC
    cf_controller = CrazyflieController()
    mpc = MPC(target_height=1)#0.5)
    
    # Simulation loop
    for step in range(mpc.sim_steps):
        # Get current state
        current_state = cf_controller.get_state()

        # Logging
        r_meas = current_state['position']
        v_meas = current_state['velocity']
        q_meas = current_state['orientation']
        yaw_meas = yaw_from_quat_np(*q_meas)

        current_state = np.concatenate([current_state['position'].ravel(), current_state['velocity'].ravel(), current_state['orientation'].ravel(), current_state['angular_velocity'].ravel()])
        
        # Logging
        idx_ref = min(step+1, mpc.trajectory1.shape[0]-1)
        r_ref = mpc.trajectory1[idx_ref, 0:3]
        qw,qx,qy,qz = mpc.trajectory1[idx_ref, 6:10]
        yaw_ref = yaw_from_quat_np(qw,qx,qy,qz)

        mpc_output = mpc.solve(step, current_state)

        # Apply control and step simulation
        new_state = cf_controller.step_simulation(mpc_output, mpc.trajectory1[step+1])#[:][step+1])
        
        u_vec = np.array([mpc_output['thrust'], mpc_output['moment_x'], mpc_output['moment_y'], mpc_output['moment_z']])
        
        # Logging
        mpc.log["r_ref"][step, :]   = r_ref
        mpc.log["r_meas"][step, :]  = r_meas
        mpc.log["yaw_ref"][step]    = yaw_ref
        mpc.log["yaw_meas"][step]   = yaw_meas
        mpc.log["u"][step, :]       = u_vec

        mpc.log["e_r"][step, :]   = r_meas - r_ref
        mpc.log["e_yaw"][step]    = wrap_pi(yaw_meas - yaw_ref)

        # Print state every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Height = {new_state['position'][2]:.3f}m")
    
    current_state = cf_controller.get_state()
    current_state = np.concatenate([current_state['position'].ravel(), current_state['orientation'].ravel(), current_state['velocity'].ravel(), current_state['angular_velocity'].ravel()])

    mpc.simX1[step,:] = current_state

    # Logging
    last = mpc.sim_steps
    mpc.log["r_ref"][last,:]  = mpc.trajectory1[min(last, mpc.trajectory1.shape[0]-1), 0:3]
    qw,qx,qy,qz = mpc.trajectory1[min(last, mpc.trajectory1.shape[0]-1), 6:10]
    mpc.log["yaw_ref"][last]  = yaw_from_quat_np(qw,qx,qy,qz)

    s = cf_controller.get_state()
    mpc.log["r_meas"][last,:]  = s['position']
    mpc.log["yaw_meas"][last]  = yaw_from_quat_np(*s['orientation'])

    mpc.log["e_r"][last,:]   = mpc.log["r_meas"][last,:] - mpc.log["r_ref"][last,:]
    mpc.log["e_yaw"][last]   = wrap_pi(mpc.log["yaw_meas"][last] - mpc.log["yaw_ref"][last])

    
    # Plotting obstacles and drone trajectories on a 3D plot
    traj_plot1 = mpc.trajectory1[:mpc.sim_steps+1, :3]
    x_obs = 0.2
    y_obs = 0.2
    z_obs = 0.4
    r_obs = 0.1
    margin = 0.01
    r_safe = r_obs + margin

    # Plot all 3 x, y, and z on a 3D plot
    #plt.ion()  # interactive mode ON
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Label axes
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Quadrotor Trajectory')

    # Plot the reference trajectory
    ax.plot(traj_plot1[:, 0], traj_plot1[:, 1], traj_plot1[:, 2], '--', label='Reference Trajectory for 1', color='r')

    ## Uncomment below for animation of drone trajectories
    line1, = ax.plot([], [], [], 'b', label='Drone 1')

    # Animation loop (Couldn't get this to work)
    for t in range(len(mpc.simX1)):
        line1.set_data(mpc.simX1[:t+1, 0], mpc.simX1[:t+1, 1])
        line1.set_3d_properties(mpc.simX1[:t+1, 2])
        plt.draw()
        plt.pause(0.02)  # adjust speed

    ax.legend()
    ax.grid(True)

    # Lots of Plots
    T = mpc.log["t"]
    def _trim(a):
        n = len(T)
        return a[:n] if a.shape[0] >= n else np.pad(a, ((0, n-a.shape[0]),) + ((0,0),)*(a.ndim-1))

    r_ref   = _trim(mpc.log["r_ref"])
    r_meas  = _trim(mpc.log["r_meas"])
    yaw_ref = mpc.log["yaw_ref"][:len(T)]
    yaw_meas= mpc.log["yaw_meas"][:len(T)]
    u_log   = _trim(mpc.log["u"])
    e_r     = _trim(mpc.log["e_r"])
    e_yaw   = mpc.log["e_yaw"][:len(T)]

    # Trajectory Position tracking
    fig1, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes_labels = ['X [m]', 'Y [m]', 'Z [m]']
    for i, ax in enumerate(axs):
        ax.plot(T, r_ref[:, i], '--', label=f'{axes_labels[i]} ref')
        ax.plot(T, r_meas[:, i], label=f'{axes_labels[i]} meas')
        ax.set_ylabel(axes_labels[i])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best')
    axs[-1].set_xlabel('Time [s]')
    fig1.suptitle('Position Tracking (r)')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Yaw tracking
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 3.2))
    ax2.plot(T, yaw_ref, '--', label='Yaw ref')
    ax2.plot(T, yaw_meas, label='Yaw meas')
    ax2.set_title('Yaw Tracking (ψ)')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Yaw [rad]')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='best')
    fig2.tight_layout()

    # Position & yaw errors 
    fig3, axs3 = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    axs3[0].plot(T, e_r[:,0], label='e_x')
    axs3[0].plot(T, e_r[:,1], label='e_y')
    axs3[0].plot(T, e_r[:,2], label='e_z')
    axs3[0].set_title('Position Error')
    axs3[0].set_ylabel('Error [m]')
    axs3[0].grid(True, linestyle='--', alpha=0.5)
    axs3[0].legend(loc='best')

    axs3[1].plot(T, e_yaw, label='e_yaw')
    axs3[1].set_title('Yaw Error')
    axs3[1].set_xlabel('Time [s]')
    axs3[1].set_ylabel('Error [rad]')
    axs3[1].grid(True, linestyle='--', alpha=0.5)
    axs3[1].legend(loc='best')
     
    fig3.tight_layout()
 
    # Control inputs
    fig4, ax4 = plt.subplots(1, 1, figsize=(9, 4.2))
    ax4.plot(T, u_log[:,0], label='u1 (thrust) [N]')
    ax4.plot(T, u_log[:,1], label='τx [N·m]')
    ax4.plot(T, u_log[:,2], label='τy [N·m]')
    ax4.plot(T, u_log[:,3], label='τz [N·m]')
    ax4.set_title('Control Inputs u')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='best')
    fig4.tight_layout()

    plt.show()
    p.disconnect()

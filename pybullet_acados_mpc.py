import pybullet as p
import pybullet_data
import numpy as np
import time
import math

import matplotlib.pyplot as plt

from mpc import MPC
from constants import constants

# Quick Generic Helpers for some logging stuff
def yaw_from_quat_np(qw, qx, qy, qz):
    return math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class CrazyflieController:
    def __init__(self, urdf_path="./cf2x.urdf"):
        self.const = constants()
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
        c = self.const.robot['km'] / self.const.robot['kf']
        control_vector = np.array([thrust, moment_x, moment_y, moment_z])

        # mixer matrix
        l = self.const.robot['arm']
        mat = np.array([[1, 1, 1, 1], 
                        [l/math.sqrt(2), -l/math.sqrt(2), l/math.sqrt(2), -l/math.sqrt(2)], 
                        [-l/math.sqrt(2), -l/math.sqrt(2), l/math.sqrt(2), l/math.sqrt(2)],
                        [c, -c, -c, c]])

        # [thrust, roll_moment, pitch_moment, yaw_moment] = mat * [F0, F1, F2, F3]
        motor_forces = np.linalg.inv(mat) @ control_vector
        
        return motor_forces

    def apply_motor_forces(self, motor_forces, moment_z):
        """
        Apply thrust forces at the four rotors.
        """
        # pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        # #print("curr traj 6-10", cur_traj[6:10])
        l = self.const.robot['arm']
        pos_xy = l / math.sqrt(2)
        positions = [
            [ pos_xy, pos_xy, 0.0],
            [ pos_xy, -pos_xy, 0.0],
            [-pos_xy, pos_xy, 0.0],
            [-pos_xy, -pos_xy, 0.0],
        ]

        for f_i, pos in zip(motor_forces, positions):
            p.applyExternalForce(
                self.cf_id,
                -1,
                [0.0, 0.0, float(f_i)],
                pos,
                p.LINK_FRAME,
            )

        p.applyExternalTorque(
            self.cf_id,
            -1,
            [0.0, 0.0, float(moment_z)],
            p.LINK_FRAME,
        )


    def get_state(self):
        """Get current state of the Crazyflie for MPC feedback"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        vel, ang_vel = p.getBaseVelocity(self.cf_id)

        R = self.quaternion2rotation_matrix(orn)
        ang_vel_body = R.T @ np.array(ang_vel)

        state = {
            'position': np.array(pos),
            'velocity': np.array(vel),
            'orientation': np.array([orn[3], orn[0], orn[1], orn[2]]),       
            'angular_velocity': ang_vel_body,
        }
        return state
    
    def step_simulation(self, mpc_output):# dt=1/240):
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
        self.apply_motor_forces(motor_forces, mpc_output['moment_z'])
        p.stepSimulation()
        return self.get_state()


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
        mpc.simX1[step, :] = current_state
        
        # Logging
        idx_ref = min(step+1, mpc.trajectory.shape[0]-1)
        r_ref = mpc.trajectory[idx_ref, 0:3]
        qw,qx,qy,qz = mpc.trajectory[idx_ref, 6:10]
        yaw_ref = yaw_from_quat_np(qw,qx,qy,qz)

        mpc_output = mpc.solve(step, current_state)

        # Apply control and step simulation
        new_state = cf_controller.step_simulation(mpc_output)
        
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
    
    s = cf_controller.get_state()
    current_state = np.concatenate([
        s['position'].ravel(),
        s['velocity'].ravel(),
        s['orientation'].ravel(),
        s['angular_velocity'].ravel(),
    ])
    last = mpc.sim_steps
    mpc.simX1[last, :] = current_state

    # Logging
    idx_last = min(last, mpc.trajectory.shape[0] - 1)
    mpc.log["r_ref"][last, :] = mpc.trajectory[idx_last, 0:3]
    qw, qx, qy, qz = mpc.trajectory[idx_last, 6:10]
    mpc.log["yaw_ref"][last] = yaw_from_quat_np(qw, qx, qy, qz)

    mpc.log["r_meas"][last, :] = s['position']
    mpc.log["yaw_meas"][last] = yaw_from_quat_np(*s['orientation'])
    mpc.log["e_r"][last, :] = mpc.log["r_meas"][last, :] - mpc.log["r_ref"][last, :]
    mpc.log["e_yaw"][last] = wrap_pi(mpc.log["yaw_meas"][last] - mpc.log["yaw_ref"][last])
    
    # Plotting obstacles and drone trajectories on a 3D plot
    traj_plot1 = mpc.trajectory[:mpc.sim_steps+1, :3]
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

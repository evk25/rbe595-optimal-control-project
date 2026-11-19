import numpy as np

def linear_trajectory(
        degree: str = "quintic", 
        t: tuple[float, float] = (0.0, 10.0),
        dt: float = 0.1,
        p: tuple[list[float], list[float]] = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        v: tuple = ([0, 0, 0], [0, 0, 0]),
        a: tuple = ([0, 0, 0], [0, 0, 0])
    ) -> dict:
    """
    Given time, and initial and final conditions, build a linear trajectory.

    Parameters
    ----------
    degree : str
        'cubic' or 'quintic'
    t : tuple
        time bounds for the trajectory
    dt : float
        granularity of trajectory     
    p : tuple(float, float)
        Initial and final position
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
    t_0 = t[0] 
    t_f = t[1]
    p_0 = np.array(p[0])
    p_f = np.array(p[1])
    v_0 = np.array(v[0])
    v_f = np.array(v[1])
    dt = dt
    D = len(p_0)   # Size of state vector

    if degree == 'quintic':
        a_0, a_f = np.array(a[0]), np.array(a[1])

    # Create time range
    times = np.arange(t_0, t_f + dt, dt, float)
    N = len(times) # number of steps in trajectory

    p_traj = np.zeros((N, D))
    v_traj = np.zeros((N, D))
    a_traj = np.zeros((N, D))

    # Solve trajectory for each dimension independently
    for d in range(D):
        if degree == 'cubic':
            A = np.array([
                [1, t_0, t_0**2, t_0**3],
                [0, 1, 2*t_0, 3*t_0**2],
                [1, t_f, t_f**2, t_f**3],
                [0, 1, 2*t_f, 3*t_f**2]
            ])
            b = np.array([p_0[d], v_0[d], p_f[d], v_f[d]])
            coeffs = np.linalg.solve(A, b)

            for i, time in enumerate(times):
                p_traj[step, d] = coeffs[0] + coeffs[1]*time + coeffs[2]*time**2 + coeffs[3]*time**3
                v_traj[step, d] = coeffs[1] + 2*coeffs[2]*time + 3*coeffs[3]*time**2
                a_traj[step, d] = 2*coeffs[2] + 6*coeffs[3]*time

        elif degree == 'quintic':
            A = np.array([
                [1, t_0, t_0**2, t_0**3, t_0**4, t_0**5],
                [0, 1, 2*t_0, 3*t_0**2, 4*t_0**3, 5*t_0**4],
                [0, 0, 2, 6*t_0, 12*t_0**2, 20*t_0**3],
                [1, t_f, t_f**2, t_f**3, t_f**4, t_f**5],
                [0, 1, 2*t_f, 3*t_f**2, 4*t_f**3, 5*t_f**4],
                [0, 0, 2, 6*t_f, 12*t_f**2, 20*t_f**3]
            ])
            b = np.array([p_0[d], v_0[d], a_0[d], p_f[d], v_f[d], a_f[d]])
            coeffs = np.linalg.solve(A, b)

            for step, time in enumerate(times):
                p_traj[step, d] = (
                    coeffs[0] + coeffs[1]*time + coeffs[2]*time**2 + coeffs[3]*time**3 + coeffs[4]*time**4 + coeffs[5]*time**5)
                v_traj[step, d] = (
                    coeffs[1] + 2*coeffs[2]*time + 3*coeffs[3]*time**2 + 4*coeffs[4]*time**3 + 5*coeffs[5]*time**4)
                a_traj[step, d] = (
                    2*coeffs[2] + 6*coeffs[3]*time + 12*coeffs[4]*time**2 + 20*coeffs[5]*time**3)
                
    return {'t': times, 'p': p_traj, 'v': v_traj, 'a': a_traj}

def circular_trajectory(
        t: tuple[float, float] = (0.0, 10.0),
        dt: float = 0.1,
        center: tuple[float, float, float] = (0.0, 0.0, 1.0),
        radius: float = 1.0,
        omega: float = 2.0 * np.pi / 10.0,  # rad/s (1 full loop in 10 s)
    ) -> dict:

    t_0, t_f = t
    times = np.arange(t_0, t_f + dt, dt, float)
    N = len(times)

    cx, cy, cz = center
    theta = omega * (times - t_0)
    p_traj = np.zeros((N, 3))
    v_traj = np.zeros((N, 3))
    a_traj = np.zeros((N, 3))

    p_traj[:, 0] = cx + radius * np.cos(theta)
    p_traj[:, 1] = cy + radius * np.sin(theta)
    p_traj[:, 2] = cz

    v_traj[:, 0] = -radius * omega * np.sin(theta)
    v_traj[:, 1] =  radius * omega * np.cos(theta)
    
    a_traj[:, 0] = -radius * (omega ** 2) * np.cos(theta)
    a_traj[:, 1] = -radius * (omega ** 2) * np.sin(theta)

    return {'t': times, 'p': p_traj, 'v': v_traj, 'a': a_traj}



if __name__ == "__main__":
    print("Test Linear Trajectory from [0, 0, 0] to [1, 1, 1]:")
    traj = linear_trajectory()
    print("t bounds" + str(traj['t']))
    print("p range" + str(traj['p']))
    print("v range" + str(traj['v']))
    print("a range" + str(traj['a']))


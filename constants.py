from numpy import pi

import xml.etree.ElementTree as ET

class constants():
    def __init__(self) -> None:
        self.robot = {"arm": 0.0,
                           "kf": 0.0,
                           "km": 0.0,
                           "thrust2weight": 0.0,
                           "max_speed_kmh": 0.0,
                           "gnd_eff_coeff": 0.0,
                           "prop_radius": 0.0,
                           "drag_coeff_xy": 0.0,
                           "drag_coeff_z": 0.0,
                           "dw_coeff_1": 0.0,
                           "dw_coeff_2": 0.0,
                           "dw_coeff_3": 0.0,
                           "pwm2rpm_scale": 0.0,
                           "pwm2rpm_const": 0.0,
                           "pwm_min": 0.0,
                           "pwm_max": 0.0}

        tree = ET.parse("cf2x.urdf")
        root = tree.getroot()

        properties = root.find(".//properties")
        if properties is not None:
            for key in self.robot:
                entry = properties.attrib.get(key)
                if entry is not None:
                    self.robot[key] =float(entry)

        self.mass = 0.027 # kg
        self.max_rpm = 21000  # typical max RPM for CF2.0
        self.omega_max = self.max_rpm * 2*pi / 60.0  # rad/s
        self.mu = self.robot["kf"] * (self.max_rpm ** 2) # max thrust
        self.g = 9.81 # gravity
        self.I_11 = 1.4e-5 # Inertia about b1
        self.I_22 = 1.4e-5 # Inertia about b2
        self.I_33 = 2.17e-5 # Inertia about b3

if __name__ == "__main__":
    const = constants()
    print(const.robot)
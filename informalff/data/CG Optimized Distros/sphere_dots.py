import json
import numpy as np
from scipy import optimize as opt
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize(v):
    v *= 1 / np.linalg.norm(v)
    return v

def random_dots(n):
    t = np.random.random(n)
    p = np.random.random(n)
    dots = []
    for i in range(n):
        temp = np.array([t[i], p[i]])
        dots.append(temp)
    return dots

def get_angle(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(numerator/denominator)

def correct_angles(v):
    theta = 2 * np.pi * v[0]
    phi = (v[1] - 0.5) * np.pi

    while theta < 0:
        theta += 2 * np.pi

    while theta > 2 * np.pi:
        theta -= 2 * np.pi
    
    while phi < -1 * np.pi / 2:
        phi += np.pi / 2
    
    while phi > np.pi / 2:
        phi -= np.pi / 2
    
    return [theta, phi]

def get_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])

def compute_sphere_energy(dots):
    n = int(len(dots) / 2)
    vectors = dots.reshape((n, 2))
    angles = [correct_angles(v) for v in vectors]
    coords = [get_cartesian(*a) for a in angles]
    energies = []
    for i in range(n):
        for j in range(i, n):
            if i != j:
                r = get_angle(coords[i], coords[j])
                if r != 0:
                    energies.append( 1/r**2 )
    e_avg = sum(energies) / len(energies)
    e_stddev = [ (e - e_avg)**2 for e in energies ]
    return e_avg + np.sqrt(sum(e_stddev) / len(energies))

dots_per_shell = list(range(3, 71))

for puntos in dots_per_shell:

    print(f" {puntos}", end="", flush=True)

    d = random_dots(puntos)
    dots = np.array(d).flatten()

    res = opt.minimize(compute_sphere_energy, x0=dots, method='CG',
                       options={"disp" : False, "maxiter" : 10000,
                                "return_all" : False, "gtol" : 1e-8})

    solution = res.x
    solution.resize((puntos,2))

    rtp_sol = [correct_angles(v) for v in solution]
    xyz_sol = [get_cartesian(*a) for a in rtp_sol]
    xyz = list(zip(*xyz_sol))

    to_plot = {q : xyz[i] for i, q in enumerate("XYZ")}

    print(",", end="", flush=True)

    with open(f"distros_{puntos}.json", "w") as f:
        json.dump(to_plot, f)
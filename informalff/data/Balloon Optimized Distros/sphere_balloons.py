import json
import numpy as np
from scipy import optimize as opt
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool
from datetime import datetime
startTime = datetime.now()

def normalize(v):
    v *= 1 / np.linalg.norm(v)
    return v

def random_dots(n):
    dots = [np.random.rand(3) for i in range(n)]
    return dots

def opt_vector(vi, coords, n, threshold):
    motion_vector = np.array([0,0,0], dtype=np.float64)
    for j in range(n):
        if vi != j:
            v = coords[vi] - coords[j]
            vn = np.linalg.norm(v)
            proj = coords[j] + (np.dot(coords[vi], v)) * coords[vi] - coords[vi]
            length = np.linalg.norm(proj)

            if length > 1e-10:
                factor = 1 / (np.exp((vn / threshold)**2 / 2) - 1)
                motion_vector -= proj * 1/length * factor

    motion_vector *= 1.0/(n-1)
    return motion_vector

def get_opt_vectors(coords):
    n = len(coords)
    threshold = 2 / np.sqrt(n)
    preopt = [[i, coords, n, threshold] for i in range(n)]
    with Pool() as p:
        results = p.starmap(opt_vector, preopt)
    return results

def optimize(init_coords, threshold=1e-6, maxiter=100000):

    coords = init_coords
    last_criterion = 0
    for iter in range(maxiter):

        motion = get_opt_vectors(coords)

        coords = [normalize(coords[i] + motion[i]) for i in range(len(coords))]

        norms = [np.linalg.norm(m) for m in motion]
        avg = sum(norms) / len(motion)
        criterion = np.sqrt(sum([(m - avg)**2 for m in norms]) / len(motion))
        
        if np.abs(last_criterion - criterion) < threshold:
            print(f"Convergence achieved in {iter + 1} iterations!")
            # print(motion)
            # print(criterion)
            return coords
        else:
            last_criterion = criterion

    raise RuntimeError("The number of iterations was exceeded "
                       "and no minimum was found.")

for puntos in range(3, 71):
    print("-" * 60)
    print(f"Optimizing sphere with {puntos} dots ...")
    np.random.seed(42)
    initial_coords = random_dots(puntos)
    # initial_coords = [np.array([1.0,0,0]),
    #                   np.array([0,1.0,0]),
    #                   np.array([0,0,1.0]),
    #                   np.array([-1.0,0,0]),
    #                   np.array([0,-1.0,0]),
    #                   np.array([0,0,-1.0])]

    coords = [normalize(v) for v in initial_coords]

    print(f"Before optimizing: {datetime.now() - startTime}", flush=True)
    results = optimize(coords, 1e-10)
    print(f"After optimizing: {datetime.now() - startTime}", flush=True)
    print("-" * 60)

    # for r in results:
    #     for s in results:
    #         print(np.linalg.norm(r - s), end=", ")
    #     print("")

    final = {q : [] for q in "XYZ"}
    for r in results:
        final["X"].append(r[0])
        final["Y"].append(r[1])
        final["Z"].append(r[2])

    with open(f"distros_{puntos}.json", "w") as f:
        json.dump(final, f)
import numpy as np
import matplotlib.pyplot as plt

w = 400
h = 300

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect_triangle(O, D, V0, V1, V2):
    # Calcula la normal del triangulo
    E1 = V1 - V0
    E2 = V2 - V0
    N = np.cross(E1, E2)
    
    # Calcula la interseccion del rayo con el plano que contiene al triangulo
    t = intersect_plane(O, D, V0, N)
    
    # Verifica si la interseccion se encuentra dentro del triangulo
    if t != np.inf:
        P = O + t * D
        edge1 = V1 - V0
        edge2 = V2 - V1
        edge3 = V0 - V2
        
        # Comprueba si el punto de interseccion esta dentro del triangulo
        cp1 = np.cross(edge1, P - V0)
        cp2 = np.cross(edge2, P - V1)
        cp3 = np.cross(edge3, P - V2)
        
        if np.dot(cp1, N) >= 0 and np.dot(cp2, N) >= 0 and np.dot(cp3, N) >= 0:
            return t
    
    return np.inf

def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['v0'], obj['v1'], obj['v2'])

def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        E1 = obj['v1'] - obj['v0']
        E2 = obj['v2'] - obj['v0']
        N = normalize(np.cross(E1, E2))
    return N
    
def get_color(obj, M):
    if obj['type'] == 'triangle':
        return obj['color']
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toO = normalize(O - M)

    col_ray = ambient
    for light in lights:
        toL = normalize(light['position'] - M)
        l = [intersect(M + N * 0.0001, toL, obj_sh) for k, obj_sh in enumerate(scene) if k != obj_idx]
        if l and min(l) < np.inf:
            return
        col_ray += obj.get('diffuse_c') * max(np.dot(N, toL), 0) * color
        col_ray += obj.get('specular_c') * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * light['color']
    return obj, M, N, col_ray

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius), color=np.array(color), reflection=.5, diffuse_c=.75, specular_c=.5)
    
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (color_plane0 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
        diffuse_c=.75, specular_c=.5, reflection=.25)
    
def add_triangle(v0, v1, v2, color):
    return dict(type='triangle', v0=np.array(v0), v1=np.array(v1), v2=np.array(v2),
                color=np.array(color), reflection=0.25, diffuse_c=.75, specular_c=.5)

# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
         add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
         add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
         add_plane([0., -.5, 0.], [0., 1., 0.],),
         add_triangle([0, 0, 0], [.5, -.5, 1.5], [0., .5, 1.5], [1.0, 0.0, 0.0])
    ]

# Light position and color.
lights = [
    {'position': np.array([-2.0, 2.0, 0.0]), 'color': np.array([0.5, 0.5, 0.5])},
    {'position': np.array([1.0, 4.0, 1.0]), 'color': np.array([0.8, 0.8, 0.8])}
]

# Default light and material parameters.
ambient = .05
specular_k = 50

depth_max = 5  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([-0.5, 4, -2.])  # Camera.
Q = np.array([-1., 0., 0.])  # Camera pointing to. 
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-2., -2. / r + .25, 4., 4. / r + .25)

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # Loop through initial and secondary rays.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # Reflection: create a new ray.
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

plt.imsave('outputs/pt3.png', img)
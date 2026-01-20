from Functions import *
import taichi as ti
ti.init(arch=ti.gpu)

@ti.func
def CubeSDF(a: ti.types.vector(3, float), P: ti.types.vector(3, float), S:float):
    x = ti.abs(a[0] - P[0]) - S
    y = ti.abs(a[1] - P[1]) - S
    z = ti.abs(a[2] - P[2]) - S
    return ti.max(x, y, z)

@ti.func
def SphereSDF(a: ti.types.vector(3, float), P: ti.types.vector(3, float), S:float):
    return (a-P).norm() -S

@ti.func
def PlaneSDF(a: ti.types.vector(3, float), Ax: ti.types.vector(3, float), O:float,):
    return a.dot(Ax) + O

@ti.func
def Wrap(a: ti.types.vector(3, float), S: float):
    x = ti.math.mod(a[0] - S/2, S) - S/2
    y = ti.math.mod(a[1] - S/2, S) - S/2
    z = ti.math.mod(a[2] - S/2, S) - S/2
    return ti.Vector([x, y, z])

@ti.func
def CrossSDF(a: ti.types.vector(3, float), P: ti.types.vector(3, float), S:float):
    x = ti.max(ti.abs(a[0] - P[0]),ti.abs(a[1] - P[0]))
    y = ti.max(ti.abs(a[1] - P[1]),ti.abs(a[2] - P[1]))
    z = ti.max(ti.abs(a[2] - P[2]),ti.abs(a[0] - P[2]))
    return ti.min(x,y,z) - S

@ti.func
def M_S(o: ti.types.vector(3, float), r: int):
    scale = 2 * (1/3) ** r
    s = (1/3) ** r * (1/3)
    cube = CubeSDF(o, ti.Vector([0.0, 0.0, 0.0]), 1.0)
    cross = -CrossSDF(Wrap(o, scale), ti.Vector([0.0, 0.0, 0.0]), s)
    return ti.max(cube, cross)

@ti.func
def MengerSpongeSDF(o: ti.types.vector(3, float), iterations: int = 5):
    d = M_S(o, 0)
    for i in range(1, iterations):
        d = ti.max(d, M_S(o, i))
    return d

@ti.func
def CubeSDF4D(a: ti.types.vector(3, float), P: ti.types.vector(3, float), S:float, W):
    x = ti.abs(a[0] - P[0]) - S
    y = ti.abs(a[1] - P[1]) - S
    z = ti.abs(a[2] - P[2]) - S
    W = ti.abs(W) - S
    return ti.max(x, y, z, W)

@ti.func
def Gasket(p0: ti.types.vector(3, float)):
    p = ti.Vector([p0.x, p0.y, p0.z, 1.0])
    for n in range(15):

        p_xyz = ti.Vector([
            ti.math.mod(p.x - 1.0, 2.0) - 1.0,
            ti.math.mod(p.y - 1.0, 2.0) - 1.0,
            ti.math.mod(p.z - 1.0, 2.0) - 1.0
        ])
        p = ti.Vector([p_xyz.x, p_xyz.y, p_xyz.z, p.w])

        scale = 1.2 / max(p_xyz.dot(p_xyz), 1e-6)
        p *= scale

    F1 = (ti.Vector([p.x, p.z]) / p.w).norm() * 0.25
    F2 = (abs(p.y) * 0.35) / p.w
    return  F2



import math
import taichi as ti
ti.init(arch=ti.gpu)

@ti.func
def Skybox(dr: ti.types.vector(3, float)):
    li = ti.Vector([1.0, 0.3, 0.0])
    li = li.normalized()
    ligtht = ti.max( dr.dot(li) ,0 ) * 2
    return ligtht * ti.Vector([1.0, 1.0, 1.0])



@ti.func
def vrotate(v: ti.types.vector(3, float),camera_yp: ti.types.vector(2, float)):
    yaw = camera_yp[0]
    pitch = camera_yp[1]

    v = ti.Vector([v[0],
                   v[2] * ti.sin(pitch) + v[1] * ti.cos(pitch),
                   v[2] * ti.cos(pitch) - v[1] * ti.sin(pitch)])

    v = ti.Vector([
                   v[0] * ti.cos(yaw) + v[2] * ti.sin(yaw),
                   v[1],
                   -v[0] * ti.sin(yaw) + v[2] * ti.cos(yaw)])
    return v



def vrotate_p(v, yp):
    yaw, pitch = yp

    # pitch
    v = ti.Vector([
        v[0],
        v[2] * math.sin(pitch) + v[1] * math.cos(pitch),
        v[2] * math.cos(pitch) - v[1] * math.sin(pitch)
    ])

    # yaw
    v = ti.Vector([
        v[0] * math.cos(yaw) + v[2] * math.sin(yaw),
        v[1],
        -v[0] * math.sin(yaw) + v[2] * math.cos(yaw)
    ])
    return v

@ti.func
def v_smoothstep(a: ti.types.vector(3, float), b: ti.types.vector(3, float), t):
    return a + (b - a) * t

@ti.func
def random_vector(n):
    u = ti.random()
    v = ti.random()

    phi = 2.0 * ti.math.pi * u
    z = v
    r = ti.sqrt(1.0 - z * z)

    vec = ti.Vector([
        r * ti.cos(phi),
        r * ti.sin(phi),
        z
    ])

    if n.dot(vec) < 0:
        vec = -vec

    return vec

@ti.func
def randompoint(power):
    r = ti.random() * 2 * ti.math.pi
    v = ti.Vector([ti.cos(r),
                   ti.sin(r),
                   0])
    return v * ti.sqrt( ti.random() ) * power







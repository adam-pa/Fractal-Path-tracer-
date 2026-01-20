import taichi as ti
ti.init(arch=ti.gpu)

@ti.func
def ObjectColor(n: int):
    # red, gree, blue
    props = ti.Vector([1.0, 1.0, 1.0])

    if n == 0:
        props = ti.Vector([0.5, 0.5, 1.0])
    if n == 1:
        props = ti.Vector([1.0, 1.0, 1.0])
    if n == 2:
        props = ti.Vector([1.0, 1.0, 1.0])
    if n == 3:
        props = ti.Vector([1.0, 1.0, 1.0])
    if n == 4:
        props = ti.Vector([1.0, 1.0, 1.0])
    if n == 5:
        props = ti.Vector([1.0, 1.0, 1.0])


    return props



@ti.func
def ObjecProperties(n: int):
    # roughness, specular, emission
    props = ti.Vector([1.0, 0.0, 0.0]) #it has to be in form 1.0 with 0

    if n == 0:
        props = ti.Vector([1.0, 0.05, 0.0])
    if n == 1:
        props = ti.Vector([1.0, 0.0, 1.0])
    if n == 2:
        props = ti.Vector([1.0, 0.05, 0.0])
    if n == 3:
        props = ti.Vector([1.0, 0.0, 0.0])
    if n == 4:
        props = ti.Vector([1.0, 0.0, 0.0])
    if n == 5:
        props = ti.Vector([1.0, 0.0, 0.0])
    if n == 8:
        props = ti.Vector([1.0, 0.0, 0.0])


    return props

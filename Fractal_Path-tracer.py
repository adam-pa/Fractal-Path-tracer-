import time
from Object_Functions import *
from Materials import *
from Functions import *
import taichi as ti

ti.init(arch=ti.gpu)
prev_time = time.time()

n = 1500
pixels = ti.Vector.field(3, dtype=float, shape=(n, n))

# Camera state
camera_pos = ti.Vector([0.0, 1.0, -1.0])
camera_speed = 1.5
camera_yp = ti.Vector([0.0, 0.0])
camera_a = 0.01
# Options
ni = 212  # ray iterations
state: int = 0 #preview/render


@ti.func
def argmin_static(arr):
    min_idx = 0
    min_val = arr[0]
    for i in ti.static(range(arr.n)):

        if arr[i] < min_val:
            min_val = arr[i]
            min_idx = i
    return min_idx


@ti.func
def object(a: ti.types.vector(3, float)):  # Object SDF
    min_val = 1e9
    #CubeDF = CubeSDF(a, ti.Vector([0, 1.0, 2]), 0.4)
    PlaneDF = PlaneSDF(a, ti.Vector([0, 1, 0]), -0.1)
    SphereDF = SphereSDF(a, ti.Vector([-1.5, 1.0, -0.5]), 0.5)
    SphereDF2 = SphereSDF(a, ti.Vector([0.0, 0.0, 0.0]), 5)
    MengerSpongeDF = MengerSpongeSDF(a - ti.Vector([0, 0.5, 5]))
    #Light = SphereSDF(a, ti.Vector([0, 2, 0]), 1)
    Gasket1 = Gasket(a)

    # objects
    #c1 =CubeSDF(a, ti.Vector([7, 0, 0]), 5)
    #c2 = CubeSDF(a, ti.Vector([-7, 0, 0]), 5)
    #c3 = CubeSDF(a, ti.Vector([0, 0, 8]), 5)
    #c4 = CubeSDF(a, ti.Vector([0, 9, 2]), 5)
    #b1 = CubeSDF(a*ti.Vector([1.0, 0.5, 1.0]), ti.Vector([-0.7, 0.4, -0.3]), 0.4)
    #b2 = CubeSDF(a * ti.Vector([1.0, 0.7, 1.0]), ti.Vector([0.3, 0.4, 1]), 0.4)
    fractal = ti.max(Gasket1, SphereDF2)


    objects = ti.Vector([PlaneDF, SphereDF, fractal])
    # objects



    for i in ti.static(range(objects.n)):
        min_val = ti.min(min_val, objects[i])

    index = argmin_static(objects)

    return ti.Vector([min_val, index])


@ti.func
def Ray(dr: ti.types.vector(3, float), rp: ti.types.vector(3, float)):  # Ray Calculation
    dir = dr
    g = rp
    for i in range(ni):
        O = object(g)[0]
        g = g + dir * O
        if O < 0.0001:
            break

    return g


@ti.func
def Normal(r: ti.types.vector(3, float)):  # Normal Calculation
    epsilon = 0.0001
    N = ti.Vector([
        object(r + ti.Vector([epsilon, 0, 0]))[0] - object(r - ti.Vector([epsilon, 0, 0]))[0],
        object(r + ti.Vector([0, epsilon, 0]))[0] - object(r - ti.Vector([0, epsilon, 0]))[0],
        object(r + ti.Vector([0, 0, epsilon]))[0] - object(r - ti.Vector([0, 0, epsilon]))[0]
    ])
    return N.normalized()


@ti.func
def RayTraced(x, y, rp, cam_yp: ti.types.vector(2, float),
                        cam_ad: ti.types.vector(2, float),  frame):

    dr = ti.Vector([x, y, 1]).normalized() + randompoint(0.0005)
    dr = vrotate(dr, cam_yp)
    fp = rp + dr * cam_ad[1]
    rp += vrotate( randompoint( cam_ad[0] ), cam_yp)
    dr = (fp - rp ).normalized()

    pixelcolor = ti.Vector([0.0, 0.0, 0.0])
    color = ti.Vector([1.0, 1.0, 1.0])

    emission_color = ti.Vector([1.0, 1.0, 1.0]) * 1

    bounces = 5

    for i in range(bounces):
        rp = Ray(dr, rp)
        n = Normal(rp)
        rp = rp + n * 0.0002

        if ObjecProperties(object(rp)[1])[2] > ti.random():
            pixelcolor += emission_color * 5
            break

        if object(rp)[0] > 100:
            pixelcolor += Skybox(dr)
            break

        Diffuse = random_vector(n)
        Specular = dr - 2 * n.dot(dr) * n

        dr = v_smoothstep(Specular, Diffuse, ObjecProperties(object(rp)[1])[0])

        if ObjecProperties(object(rp)[1])[1] > ti.random():
            dr = Specular

        pixelcolor *= 0.999 #power loss
        color *= ObjectColor(object(rp)[1])

    return pixelcolor * color

@ti.func
def Preview(x, y, rp, cam_yp: ti.types.vector(2, float) ):

    dr = vrotate(ti.Vector([x, y, 1]).normalized(), cam_yp )
    ray = Ray(dr, rp)
    n = Normal(ray)
    li = ti.Vector([1.0, 0.3, 0.0])
    li = li.normalized()
    h = (li - dr).normalized()

    colorp = ti.Vector([1.0, 0.5, 0.0])
    fogdistance = 100
    specstrengh = 4  # Specular Size
    specpower = 0.1  # Specular Power

    diffuse = ti.max(n.dot(li), 0.0)
    fog = -(ti.min((ray - rp).norm() / fogdistance, 1) - 0.5) + 0.5
    spec = (ti.max(n.dot(h), 0.0) ** specstrengh) * specpower * diffuse
    return (diffuse * colorp + spec) * fog



accum = ti.Vector.field(3, dtype=float, shape=(n, n))
frame_count = ti.field(dtype=int, shape=(n, n))

@ti.kernel
def reset_accum():
    for i, j in accum:
        accum[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def paint_frame(cam_pos: ti.types.vector(3, float),
                cam_yp: ti.types.vector(2, float),
                cam_a: float,
                frame: int, state: int):

    for i, j in pixels:
        dr = vrotate(ti.Vector([0.0, 0.0, 1.0]), cam_yp)
        auto_fc = ti.math.length( cam_pos - Ray(dr, cam_pos))

        x = (i / n) * 2 - 1
        y = (j / n) * 2 - 1


        if state > 0 :
            color = RayTraced(x, y, cam_pos, cam_yp, ti.Vector([cam_a, auto_fc]), frame)
            accum[i, j] = (accum[i, j] * frame + color) / (frame + 1)
            pixels[i, j] = accum[i, j]
        else:
            pixels[i, j] = Preview(x,y,cam_pos, cam_yp)



window = ti.ui.Window("Raymarching", (n, n), vsync=False)
canvas = window.get_canvas()
frame = 1



while window.running:

    if window.is_pressed(ti.ui.SHIFT):
        multiply = 2
    else:
        multiply = 1
    if window.is_pressed(ti.ui.CTRL):
        multiply = 0.25

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time
    m_speed = camera_speed * dt * multiply
    r_speed = camera_speed * dt * 0.75


    if window.is_pressed('w'):
        frame = 0
        camera_pos += vrotate_p(ti.Vector([0, 0, 1]), camera_yp) * m_speed
    if window.is_pressed('s'):
        frame = 0
        camera_pos -= vrotate_p(ti.Vector([0, 0, 1]), camera_yp) * m_speed
    if window.is_pressed('a'):
        frame = 0
        camera_pos -= vrotate_p(ti.Vector([1, 0, 0]), camera_yp) * m_speed
    if window.is_pressed('d'):
        frame = 0
        camera_pos += vrotate_p(ti.Vector([1, 0, 0]), camera_yp) * m_speed
    if window.is_pressed('q'):
        frame = 0
        camera_pos -= ti.Vector([0, 1, 0]) * m_speed
    if window.is_pressed('e'):
        frame = 0
        camera_pos += ti.Vector([0, 1, 0]) * m_speed

    if window.is_pressed(ti.ui.LEFT):
        frame = 0
        camera_yp -= ti.Vector([1, 0]) * r_speed * 0.75
    if window.is_pressed(ti.ui.RIGHT):
        frame = 0
        camera_yp += ti.Vector([1, 0]) * r_speed * 0.75
    if window.is_pressed(ti.ui.UP):
        frame = 0
        camera_yp += ti.Vector([0, 1]) * r_speed * 0.75
    if window.is_pressed(ti.ui.DOWN):
        frame = 0
        camera_yp -= ti.Vector([0, 1]) * r_speed * 0.75

    if window.is_pressed(ti.ui.LMB):
        if camera_a > 0:
            frame = 0
            camera_a -= m_speed * 0.05
    if window.is_pressed(ti.ui.RMB):
        if camera_a < 1:
            frame = 0
            camera_a += m_speed * 0.05
    if camera_a < 0:
        camera_a = 0


    if window.is_pressed('r'):  #whem r pressed start rendering
        state: int = 1
        frame = 0

    if window.is_pressed('v'):  #whem v pressed go to viewport
        state: int = 0
        frame = 0

    if state > 0 == frame:
        reset_accum()


    paint_frame(camera_pos, camera_yp, camera_a, frame, state)  # Window
    frame += 1
    canvas.set_image(pixels)

    if frame % 500 == 0 < state:
        filename = f'imwrite_export.png'
        ti.tools.imwrite(pixels.to_numpy(), filename)

    window.show()

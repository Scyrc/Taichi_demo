import time

import taichi as ti

ti.init(arch=ti.cuda)

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1.0 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
springY = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, float, shape=(1,))
ball_center[0] = [0, 0, 0]

pos = ti.Vector.field(3, float, shape=(n, n))
vel = ti.Vector.field(3, float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, float, shape=n * n)
colors = ti.Vector.field(3, float, shape=n * n)

bending_springs = False


@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in pos:
        pos[i, j] = [-0.5 + i * quad_size + random_offset[0],
                     0.6,
                     -0.5 + j * quad_size + random_offset[1]
                     ]

        vel[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mass_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + j + 1

        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + j + 1
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.344, 0.52)


initialize_mass_indices()

spring_offsets = []
if bending_springs:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))
else:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))


@ti.kernel
def substep():
    for i in ti.grouped(vel):
        vel[i] += gravity * dt

    for i in ti.grouped(pos):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                pos_ij = pos[i] - pos[j]
                v_ij = vel[i] - vel[j]
                d = pos_ij.normalized()
                curr_dis = pos_ij.norm()
                original_dist = quad_size * float(i - j).norm()
                # spring force
                force += -springY * d * (curr_dis / original_dist - 1)
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size

        vel[i] += force * dt

    for i in ti.grouped(pos):
        vel[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = pos[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            vel[i] -= min(vel[i].dot(normal), 0) * normal

        if i[1] != 0:
            pos[i] += dt * vel[i]


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = pos[i, j]


window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()
time_start = time.time()
while window.running:
    if current_t > 1.5:
        time_end = time.time()
        print(time_end - time_start)
        time_start = time.time()
        # Reset
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt

    update_vertices()
    camera.position(0.0, 0, 5)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices, indices=indices, per_vertex_color=colors, two_sided=True)
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()

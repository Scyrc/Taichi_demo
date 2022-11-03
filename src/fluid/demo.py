import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

# ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)


# ti.init(arch=ti.cuda)

def init_scene():
    floor = ti.Vector.field(3, float, shape=2 * 2)
    floor1 = ti.Vector.field(3, float, shape=2 * 2)
    floor2 = ti.Vector.field(3, float, shape=2 * 2)
    floor3 = ti.Vector.field(3, float, shape=2 * 2)
    floor4 = ti.Vector.field(3, float, shape=2 * 2)

    colors = ti.Vector.field(3, float, shape=2 * 2)

    size = 20

    height = 3
    floor[0] = [-1 * size, 0, -1 * size]
    floor[1] = [1 * size, 0, -1 * size]
    floor[2] = [-1 * size, 0, 1 * size]
    floor[3] = [1 * size, 0, 1 * size]

    floor1[0] = [-1 * size, height, 1 * size]
    floor1[1] = [-1 * size, height, -1 * size]
    floor1[2] = [-1 * size, 0, 1 * size]
    floor1[3] = [-1 * size, 0, -1 * size]

    floor2[0] = [-1 * size, height, -1 * size]
    floor2[1] = [1 * size, height, -1 * size]
    floor2[2] = [-1 * size, 0, -1 * size]
    floor2[3] = [1 * size, 0, -1 * size]

    floor3[0] = [1 * size, height, -1 * size]
    floor3[1] = [1 * size, height, 1 * size]
    floor3[2] = [1 * size, 0, -1 * size]
    floor3[3] = [1 * size, 0, 1 * size]

    floor4[0] = [1 * size, height, 1 * size]
    floor4[1] = [-1 * size, height, 1 * size]
    floor4[2] = [1 * size, 0, 1 * size]
    floor4[3] = [-1 * size, 0, 1 * size]

    for i in range(0, 4):
        colors[i] = (255 / 255.0, 250 / 255.0, 250 / 255.0)

    indices = ti.field(int, shape=2 * 3)
    indices[0] = 0
    indices[1] = 2
    indices[2] = 1

    indices[3] = 3
    indices[4] = 1
    indices[5] = 2

    return [floor, floor1, floor2, floor3, floor4], indices, colors


def draw_box_line(x_max, y_max, z_max):
    lines = ti.Vector.field(3, dtype=ti.f32, shape=24)
    lines[0] = ti.Vector([0, 0, 0])
    lines[1] = ti.Vector([x_max, 0, 0])

    lines[2] = ti.Vector([0, 0, 0])
    lines[3] = ti.Vector([0, y_max, 0])

    lines[4] = ti.Vector([0, 0, 0])
    lines[5] = ti.Vector([0, 0, z_max])

    lines[6] = ti.Vector([0, y_max, 0])
    lines[7] = ti.Vector([x_max, y_max, 0])

    lines[8] = ti.Vector([0, y_max, 0])
    lines[9] = ti.Vector([0, y_max, z_max])

    lines[10] = ti.Vector([x_max, 0, z_max])
    lines[11] = ti.Vector([x_max, y_max, z_max])

    lines[12] = ti.Vector([x_max, 0, z_max])
    lines[13] = ti.Vector([0, 0, z_max])

    lines[14] = ti.Vector([x_max, 0, z_max])
    lines[15] = ti.Vector([x_max, 0, 0])

    lines[16] = ti.Vector([x_max, y_max, z_max])
    lines[17] = ti.Vector([x_max, y_max, 0])

    lines[18] = ti.Vector([x_max, y_max, z_max])
    lines[19] = ti.Vector([0, y_max, z_max])

    lines[20] = ti.Vector([x_max, 0, 0])
    lines[21] = ti.Vector([x_max, y_max, 0])

    lines[22] = ti.Vector([0, 0, z_max])
    lines[23] = ti.Vector([0, y_max, z_max])

    return lines


if __name__ == "__main__":
    kk = 2
    x_max = 4 * kk
    y_max = 9 * kk
    z_max = 4 * kk
    ps = ParticleSystem((x_max, y_max, z_max))

    ps.add_cube(lower_corner=[1 * kk, 1 * kk, 1 * kk],
                cube_size=[2 * kk, 6 * kk, 2 * kk],
                velocity=[0.0, 0.0, 0.0],
                density=1000.0,
                color=0x956333,
                material=1)

    # ps.add_cube(lower_corner=[3, 1],
    #             cube_size=[2.0, 2.0, 2.0],
    #             velocity=[0.0, -20.0, 5.0],
    #             density=1000.0,
    #             color=0x956333,
    #             material=1)

    wcsph_solver = WCSPHSolver(ps)

    floors, indices, colors = init_scene()

    window = ti.ui.Window("Free Fall on GGui", (1024, 1024))
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 20, 40)
    # camera.lookat(0.0, 0, 0)

    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    camera_x = 0
    camera_y = 20
    camera_z = 40
    current_t = 0.0

    line_x = ti.Vector.field(3, dtype=ti.f32, shape=2)
    line_x[0] = ti.Vector([20, 0, 0])
    line_x[1] = -line_x[0]

    line_y = ti.Vector.field(3, dtype=ti.f32, shape=2)
    line_y[0] = ti.Vector([0, 20, 0])
    line_y[1] = -line_y[0]

    line_z = ti.Vector.field(3, dtype=ti.f32, shape=2)
    line_z[0] = ti.Vector([0, 0, 20])
    line_z[1] = -line_z[0]

    lines = draw_box_line(x_max, y_max, z_max)
    while window.running:
        # if window.get_event(ti.ui.PRESS):
        #     if window.event.key == 'r':
        #         ps.add_cube(lower_corner=[1 * kk, 1 * kk, 1 * kk],
        #                     cube_size=[2 * kk, 6 * kk, 2 * kk],
        #                     velocity=[0.0, 0.0, 0.0],
        #                     density=1000.0,
        #                     color=0x956333,
        #                     material=1)

        t = 0.5
        if window.is_pressed('a'):
            camera_x -= t
        if window.is_pressed('d'):
            camera_x += t

        if window.is_pressed('w'):
            camera_z -= t

        if window.is_pressed('s'):
            camera_z += t

        if window.is_pressed('q'):
            camera_y += t

        if window.is_pressed('e'):
            camera_y -= t

        # if window.is_pressed('r'):
        # particleSystem.initialize_mass_points()

        for i in range(25):
            wcsph_solver.step()
        particle_info = ps.dump()

        scene.point_light(pos=(30, 30, 30), color=(255 / 255.0, 198 / 255.0, 107 / 255.0))
        scene.ambient_light((0.5, 0.5, 0.5))
        camera.position(camera_x, camera_y, camera_z)
        scene.set_camera(camera)
        for floor in floors:
            scene.mesh(floor, indices=indices, per_vertex_color=colors, two_sided=True)

        # gui.circles(particle_info['position'] * ps.screen_to_world_ratio / 512,
        #             radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
        #             color=0x956333)

        scene.particles(ps.x, radius=ps.particle_radius, color=(0 / 255.0, 191 / 255.0, 255 / 255.0))
        # scene.particles(particleSystem.t_list[1], radius=ball_radius * 1, color=(144 / 255.0, 238 / 255.0, 144 / 255.0))

        scene.lines(line_x, width=3, color=(1, 0, 0))
        scene.lines(line_y, width=3, color=(0, 1, 0))
        scene.lines(line_z, width=3, color=(0, 0, 1))
        scene.lines(lines, width=3, color=(231/255.0, 101/255.0, 26/255.0))

        canvas.scene(scene)
        window.show()

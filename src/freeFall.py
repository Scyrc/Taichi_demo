import time

import taichi as ti

ti.init(arch=ti.cuda)


@ti.data_oriented
class ParticleSystem:
    def __init__(self, n: int, r: float):
        self.n = n
        self.pos = ti.Vector.field(3, float, shape=(n, n))
        self.vel = ti.Vector.field(3, float, shape=(n, n))
        # self.mass = ti.Vector.field(3, float, shape=(n, n))
        self.vertices = ti.Vector.field(3, float, shape=n * n)
        self.gravity = ti.Vector([0, -9.8, 0])
        self.dt = 4e-2 / self.n
        self.reflact_factor = 0.9
        self.drag_factor = -0.05

        self.radius = r
        self.scene_size = 20

        self.quad_size = self.scene_size * 2 / 10
        # self.zone = [[]]*100
        self.zone = ti.Matrix.field(n=n, m=3, dtype=float, shape=(n, n))
        self.zone_num = ti.field(int, shape=(n, n))

        self.initialize_mass_points()
        self.wind_on = ti.field(int, shape=())
        self.wind_on[None] = 0
        self.wind_force = ti.Vector([5.0, 0, 6.0])

    def change_wind(self):
        if self.wind_on[None] == 0:
            self.wind_on[None] = 1
        else:
            self.wind_on[None] = 0

        print(self.wind_on)


    @ti.kernel
    def initialize_mass_points(self):

        for i, j in self.pos:
            random_offset = ti.Vector([ti.random() - 0.5, ti.random(), ti.random() - 0.5]) * 1.5
            self.pos[i, j] = [0 + random_offset[0] * self.scene_size,
                              5 + random_offset[1] * self.scene_size,
                              0 + random_offset[2] * self.scene_size,
                              ]

            self.vel[i, j] = [0, 0, 0]

    @ti.kernel
    def update_vertices(self):
        for i, j in ti.ndrange(self.n, self.n):
            self.vertices[i * self.n + j] = self.pos[i, j]

    # @ti.func
    # def set_zone(self):
    #     for i, j in self.zone_num:
    #         self.zone_num[i, j] = 0
    #
    #     for i in ti.grouped(self.pos):
    #         x = self.pos[i][0]
    #         z = self.pos[i][2]
    #         x_index = ti.cast(ti.max((x + self.scene_size) % self.quad_size, 9), ti.int32)
    #         z_index = ti.cast(ti.max((x + self.scene_size) % self.quad_size, 9), ti.int32)
    #
    #         # self.zone[x_index * 10 + z_index].append(self.pos[i])
    #         count = self.zone_num[x_index, z_index]
    #         for j in ti.static(range(3)):
    #             self.zone[x_index, z_index][ti.static(count), j] = self.pos[i][j]
    #
    #         self.zone_num[x_index, z_index] += 1

    @ti.kernel
    def take_step(self):
        for i in ti.grouped(self.vel):
            self.vel[i] += self.gravity * self.dt
            if self.wind_on[None] == 1:
                self.vel[i] += self.wind_force * self.dt

            # self.vel[i] += self.vel[i] * self.drag_factor * self.dt

        # self.set_zone()
        # 检测粒子间的碰撞
        for i, j in ti.ndrange(self.n, self.n):
            for m, n in ti.ndrange(self.n, self.n):
                if i != m or j != n:
                    pos_diff = self.pos[i, j] - self.pos[m, n]
                    if pos_diff.norm() <= 2 * self.radius + 0.001:
                        normal = pos_diff.normalized()
                        self.vel[i, j] -= min(self.vel[i, j].dot(normal), 0) * normal
                        normal = -1 * normal
                        self.vel[m, n] -= min(self.vel[m, n].dot(normal), 0) * normal

        # 限制粒子小球在场景中
        for i in ti.grouped(self.pos):
            if self.pos[i][1] < 0.01 + self.radius and self.vel[i][1] < 0:
                self.vel[i][1] = -1 * self.reflact_factor * self.vel[i][1]

            if self.pos[i][0] > (self.scene_size / 1.0) * 0.98 and self.vel[i][0] > 0:
                self.vel[i][0] = -1 * self.reflact_factor * self.vel[i][0]

            if self.pos[i][0] < -(self.scene_size / 1.0) * 0.98 and self.vel[i][0] < 0:
                self.vel[i][0] = -1 * self.reflact_factor * self.vel[i][0]

            if self.pos[i][2] < -(self.scene_size / 1.0) * 0.98 and self.vel[i][2] < 0:
                self.vel[i][2] = -1 * self.reflact_factor * self.vel[i][2]

            if self.pos[i][2] > (self.scene_size / 1.0) * 0.98 and self.vel[i][2] > 0:
                self.vel[i][2] = -1 * self.reflact_factor * self.vel[i][2]

            self.pos[i] += self.dt * self.vel[i]

    def show_system(self):

        scene.particles(self.vertices, radius=ball_radius * 1, color=(144 / 255.0, 238 / 255.0, 144 / 255.0))
        # for i in range(self.n * self.n):
        #     t = ti.Vector.field(3, float, shape=1)
        #     t[0] = self.vertices[i]
        #     scene.particles(t, radius=ball_radius * 1, color=(144 / 255.0, 238 / 255.0, 144 / 255.0))


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




if __name__ == "__main__":

    ball_radius = 0.1
    ball_center = ti.Vector.field(3, float, shape=(2,))
    ball_center[0] = [0, 0.5, 0]
    ball_center[1] = [0, 0.8, 0]
    N = 33

    particleSystem = ParticleSystem(N, ball_radius)

    dt = particleSystem.dt
    substeps = int(1 / 60 // dt)

    window = ti.ui.Window("Free Fall on GGui", (1024, 1024))
    canvas = window.get_canvas()
    canvas.set_background_color((0, 0, 0))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    floors, indices, colors = init_scene()
    camera.position(0, 20, 40)
    # camera.lookat(0.0, 0, 0)

    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    camera_x = 0
    camera_y = 20
    camera_z = 40
    current_t = 0.0

    while window.running:

        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                particleSystem.initialize_mass_points()
                print("reset")

            if window.event.key == 'y':
                particleSystem.change_wind()
                print("change wind")

        if window.is_pressed('a'):
            camera_x -= 0.25
        if window.is_pressed('d'):
            camera_x += 0.25

        if window.is_pressed('w'):
            camera_z -= 0.25

        if window.is_pressed('s'):
            camera_z += 0.25

        if window.is_pressed('q'):
            camera_y += 0.25

        if window.is_pressed('e'):
            camera_y -= 0.25







        # if current_t > 200:
        #     # Reset
        #     particleSystem.initialize_mass_points()
        #     current_t = 0

        for i in range(substeps):
            particleSystem.take_step()
            # current_t += dt

        particleSystem.update_vertices()

        scene.point_light(pos=(0, 6, 6), color=(255 / 255.0, 198 / 255.0, 107 / 255.0))
        scene.ambient_light((0.5, 0.5, 0.5))
        camera.position(camera_x, camera_y, camera_z)
        scene.set_camera(camera)

        for floor in floors:
            scene.mesh(floor, indices=indices, per_vertex_color=colors, two_sided=True)

        # particleSystem.show_system()

        scene.particles(particleSystem.vertices, radius=ball_radius * 1, color=(144 / 255.0, 238 / 255.0, 144 / 255.0))

        canvas.scene(scene)
        window.show()

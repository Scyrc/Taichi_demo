## Taichi Demo
### 1.Taichi实现自由落体
- 用户输入: w,a,s,d,q,e用于调整相机位置,y用于控制风力,r重置系统到初始状态
- 演示
![image](https://github.com/ywsimon/Taichi_demo/blob/master/gif/freeFall.gif)
- 细节
  - 时间积分：显示欧拉
  - 空间积分：粒子系统
  - 受力：重力、风力
  - 碰撞：小球与平面、小球之间

### 2.Taichi实现布料模拟
- 用户输入: w,a,s,d,q,e用于调整相机位置,y用于控制风力,r重置系统到初始状态
- 演示
![image](https://github.com/ywsimon/Taichi_demo/blob/master/gif/cloth_ball_interaction.gif)
- 细节
  - 时间积分：显示欧拉
  - 空间积分：弹簧-质点
  - 受力：重力、风力、弹簧力
  - 碰撞：布料与球、布料自碰撞(未实现)

### 3.Taichi实现流体模拟
- 用户输入: w,a,s,d,q,e用于调整相机位置
- 演示
![image](https://github.com/ywsimon/Taichi_demo/blob/master/gif/fluid.gif)
- 细节
  - 时间积分：显示欧拉
  - 空间积分：wcsph
  - 碰撞：流体粒子与平面
### 参考
- https://docs.taichi-lang.cn/docs/cloth_simulation
- https://github.com/taichiCourse01/taichi_sph


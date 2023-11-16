# 初始粒子
initial_particles = np.linspace(-10, 10, 1000)

# 创建粒子滤波器实例
pf = ParticleFilter(initial_particles, state_update, measurement_update, N=100)

# 模拟一些测量数据
measurements = np.random.normal(0, 1, size=50)

# 使用粒子滤波器估计状态
estimates = []
for measurement in measurements:
    pf.predict()
    pf.update(measurement)
    pf.resample()
    estimates.append(pf.estimate())

# 绘制结果
plt.plot(measurements, label='Measurements')
plt.plot(estimates, label='Estimates')
plt.legend()
plt.show()


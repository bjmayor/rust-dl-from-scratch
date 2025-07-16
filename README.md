# rust-dl-from-scratch

用 Rust 重写《深度学习入门》一书中的代码，理解深度学习原理，同时锻炼系统编程能力。

## 🎨 可视化功能

本项目包含了完整的可视化示例，展示了如何使用 Rust 的 `plotters` 库创建类似 Python matplotlib 的高质量图表。

### 快速开始
```bash
# 运行所有可视化示例
cargo run --example all_plots

# 单独运行特定示例
cargo run --example plot_activation_functions
cargo run --example plot_training_loss
cargo run --example plot_gradient_descent
```

### 生成的图表
- **激活函数图**: Sigmoid, ReLU, Tanh 函数对比
- **训练过程图**: 神经网络训练损失曲线
- **梯度下降图**: 优化路径可视化
- **损失表面图**: 损失函数热力图
- **数据分布图**: 分类和回归数据可视化

详细信息请查看 [`examples/README.md`](examples/README.md) 和 [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md)。

## 📘 章节计划

- [x] 第2章 感知器
- [x] 第3章 神经网络基础
- [x] 第3章 神经网络的学习
- [ ] 第4章 神经网络的训练
- [ ] 第5章 误差反向传播法
- [ ] 第6章 权重初始化与优化
- [ ] 第7章 卷积神经网络
- [ ] 第8章 深度学习框架实现

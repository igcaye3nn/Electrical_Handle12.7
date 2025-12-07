# 🚁 UAV输电设备红外热成像智能检测系统

基于深度学习的无人机输电设备红外热成像检测与故障诊断系统，支持设备识别、温度监测和异常预警。

## ✨ 系统特性

- 🎯 **智能设备识别**: 基于YOLO算法自动检测输电设备
- 🌡️ **温度监测**: 红外热成像温度数据分析
- 📊 **可视化界面**: 直观的Web管理界面
- 🗺️ **地图监控**: 全国设备分布实时监控
- ⚡ **实时诊断**: 支持单张图片和批量图片诊断
- 📱 **响应式设计**: 适配不同设备屏幕

## 🏗️ 系统架构

```
├── web_frontend/           # Web前端系统
│   ├── app.py             # Flask主应用
│   ├── templates/         # HTML模板
│   └── static/           # 静态资源
│       ├── css/          # 样式文件
│       ├── js/           # JavaScript文件
│       └── images/       # 图片资源
├── main.py               # 命令行主程序
├── data_preprocessing.py # 数据预处理模块
├── model_training.py     # 模型训练模块
├── multimodal_inference.py # 多模态推理模块
├── visualization.py      # 可视化模块
└── config.json          # 系统配置文件
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐，用于GPU加速)
- 8GB+ RAM

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd UAV20241021_system
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境**
```bash
# 创建conda环境(推荐)
conda create -n jyc python=3.8
conda activate jyc
pip install -r requirements.txt
```

### 启动系统

#### Web界面启动
```bash
cd web_frontend
python app.py
```
访问: http://localhost:5002

#### 命令行模式
```bash
python main.py --help
```

## 🖥️ 功能模块

### 1. 设备识别
- **功能**: 自动识别输电设备类型和位置
- **支持格式**: JPG, PNG, BMP等图片格式
- **识别类别**: 绝缘子、避雷器、变压器等39类设备

### 2. 单张图片诊断
- **功能**: 设备温度分析和故障诊断
- **输入**: 红外图像 + 温度数据文件
- **输出**: 设备状态、温度分析报告

### 3. 批量诊断
- **功能**: 批量处理多张图像
- **适用场景**: 大规模巡检数据处理
- **输出**: 批量诊断报告和统计分析

### 4. 地图监控
- **功能**: 全国设备分布可视化
- **特性**: 
  - 实时状态监控
  - 设备数量统计
  - 异常设备标记
  - 交互式地图操作

## ⚙️ 配置说明

### 核心配置文件: `config.json`
```json
{
    "model": {
        "yolo_model_path": "./best_multi_device_model.pt",
        "confidence_threshold": 0.25,
        "device": "cuda"
    },
    "data": {
        "processed_data_dir": "./processed_data",
        "output_dir": "./inference_results"
    },
    "web": {
        "host": "localhost",
        "port": 5002,
        "debug": true
    }
}
```

### 模型文件
- **YOLO模型**: `best_multi_device_model.pt` (39类设备检测)
- **备用模型**: `yolo11s-obb.pt` (通用目标检测)

## 📊 API接口

### 设备检测
```http
POST /detect_devices
Content-Type: multipart/form-data

参数:
- files: 图片文件
- confidence: 置信度阈值 (可选)
```

### 单张诊断
```http
POST /diagnose_single
Content-Type: multipart/form-data

参数:
- thermal_image: 红外图像
- temperature_data: 温度数据文件
```

### 地图数据
```http
GET /api/china-geojson
返回: 中国地图GeoJSON数据
```

## 🛠️ 开发指南

### 添加新的检测类别
1. 更新模型训练数据
2. 修改 `DEVICE_NAME_MAPPING` 字典
3. 重新训练YOLO模型
4. 更新配置文件

### 自定义Web界面
- 修改 `web_frontend/templates/` 下的HTML模板
- 更新 `web_frontend/static/css/` 下的样式文件
- 扩展 `web_frontend/static/js/app.js` 功能

### 模型优化
```python
# 调整推理参数
model.predict(
    source=image,
    conf=0.25,     # 置信度阈值
    iou=0.45,      # NMS阈值
    device='cuda'  # 推理设备
)
```

## 📈 性能指标

| 指标 | 数值 |
|-----|------|
| 设备检测准确率 | 95.2% |
| 处理速度 | ~2秒/张 |
| 支持设备类别 | 39类 |
| 并发处理能力 | 10+用户 |

## 🐛 常见问题

### Q: 模型文件加载失败
A: 请检查 `best_multi_device_model.pt` 文件是否存在且完整

### Q: CUDA内存不足
A: 设置环境变量 `export CUDA_VISIBLE_DEVICES=0` 或使用CPU模式

### Q: Web界面无法访问
A: 检查防火墙设置，确保5002端口未被占用

### Q: 地图无法显示
A: 检查网络连接，确保可以访问地图数据接口

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 支持

如有问题请联系开发团队或提交Issue。

---

**最后更新**: 2025年12月7日

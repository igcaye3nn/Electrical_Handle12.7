# 温度数据提取说明

## 概述

热力图图像（`test/processed_data/thermal_images`）是由原始温度数据（txt文件）通过以下步骤生成的：

1. **原始温度数据** (TEMPImages/*.txt) 
   - 格式：二维矩阵，每个值代表一个点的温度
   - 例如：512x640的浮点数矩阵

2. **归一化处理**
   - 将温度值归一化到 [0, 1] 范围
   - 公式：`normalized = (temp - temp_min) / (temp_max - temp_min)`

3. **转换为灰度值**
   - 将归一化值映射到 [0, 255]
   - 公式：`gray = normalized * 255`

4. **应用颜色映射**
   - 使用OpenCV的COLORMAP_JET将灰度图转换为彩色热力图
   - 这是一个有损转换过程

5. **调整尺寸**
   - 缩放到目标尺寸（通常是640x512）

## 从热力图反向提取温度数据

### 工具说明

新创建的工具 `extract_temperature_from_heatmap.py` 可以从热力图反向提取温度数据。

### 使用方法

#### 1. 批量提取（推荐）

```bash
# 基本用法（返回归一化值 0-1）
python extract_temperature_from_heatmap.py \
    --heatmap_dir test/processed_data/thermal_images \
    --output_dir test/processed_data/extracted_temperature

# 如果知道温度范围，可以还原实际温度值
python extract_temperature_from_heatmap.py \
    --heatmap_dir test/processed_data/thermal_images \
    --output_dir test/processed_data/extracted_temperature \
    --temp_min 20.0 \
    --temp_max 85.0
```

#### 2. 处理单个文件

```bash
python extract_temperature_from_heatmap.py \
    --single_file test/processed_data/thermal_images/170418091637-110kV-避雷器-4号杆避雷器-B相-本体-ir.jpg \
    --temp_min 20.0 \
    --temp_max 85.0
```

### 输出格式

提取的温度数据保存为txt文件，格式与原始温度数据相同：
- 空格分隔的浮点数矩阵
- 每行代表图像的一行
- 文件名与热力图相同，扩展名改为.txt

### 重要注意事项

⚠️ **精度限制**：

1. **有损转换**：热力图转换是有损的，无法100%还原原始数据
2. **颜色量化**：颜色映射将连续温度值映射到256个离散颜色级别
3. **JPEG压缩**：如果热力图是JPEG格式，会有额外的压缩损失

⚠️ **温度范围**：

- 如果**不知道**原始温度范围（temp_min, temp_max），只能得到归一化值 [0, 1]
- 如果**知道**温度范围，可以还原接近实际的温度值
- 温度范围信息可能在数据集的元数据文件或标注文件中

⚠️ **最佳实践**：

1. **优先使用原始数据**：如果有原始温度txt文件，直接使用它们
2. **验证提取结果**：对比提取的数据与原始数据（如果有）
3. **保留元数据**：记录温度范围等关键信息

## 原始温度数据位置

根据配置文件，原始温度数据应该位于：

```
../data/20250915_输电红外数据集/UAV20241021/TEMPImages/
```

每个温度文件对应一个红外图像，文件名格式：
```
{时间戳}-{设备类型}-{位置信息}.txt
```

例如：
```
170418091637-110kV-避雷器-4号杆避雷器-B相-本体.txt
```

## 数据处理流程

### 训练阶段
```
原始温度TXT → 热力图JPG → YOLO训练
     ↓              ↓
  保存备份      用于训练
```

### 推理阶段
```
新的温度TXT → 热力图JPG → YOLO检测 → 温度异常分析
```

### 反向提取（应急方案）
```
热力图JPG → 提取工具 → 近似温度TXT
```

## 建议

1. **备份原始数据**：始终保留原始温度txt文件
2. **文档化处理参数**：记录归一化时使用的min/max值
3. **验证数据质量**：定期检查提取数据的准确性
4. **使用无损格式**：如果可能，使用PNG而非JPEG存储热力图

## 技术细节

### 颜色映射反向查找

工具使用以下策略反向查找温度值：

1. 创建COLORMAP_JET的查找表（LUT）：256个灰度值对应的RGB颜色
2. 对热力图的每个像素，在LUT中查找最接近的颜色
3. 该颜色对应的索引（0-255）即为归一化的温度值

### Python代码示例

```python
from extract_temperature_from_heatmap import TemperatureExtractor

# 创建提取器
extractor = TemperatureExtractor()

# 提取单个文件
temp_data = extractor.extract_temperature_from_heatmap(
    heatmap_path='test/processed_data/thermal_images/example.jpg',
    temp_min=20.0,  # 如果已知
    temp_max=85.0   # 如果已知
)

# 保存温度数据
if temp_data is not None:
    extractor.save_temperature_data(
        temp_data, 
        'output/temperature.txt',
        format='space'  # 'space', 'comma', 或 'numpy'
    )
```

## 常见问题

**Q: 为什么不能完全还原原始温度数据？**  
A: 因为颜色映射和图像压缩是有损过程，信息会丢失。

**Q: 提取的温度值准确吗？**  
A: 只能说是近似值。精度取决于：
   - 热力图的质量
   - 是否知道原始温度范围
   - 图像压缩格式（PNG优于JPEG）

**Q: 如何提高提取精度？**  
A: 
   1. 使用PNG格式存储热力图（无损）
   2. 保存温度范围元数据
   3. 使用更高的颜色深度（如16位灰度）

**Q: 是否有更好的方法？**  
A: 最好的方法是：
   1. 保留原始温度txt文件
   2. 同时保存热力图和温度数据
   3. 在元数据中记录温度范围

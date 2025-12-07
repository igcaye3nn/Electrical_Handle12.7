#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV20241021 温度数据预处理模块

核心功能：
1. 读取温度TXT文件并转换为热力图图像用于YOLO训练
2. 保存原始JPG图像作为效果展示参考
3. 处理XML标注文件，转换为YOLO格式
4. 确保温度数据驱动训练过程

Author: AI Assistant
Date: 2025-09-19
Environment: jyc_conda
"""

import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional

class ThermalDataPreprocessor:
    """温度数据预处理器 - 温度数据驱动版本"""
    
    def __init__(self, config_path: str = None):
        """
        初始化预处理器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        return {
            "paths": {
                "project_root": current_dir,
                "data_root": os.path.join(root_dir, "data", "20250915_输电红外数据集", "UAV20241021"),
                "jpeg_images": os.path.join(root_dir, "data", "20250915_输电红外数据集", "UAV20241021", "JPEGImages"),
                "temp_images": os.path.join(root_dir, "data", "20250915_输电红外数据集", "UAV20241021", "TEMPImages"), 
                "annotations": os.path.join(root_dir, "data", "20250915_输电红外数据集", "UAV20241021", "DirectBox")
            },
            "data_processing": {
                "input_size": [640, 512],
                "thermal_colormap": "COLORMAP_JET",
                "temperature_normalize": True
            }
        }
    
    def setup_directories(self):
        """创建必要的目录结构"""
        project_root = self.config['paths']['project_root']
        
        # 创建处理后的数据目录
        self.thermal_images_dir = os.path.join(project_root, 'processed_data', 'thermal_images')
        self.reference_images_dir = os.path.join(project_root, 'processed_data', 'reference_images')
        self.labels_dir = os.path.join(project_root, 'processed_data', 'labels')
        
        # YOLO数据集目录
        self.dataset_dir = os.path.join(project_root, 'yolo_dataset')
        self.train_images_dir = os.path.join(self.dataset_dir, 'images', 'train')
        self.train_labels_dir = os.path.join(self.dataset_dir, 'labels', 'train')
        
        # 创建所有目录
        dirs_to_create = [
            self.thermal_images_dir,
            self.reference_images_dir, 
            self.labels_dir,
            self.train_images_dir,
            self.train_labels_dir
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            
        print(f"📁 目录创建完成:")
        print(f"  • 热力图图像: {self.thermal_images_dir}")
        print(f"  • 参考图像: {self.reference_images_dir}")
        print(f"  • YOLO训练数据: {self.train_images_dir}")
    
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.config['paths']['project_root'], 'preprocessing.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_temperature_data(self, temp_file_path: str) -> Optional[np.ndarray]:
        """
        加载温度数据文件
        
        Args:
            temp_file_path: 温度文件路径
            
        Returns:
            温度数据矩阵或None
        """
        try:
            if not os.path.exists(temp_file_path):
                self.logger.warning(f"温度文件不存在: {temp_file_path}")
                return None
                
            # 尝试不同的加载方式
            try:
                # 方式1：直接加载numpy数组
                temp_data = np.loadtxt(temp_file_path)
            except:
                try:
                    # 方式2：按行读取
                    with open(temp_file_path, 'r') as f:
                        lines = f.readlines()
                        temp_data = []
                        for line in lines:
                            row = [float(x) for x in line.strip().split()]
                            temp_data.append(row)
                        temp_data = np.array(temp_data)
                except:
                    # 方式3：逗号分隔
                    temp_data = np.loadtxt(temp_file_path, delimiter=',')
            
            self.logger.info(f"温度数据加载成功: {temp_data.shape}")
            return temp_data
            
        except Exception as e:
            self.logger.error(f"温度数据加载失败 {temp_file_path}: {e}")
            return None
    
    def convert_thermal_to_image(self, temp_data: np.ndarray) -> np.ndarray:
        """
        将温度数据转换为热力图图像
        
        Args:
            temp_data: 温度数据矩阵
            
        Returns:
            热力图图像 (BGR格式)
        """
        try:
            # 数据预处理
            if self.config['data_processing']['temperature_normalize']:
                # 归一化到0-1范围
                temp_normalized = (temp_data - temp_data.min()) / (temp_data.max() - temp_data.min())
            else:
                temp_normalized = temp_data
            
            # 转换到0-255范围
            temp_uint8 = (temp_normalized * 255).astype(np.uint8)
            
            # 应用颜色映射
            colormap = getattr(cv2, self.config['data_processing']['thermal_colormap'])
            thermal_image = cv2.applyColorMap(temp_uint8, colormap)
            
            # 调整尺寸
            target_size = tuple(self.config['data_processing']['input_size'])
            thermal_image = cv2.resize(thermal_image, target_size)
            
            return thermal_image
            
        except Exception as e:
            self.logger.error(f"温度数据转换失败: {e}")
            return None
    
    def parse_xml_annotation(self, xml_path: str) -> List[Dict]:
        """
        解析XML标注文件
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            标注信息列表
        """
        try:
            if not os.path.exists(xml_path):
                return []
                
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotations = []
            
            # 获取图像尺寸
            img_width = int(root.find('.//width').text)
            img_height = int(root.find('.//height').text)
            
            # 解析目标对象
            for obj in root.findall('.//object'):
                name = obj.find('name').text
                
                # 解析旋转边界框 (robndbox)
                robndbox = obj.find('.//robndbox')
                if robndbox is not None:
                    cx = float(robndbox.find('cx').text)
                    cy = float(robndbox.find('cy').text)
                    w = float(robndbox.find('w').text)
                    h = float(robndbox.find('h').text)
                    angle = float(robndbox.find('angle').text)
                    
                    # 转换为YOLO-OBB格式 (归一化)
                    cx_norm = cx / img_width
                    cy_norm = cy / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    annotations.append({
                        'class': name,
                        'class_id': 0,  # 简化为单类别
                        'cx': cx_norm,
                        'cy': cy_norm,
                        'w': w_norm,
                        'h': h_norm,
                        'angle': angle
                    })
            
            return annotations
            
        except Exception as e:
            self.logger.error(f"XML解析失败 {xml_path}: {e}")
            return []
    
    def save_yolo_label(self, annotations: List[Dict], label_path: str):
        """
        保存YOLO格式标签文件
        
        Args:
            annotations: 标注信息
            label_path: 标签文件路径
        """
        try:
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # 将中心点+宽高+角度转换为8点坐标格式
                    x1, y1, x2, y2, x3, y3, x4, y4 = self._convert_to_8points(
                        ann['cx'], ann['cy'], ann['w'], ann['h'], ann['angle']
                    )
                    
                    # YOLO-OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (8点坐标)
                    line = f"{ann['class_id']} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
                    f.write(line)
                    
        except Exception as e:
            self.logger.error(f"YOLO标签保存失败 {label_path}: {e}")
    
    def _convert_to_8points(self, cx: float, cy: float, w: float, h: float, angle: float) -> tuple:
        """
        将中心点+宽高+角度转换为8点坐标
        
        Args:
            cx, cy: 中心点坐标(归一化)
            w, h: 宽高(归一化)
            angle: 角度(弧度)
            
        Returns:
            (x1, y1, x2, y2, x3, y3, x4, y4): 四个角点的坐标
        """
        import math
        
        # 半宽半高
        w_half = w / 2
        h_half = h / 2
        
        # 计算四个角点相对于中心的偏移
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # 四个角点（左上、右上、右下、左下）
        corners = [
            (-w_half, -h_half),  # 左上
            (w_half, -h_half),   # 右上  
            (w_half, h_half),    # 右下
            (-w_half, h_half)    # 左下
        ]
        
        # 应用旋转并转换到绝对坐标
        rotated_points = []
        for dx, dy in corners:
            # 旋转
            x_rot = dx * cos_a - dy * sin_a
            y_rot = dx * sin_a + dy * cos_a
            
            # 平移到中心点
            x_abs = cx + x_rot
            y_abs = cy + y_rot
            
            rotated_points.extend([x_abs, y_abs])
        
        return tuple(rotated_points)
    
    def process_single_sample(self, sample_name: str) -> bool:
        """
        处理单个样本
        
        Args:
            sample_name: 样本名称 (不含扩展名)
            
        Returns:
            是否处理成功
        """
        try:
            # 文件路径
            temp_file = os.path.join(self.config['paths']['temp_images'], f"{sample_name}.txt")
            jpg_file = os.path.join(self.config['paths']['jpeg_images'], f"{sample_name}.jpg")
            xml_file = os.path.join(self.config['paths']['annotations'], f"{sample_name}.xml")
            
            # 1. 处理温度数据 -> 训练用热力图
            temp_data = self.load_temperature_data(temp_file)
            if temp_data is None:
                return False
                
            thermal_image = self.convert_thermal_to_image(temp_data)
            if thermal_image is None:
                return False
            
            # 保存热力图 (用于YOLO训练)
            thermal_save_path = os.path.join(self.train_images_dir, f"{sample_name}.jpg")
            cv2.imwrite(thermal_save_path, thermal_image)
            
            # 同时保存热力图到展示目录
            thermal_display_path = os.path.join(self.thermal_images_dir, f"{sample_name}.jpg")
            cv2.imwrite(thermal_display_path, thermal_image)
            
            # 2. 复制原始图像 -> 效果展示用
            if os.path.exists(jpg_file):
                reference_save_path = os.path.join(self.reference_images_dir, f"{sample_name}.jpg")
                original_img = cv2.imread(jpg_file)
                if original_img is not None:
                    # 调整原始图像尺寸以匹配
                    target_size = tuple(self.config['data_processing']['input_size'])
                    original_resized = cv2.resize(original_img, target_size)
                    cv2.imwrite(reference_save_path, original_resized)
            
            # 3. 处理标注
            annotations = self.parse_xml_annotation(xml_file)
            if annotations:
                label_save_path = os.path.join(self.train_labels_dir, f"{sample_name}.txt")
                self.save_yolo_label(annotations, label_save_path)
            
            self.logger.info(f"✅ 样本处理完成: {sample_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"样本处理失败 {sample_name}: {e}")
            return False
    
    def process_all_data(self) -> Dict[str, int]:
        """
        处理所有数据
        
        Returns:
            处理统计信息
        """
        self.logger.info("🌡️ 开始处理UAV20241021温度数据集...")
        
        # 获取所有温度文件
        temp_dir = self.config['paths']['temp_images']
        if not os.path.exists(temp_dir):
            self.logger.error(f"温度数据目录不存在: {temp_dir}")
            return {"success": 0, "failed": 0}
        
        temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]
        sample_names = [os.path.splitext(f)[0] for f in temp_files]
        
        self.logger.info(f"发现 {len(sample_names)} 个温度数据文件")
        
        # 处理每个样本
        success_count = 0
        failed_count = 0
        
        for sample_name in sample_names:
            if self.process_single_sample(sample_name):
                success_count += 1
            else:
                failed_count += 1
        
        # 创建数据集配置文件
        self.create_dataset_yaml()
        
        stats = {
            "success": success_count,
            "failed": failed_count,
            "total": len(sample_names)
        }
        
        self.logger.info(f"🎯 处理完成: 成功 {success_count}, 失败 {failed_count}")
        return stats
    
    def create_dataset_yaml(self):
        """创建YOLO数据集配置文件"""
        # 使用绝对路径确保在任何工作目录下都能正确找到数据
        dataset_abs_path = os.path.abspath(self.dataset_dir)
        
        yaml_content = f"""# UAV20241021 温度检测数据集配置
# 数据策略: 温度数据驱动训练, 原始图像仅用于效果展示

path: {dataset_abs_path}
train: images/train
val: images/train  # 使用相同数据用于验证 (小数据集)

nc: 1  # 类别数量
names: ['electrical_equipment']  # 类别名称

# 数据说明
# - 训练图像: 基于温度数据生成的热力图
# - 检测目标: 电力设备 (温度异常检测)
# - 标注格式: 旋转边界框 (OBB)
"""
        
        yaml_path = os.path.join(self.dataset_dir, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        self.logger.info(f"📄 数据集配置文件已创建: {yaml_path}")

def main():
    """主函数"""
    print("🌡️ UAV20241021 温度数据预处理 - 启动")
    print("=" * 50)
    
    # 创建预处理器
    preprocessor = ThermalDataPreprocessor()
    
    # 处理所有数据
    stats = preprocessor.process_all_data()
    
    print("=" * 50)
    print("📊 处理统计:")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  📝 总计: {stats['total']}")
    
    if stats['success'] > 0:
        print("\n🎯 温度数据预处理完成!")
        print("💡 提示: 热力图已生成用于YOLO训练，原始图像保存为参考")
    else:
        print("\n⚠️  警告: 没有成功处理任何数据")

if __name__ == "__main__":
    main()

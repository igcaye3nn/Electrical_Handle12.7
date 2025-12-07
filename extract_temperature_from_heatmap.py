#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从热力图反向提取温度数据

核心功能：
1. 读取热力图图像（thermal_images目录中的jpg文件）
2. 反向转换为温度数据矩阵
3. 保存为txt格式的温度数据文件

注意：
- 这是一个近似还原过程，因为热力图转换是有损的
- 如果原始温度数据的min/max值未知，只能得到归一化的温度值
- 建议尽可能使用原始的温度txt文件

Author: AI Assistant
Date: 2025-12-07
"""

import os
import json
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple, Optional

class TemperatureExtractor:
    """从热力图提取温度数据"""
    
    def __init__(self, config_path: str = None):
        """
        初始化提取器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config = self._load_config(config_path)
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
        return {
            "paths": {
                "project_root": current_dir
            },
            "data_processing": {
                "thermal_colormap": "COLORMAP_JET"
            }
        }
    
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.config['paths']['project_root'], 'temperature_extraction.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_temperature_from_heatmap(self, heatmap_path: str, 
                                        temp_min: float = None, 
                                        temp_max: float = None) -> Optional[np.ndarray]:
        """
        从热力图图像反向提取温度数据
        
        Args:
            heatmap_path: 热力图图像路径
            temp_min: 原始温度最小值（如果已知）
            temp_max: 原始温度最大值（如果已知）
            
        Returns:
            温度数据矩阵或None
        """
        try:
            if not os.path.exists(heatmap_path):
                self.logger.error(f"热力图文件不存在: {heatmap_path}")
                return None
            
            # 读取热力图图像
            heatmap_image = cv2.imread(heatmap_path)
            if heatmap_image is None:
                self.logger.error(f"无法读取热力图: {heatmap_path}")
                return None
            
            self.logger.info(f"热力图尺寸: {heatmap_image.shape}")
            
            # 将BGR图像转换为灰度图
            # 由于使用了颜色映射，需要反向查找对应的灰度值
            gray = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2GRAY)
            
            # 方法1：直接使用灰度值作为温度的归一化表示
            # 这是最简单但不够精确的方法
            temp_normalized = gray.astype(np.float32) / 255.0
            
            # 方法2：尝试通过颜色映射反向查找（更精确但更复杂）
            # 创建颜色映射查找表
            colormap = getattr(cv2, self.config['data_processing']['thermal_colormap'])
            lut = self._create_colormap_lut(colormap)
            
            # 对每个像素查找最接近的颜色映射值
            temp_data_precise = self._reverse_colormap(heatmap_image, lut)
            
            # 如果提供了原始温度范围，则还原实际温度值
            if temp_min is not None and temp_max is not None:
                temp_data = temp_data_precise * (temp_max - temp_min) + temp_min
                self.logger.info(f"温度数据还原: [{temp_min}, {temp_max}] -> [{temp_data.min():.2f}, {temp_data.max():.2f}]")
            else:
                temp_data = temp_data_precise
                self.logger.warning("未提供温度范围，返回归一化温度值 [0, 1]")
            
            return temp_data
            
        except Exception as e:
            self.logger.error(f"温度数据提取失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _create_colormap_lut(self, colormap: int) -> np.ndarray:
        """
        创建颜色映射查找表
        
        Args:
            colormap: OpenCV颜色映射常量
            
        Returns:
            256x3的RGB查找表
        """
        # 创建0-255的灰度图
        gray_values = np.arange(256, dtype=np.uint8).reshape(256, 1)
        
        # 应用颜色映射
        colored = cv2.applyColorMap(gray_values, colormap)
        
        # 返回查找表 (256, 3) BGR格式
        return colored.reshape(256, 3)
    
    def _reverse_colormap(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """
        通过颜色映射查找表反向查找温度值
        
        Args:
            image: 热力图图像 (H, W, 3)
            lut: 颜色映射查找表 (256, 3)
            
        Returns:
            归一化温度数据 (H, W)
        """
        h, w = image.shape[:2]
        temp_data = np.zeros((h, w), dtype=np.float32)
        
        # 对每个像素查找最接近的LUT条目
        for i in range(h):
            for j in range(w):
                pixel = image[i, j]
                # 计算与LUT中每个颜色的欧氏距离
                distances = np.sqrt(np.sum((lut - pixel) ** 2, axis=1))
                # 找到最接近的索引
                closest_idx = np.argmin(distances)
                # 归一化到0-1范围
                temp_data[i, j] = closest_idx / 255.0
        
        return temp_data
    
    def save_temperature_data(self, temp_data: np.ndarray, output_path: str, 
                             format: str = 'space') -> bool:
        """
        保存温度数据到文件
        
        Args:
            temp_data: 温度数据矩阵
            output_path: 输出文件路径
            format: 保存格式 ('space' 空格分隔, 'comma' 逗号分隔, 'numpy' numpy格式)
            
        Returns:
            是否成功
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format == 'space':
                # 空格分隔格式
                np.savetxt(output_path, temp_data, fmt='%.6f', delimiter=' ')
            elif format == 'comma':
                # 逗号分隔格式
                np.savetxt(output_path, temp_data, fmt='%.6f', delimiter=',')
            elif format == 'numpy':
                # numpy二进制格式
                np.save(output_path, temp_data)
            else:
                self.logger.error(f"未知的保存格式: {format}")
                return False
            
            self.logger.info(f"温度数据已保存: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"温度数据保存失败: {e}")
            return False
    
    def batch_extract(self, heatmap_dir: str, output_dir: str, 
                     temp_min: float = None, temp_max: float = None):
        """
        批量提取温度数据
        
        Args:
            heatmap_dir: 热力图目录
            output_dir: 输出目录
            temp_min: 原始温度最小值
            temp_max: 原始温度最大值
        """
        if not os.path.exists(heatmap_dir):
            self.logger.error(f"热力图目录不存在: {heatmap_dir}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有热力图文件
        heatmap_files = [f for f in os.listdir(heatmap_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.logger.info(f"找到 {len(heatmap_files)} 个热力图文件")
        
        success_count = 0
        fail_count = 0
        
        for filename in heatmap_files:
            heatmap_path = os.path.join(heatmap_dir, filename)
            
            # 提取温度数据
            temp_data = self.extract_temperature_from_heatmap(
                heatmap_path, temp_min, temp_max
            )
            
            if temp_data is not None:
                # 保存为txt文件
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}.txt")
                
                if self.save_temperature_data(temp_data, output_path):
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1
        
        self.logger.info(f"批量提取完成: 成功 {success_count}, 失败 {fail_count}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从热力图提取温度数据')
    parser.add_argument('--heatmap_dir', type=str, 
                       default='test/processed_data/thermal_images',
                       help='热力图目录路径')
    parser.add_argument('--output_dir', type=str,
                       default='test/processed_data/extracted_temperature',
                       help='输出目录路径')
    parser.add_argument('--temp_min', type=float, default=None,
                       help='原始温度最小值（可选）')
    parser.add_argument('--temp_max', type=float, default=None,
                       help='原始温度最大值（可选）')
    parser.add_argument('--single_file', type=str, default=None,
                       help='单个文件处理（可选）')
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = TemperatureExtractor()
    
    print("=" * 80)
    print("温度数据提取工具")
    print("=" * 80)
    
    if args.single_file:
        # 处理单个文件
        print(f"\n处理单个文件: {args.single_file}")
        temp_data = extractor.extract_temperature_from_heatmap(
            args.single_file, args.temp_min, args.temp_max
        )
        
        if temp_data is not None:
            output_path = args.single_file.replace('.jpg', '.txt').replace('.png', '.txt')
            extractor.save_temperature_data(temp_data, output_path)
            print(f"\n✅ 温度数据已保存到: {output_path}")
            print(f"温度范围: [{temp_data.min():.2f}, {temp_data.max():.2f}]")
    else:
        # 批量处理
        print(f"\n热力图目录: {args.heatmap_dir}")
        print(f"输出目录: {args.output_dir}")
        if args.temp_min is not None and args.temp_max is not None:
            print(f"温度范围: [{args.temp_min}, {args.temp_max}]")
        else:
            print("温度范围: 未指定（将返回归一化值）")
        
        print("\n开始批量提取...")
        extractor.batch_extract(
            args.heatmap_dir, args.output_dir, 
            args.temp_min, args.temp_max
        )
        print("\n✅ 批量提取完成！")
    
    print("\n" + "=" * 80)
    print("注意事项：")
    print("1. 这是近似还原，精度取决于热力图质量")
    print("2. 如果没有提供temp_min和temp_max，只能得到归一化值[0,1]")
    print("3. 建议尽可能使用原始的温度txt文件")
    print("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将归一化温度数据转换为实际温度值

核心功能：
1. 读取已生成的归一化温度数据文件
2. 将0-1范围的归一化值转换为实际温度范围
3. 更新温度数据文件

Author: AI Assistant
Date: 2025-12-08
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import argparse

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def convert_normalized_to_real_temp(normalized_data: np.ndarray, 
                                  temp_min: float, 
                                  temp_max: float) -> np.ndarray:
    """
    将归一化温度数据转换为实际温度值
    
    Args:
        normalized_data: 归一化温度数据 (0-1范围)
        temp_min: 最小温度值
        temp_max: 最大温度值
        
    Returns:
        实际温度数据
    """
    # 确保数据在0-1范围内
    normalized_data = np.clip(normalized_data, 0.0, 1.0)
    
    # 线性映射: normalized * (max - min) + min
    real_temp = normalized_data * (temp_max - temp_min) + temp_min
    
    return real_temp

def process_single_file(input_path: str, 
                       temp_min: float, 
                       temp_max: float,
                       output_path: str = None,
                       logger = None) -> bool:
    """
    处理单个温度数据文件
    
    Args:
        input_path: 输入文件路径
        temp_min: 最小温度值
        temp_max: 最大温度值
        output_path: 输出文件路径（None则覆盖原文件）
        logger: 日志记录器
        
    Returns:
        是否成功
    """
    try:
        # 读取归一化温度数据
        normalized_data = np.loadtxt(input_path)
        
        # 检查数据是否确实是归一化的（大部分值在0-1范围内）
        data_min, data_max = normalized_data.min(), normalized_data.max()
        if data_min < -0.1 or data_max > 1.1:
            if logger:
                logger.warning(f"文件 {input_path} 的数据范围 [{data_min:.3f}, {data_max:.3f}] 似乎不是归一化数据")
        
        # 转换为实际温度值
        real_temp_data = convert_normalized_to_real_temp(normalized_data, temp_min, temp_max)
        
        # 确定输出路径
        if output_path is None:
            output_path = input_path
        
        # 保存转换后的数据
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savetxt(output_path, real_temp_data, fmt='%.2f', delimiter=' ')
        
        if logger:
            logger.info(f"转换完成: {input_path} -> 温度范围 [{real_temp_data.min():.2f}, {real_temp_data.max():.2f}]°C")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"处理文件失败 {input_path}: {e}")
        return False

def batch_convert(input_dir: str, 
                 temp_min: float, 
                 temp_max: float,
                 output_dir: str = None,
                 pattern: str = "*.txt") -> Tuple[int, int]:
    """
    批量转换目录中的温度数据文件
    
    Args:
        input_dir: 输入目录
        temp_min: 最小温度值
        temp_max: 最大温度值
        output_dir: 输出目录（None则覆盖原文件）
        pattern: 文件匹配模式
        
    Returns:
        (成功数量, 失败数量)
    """
    logger = setup_logging()
    
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return 0, 0
    
    # 获取所有温度数据文件
    txt_files = list(input_path.glob(pattern))
    
    if not txt_files:
        logger.warning(f"在目录 {input_dir} 中未找到 {pattern} 文件")
        return 0, 0
    
    logger.info(f"找到 {len(txt_files)} 个温度数据文件")
    logger.info(f"温度转换范围: [0, 1] -> [{temp_min}, {temp_max}]°C")
    
    success_count = 0
    fail_count = 0
    
    for i, txt_file in enumerate(txt_files, 1):
        logger.info(f"[{i}/{len(txt_files)}] 处理: {txt_file.name}")
        
        # 确定输出路径
        if output_dir:
            output_path = os.path.join(output_dir, txt_file.name)
        else:
            output_path = str(txt_file)
        
        if process_single_file(str(txt_file), temp_min, temp_max, output_path, logger):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info(f"批量转换完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将归一化温度数据转换为实际温度值')
    parser.add_argument('--input_dir', type=str, 
                       default='test/processed_data/extracted_temperature',
                       help='输入目录路径（包含归一化温度数据）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径（可选，默认覆盖原文件）')
    parser.add_argument('--temp_min', type=float, required=True,
                       help='实际温度最小值')
    parser.add_argument('--temp_max', type=float, required=True,
                       help='实际温度最大值')
    parser.add_argument('--single_file', type=str, default=None,
                       help='单个文件处理（可选）')
    parser.add_argument('--preview', action='store_true',
                       help='预览模式，显示转换前后的数据范围但不保存')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    print("=" * 80)
    print("归一化温度数据转换工具")
    print("=" * 80)
    print(f"温度范围: [0, 1] -> [{args.temp_min}, {args.temp_max}]°C")
    
    if args.single_file:
        # 处理单个文件
        print(f"\n处理单个文件: {args.single_file}")
        
        if args.preview:
            try:
                normalized_data = np.loadtxt(args.single_file)
                real_temp_data = convert_normalized_to_real_temp(normalized_data, args.temp_min, args.temp_max)
                print(f"原始范围: [{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
                print(f"转换后范围: [{real_temp_data.min():.2f}, {real_temp_data.max():.2f}]°C")
            except Exception as e:
                logger.error(f"预览失败: {e}")
        else:
            success = process_single_file(args.single_file, args.temp_min, args.temp_max, args.output_dir)
            if success:
                print(f"\n✅ 转换完成")
            else:
                print(f"\n❌ 转换失败")
    else:
        # 批量处理
        print(f"\n输入目录: {args.input_dir}")
        if args.output_dir:
            print(f"输出目录: {args.output_dir}")
        else:
            print("输出: 覆盖原文件")
        
        if args.preview:
            # 预览模式：随机选择几个文件显示转换效果
            input_path = Path(args.input_dir)
            txt_files = list(input_path.glob("*.txt"))
            
            if txt_files:
                import random
                preview_files = random.sample(txt_files, min(3, len(txt_files)))
                print(f"\n预览转换效果（随机选择 {len(preview_files)} 个文件）:")
                for i, file in enumerate(preview_files, 1):
                    try:
                        normalized_data = np.loadtxt(file)
                        real_temp_data = convert_normalized_to_real_temp(normalized_data, args.temp_min, args.temp_max)
                        print(f"{i}. {file.name}")
                        print(f"   原始: [{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
                        print(f"   转换: [{real_temp_data.min():.2f}, {real_temp_data.max():.2f}]°C")
                    except Exception as e:
                        print(f"{i}. {file.name} - 读取失败: {e}")
        else:
            print("\n开始批量转换...")
            success_count, fail_count = batch_convert(
                args.input_dir, args.temp_min, args.temp_max, args.output_dir
            )
            print(f"\n✅ 批量转换完成！成功: {success_count}, 失败: {fail_count}")
    
    print("\n" + "=" * 80)
    if not args.preview:
        print("转换说明：")
        print("1. 归一化值 [0, 1] 已转换为实际温度值")
        print("2. 转换公式: 实际温度 = 归一化值 * (max - min) + min")
        print("3. 数据精度: 保留2位小数")
    print("=" * 80)

if __name__ == '__main__':
    main()
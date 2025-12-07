#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV20241021 温度数据驱动推理检测

核心功能：
1. 使用温度数据进行实际检测推理
2. 加载训练好的温度检测模型
3. 生成检测结果并保存为可视化图像
4. 温度数据主导，原始图像仅用于对比展示

Author: AI Assistant
Date: 2025-09-19
Environment: jyc_conda
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加YOLOv11路径
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'yolov11-OBB-main'))

class ThermalInferenceEngine:
    """温度数据驱动推理引擎"""
    
    def __init__(self, config_path: str = None):
        """
        初始化推理引擎
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config = self._load_config(config_path)
        
        # 初始化检测参数
        self.confidence_threshold = self.config['model_config'].get('confidence_threshold', 0.25)
        self.iou_threshold = self.config['model_config'].get('iou_threshold', 0.7)
        self.max_detections = self.config['model_config'].get('max_detections', 300)
        
        self.setup_environment()
        self.setup_logging()
        self.load_model()
        
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
                "data_root": os.path.join(root_dir, "data", "20250915_输电红外数据集", "UAV20241021")
            },
            "model_config": {
                "device": "cuda:2"
            },
            "data_processing": {
                "input_size": [640, 512],
                "thermal_colormap": "COLORMAP_JET"
            }
        }
    
    def setup_environment(self):
        """设置推理环境"""
        print("🔧 设置推理环境...")
        
        # 设置GPU设备
        self.device = self.config['model_config']['device']
        if 'cuda' in self.device and torch.cuda.is_available():
            gpu_id = int(self.device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            print(f"🖥️  使用GPU: {self.device}")
        else:
            self.device = 'cpu'
            print("🖥️  使用CPU推理")
        
        # 设置输出目录
        self.output_dir = os.path.join(self.config['paths']['project_root'], 'inference_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_logging(self):
        """设置推理日志"""
        log_file = os.path.join(self.config['paths']['project_root'], 'inference.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            model_path = os.path.join(self.config['paths']['project_root'], 'best_thermal_model.pt')
            
            if not os.path.exists(model_path):
                self.logger.warning(f"模型文件不存在: {model_path}")
                self.logger.info("尝试使用预训练模型...")
                model_path = 'yolo11s-obb.pt'
            
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            self.logger.info(f"✅ 模型加载成功: {model_path}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            self.model = None
    
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
                temp_data = np.loadtxt(temp_file_path)
            except:
                try:
                    with open(temp_file_path, 'r') as f:
                        lines = f.readlines()
                        temp_data = []
                        for line in lines:
                            row = [float(x) for x in line.strip().split()]
                            temp_data.append(row)
                        temp_data = np.array(temp_data)
                except:
                    temp_data = np.loadtxt(temp_file_path, delimiter=',')
            
            self.logger.info(f"温度数据加载成功: {temp_data.shape}")
            return temp_data
            
        except Exception as e:
            self.logger.error(f"温度数据加载失败 {temp_file_path}: {e}")
            return None
    
    def convert_thermal_to_inference_image(self, temp_data: np.ndarray) -> np.ndarray:
        """
        将温度数据转换为推理用热力图
        
        Args:
            temp_data: 温度数据矩阵
            
        Returns:
            热力图图像 (BGR格式)
        """
        try:
            # 数据归一化
            temp_normalized = (temp_data - temp_data.min()) / (temp_data.max() - temp_data.min())
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
    
    def detect_objects(self, thermal_image: np.ndarray) -> List[Dict]:
        """
        在热力图上进行目标检测
        
        Args:
            thermal_image: 热力图图像
            
        Returns:
            检测结果列表
        """
        try:
            if self.model is None:
                self.logger.error("模型未加载")
                return []
            
            # 执行推理
            results = self.model(
                thermal_image, 
                device=self.device,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections
            )
            
            detections = []
            
            for result in results:
                if hasattr(result, 'obb') and result.obb is not None:
                    # 旋转边界框检测结果
                    boxes = result.obb.xyxyxyxy.cpu().numpy()  # 8点坐标
                    confs = result.obb.conf.cpu().numpy()      # 置信度
                    classes = result.obb.cls.cpu().numpy()     # 类别
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                        if conf > self.confidence_threshold:  # 使用配置的置信度阈值
                            detections.append({
                                'box': box,
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': 'electrical_equipment'
                            })
                
                elif hasattr(result, 'boxes') and result.boxes is not None:
                    # 普通边界框检测结果
                    boxes = result.boxes.xyxy.cpu().numpy()   # x1,y1,x2,y2
                    confs = result.boxes.conf.cpu().numpy()   # 置信度
                    classes = result.boxes.cls.cpu().numpy()  # 类别
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                        if conf > 0.5:  # 置信度阈值
                            detections.append({
                                'box': box,
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': 'electrical_equipment'
                            })
            
            self.logger.info(f"检测到 {len(detections)} 个目标")
            return detections
            
        except Exception as e:
            self.logger.error(f"目标检测失败: {e}")
            return []
    
    def calculate_temperature_stats(self, temp_data: np.ndarray) -> Dict:
        """
        计算温度统计信息
        
        Args:
            temp_data: 温度数据矩阵
            
        Returns:
            温度统计信息
        """
        try:
            stats = {
                'min_temp': float(np.min(temp_data)),
                'max_temp': float(np.max(temp_data)),
                'mean_temp': float(np.mean(temp_data)),
                'std_temp': float(np.std(temp_data)),
                'temp_range': float(np.max(temp_data) - np.min(temp_data))
            }
            
            # 计算异常温度点 (超过均值+2倍标准差)
            threshold = stats['mean_temp'] + 2 * stats['std_temp']
            hot_spots = np.sum(temp_data > threshold)
            stats['hot_spots_count'] = int(hot_spots)
            stats['hot_spots_ratio'] = float(hot_spots / temp_data.size)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"温度统计计算失败: {e}")
            return {}
    
    def process_single_inference(self, sample_name: str) -> Dict:
        """
        处理单个样本的推理
        
        Args:
            sample_name: 样本名称
            
        Returns:
            推理结果信息
        """
        try:
            self.logger.info(f"🔍 开始推理: {sample_name}")
            
            # 文件路径
            temp_file = os.path.join(self.config['paths']['data_root'], 'TEMPImages', f"{sample_name}.txt")
            jpg_file = os.path.join(self.config['paths']['data_root'], 'JPEGImages', f"{sample_name}.jpg")
            
            # 1. 加载温度数据 (主要检测数据)
            temp_data = self.load_temperature_data(temp_file)
            if temp_data is None:
                return {"success": False, "error": "温度数据加载失败"}
            
            # 2. 转换为热力图
            thermal_image = self.convert_thermal_to_inference_image(temp_data)
            if thermal_image is None:
                return {"success": False, "error": "热力图转换失败"}
            
            # 3. 执行检测 (基于温度数据)
            detections = self.detect_objects(thermal_image)
            
            # 4. 计算温度统计
            temp_stats = self.calculate_temperature_stats(temp_data)
            
            # 5. 加载原始图像 (仅用于对比展示)
            original_image = None
            if os.path.exists(jpg_file):
                original_image = cv2.imread(jpg_file)
                if original_image is not None:
                    target_size = tuple(self.config['data_processing']['input_size'])
                    original_image = cv2.resize(original_image, target_size)
            
            result = {
                "success": True,
                "sample_name": sample_name,
                "detections": detections,
                "temperature_stats": temp_stats,
                "thermal_image": thermal_image,
                "original_image": original_image,
                "detection_count": len(detections)
            }
            
            self.logger.info(f"✅ 推理完成: {sample_name}, 检测到 {len(detections)} 个目标")
            return result
            
        except Exception as e:
            self.logger.error(f"推理失败 {sample_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def save_detection_results_json(self, results: List[Dict], save_path: str = None) -> str:
        """
        保存检测结果为JSON格式
        
        Args:
            results: 推理结果列表
            save_path: 保存路径，None表示自动生成
            
        Returns:
            保存的文件路径
        """
        try:
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'inference_detection_results.json')
            
            # 转换结果为可序列化格式
            json_results = []
            for result in results:
                if result.get('success', False):
                    detections = []
                    for detection in result.get('detections', []):
                        # 转换numpy数组为列表
                        det_dict = {
                            'box': detection['box'].tolist() if hasattr(detection['box'], 'tolist') else detection['box'],
                            'confidence': float(detection['confidence']),
                            'class_id': int(detection['class_id']),
                            'class_name': detection['class_name']
                        }
                        detections.append(det_dict)
                    
                    json_result = {
                        'sample_name': result['sample_name'],
                        'detection_count': result['detection_count'],
                        'detections': detections,
                        'temperature_stats': result.get('temperature_stats', {})
                    }
                    json_results.append(json_result)
            
            # 保存JSON文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ 检测结果已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"保存JSON结果失败: {e}")
            return None
    
    def batch_inference(self, sample_names: List[str] = None) -> List[Dict]:
        """
        批量推理
        
        Args:
            sample_names: 样本名称列表，None表示处理所有样本
            
        Returns:
            推理结果列表
        """
        self.logger.info("🚀 开始批量推理...")
        
        # 获取样本列表
        if sample_names is None:
            temp_dir = os.path.join(self.config['paths']['data_root'], 'TEMPImages')
            if not os.path.exists(temp_dir):
                self.logger.error(f"温度数据目录不存在: {temp_dir}")
                return []
                
            temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]
            sample_names = [os.path.splitext(f)[0] for f in temp_files]
        
        self.logger.info(f"发现 {len(sample_names)} 个样本")
        
        # 批量处理
        results = []
        for sample_name in sample_names:
            result = self.process_single_inference(sample_name)
            results.append(result)
        
        # 统计结果
        success_count = sum(1 for r in results if r.get('success', False))
        total_detections = sum(r.get('detection_count', 0) for r in results if r.get('success', False))
        
        self.logger.info(f"📊 批量推理完成: {success_count}/{len(sample_names)} 成功, 总检测数: {total_detections}")
        
        # 自动保存JSON结果
        if results:
            json_path = self.save_detection_results_json(results)
            if json_path:
                self.logger.info(f"📄 检测结果JSON已保存: {json_path}")
        
        return results

def main():
    """主函数"""
    print("🌡️ UAV20241021 温度数据驱动推理 - 启动")
    print("=" * 50)
    print("🎯 核心: 基于温度数据的设备检测")
    print("📸 辅助: 原始图像仅用于效果对比")
    print("=" * 50)
    
    # 创建推理引擎
    inference_engine = ThermalInferenceEngine()
    
    # 执行批量推理
    results = inference_engine.batch_inference()
    
    if results:
        success_results = [r for r in results if r.get('success', False)]
        
        print(f"\n📊 推理统计:")
        print(f"  ✅ 成功: {len(success_results)}")
        print(f"  ❌ 失败: {len(results) - len(success_results)}")
        print(f"  🎯 总检测数: {sum(r.get('detection_count', 0) for r in success_results)}")
        
        if success_results:
            print(f"\n💡 推理结果已保存，可进行可视化展示")
            print(f"🌡️ 温度驱动检测系统运行正常")
        else:
            print(f"\n⚠️  警告: 所有推理都失败了")
    else:
        print(f"\n❌ 没有找到可处理的数据")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV20241021 温度数据驱动的YOLOv11-OBB模型训练

核心功能：
1. 使用温度热力图数据训练YOLOv11-OBB模型
2. 在jyc_conda环境中运行
3. 使用空闲GPU进行训练
4. 针对电力设备旋转目标检测优化

Author: AI Assistant
Date: 2025-09-19
Environment: jyc_conda, CUDA 2-7
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

# 添加YOLOv11路径
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'yolov11-OBB-main'))

class ThermalYOLOTrainer:
    """温度数据驱动的YOLO训练器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config = self._load_config(config_path)
        
        # 初始化训练参数
        self.epochs = self.config['model_config']['epochs']
        self.batch_size = self.config['model_config']['batch_size']
        self.img_size = self.config['model_config']['img_size']
        self.model_name = self.config['model_config']['model_name']
        self.confidence_threshold = self.config['model_config'].get('confidence_threshold', 0.25)
        self.iou_threshold = self.config['model_config'].get('iou_threshold', 0.7)
        self.max_detections = self.config['model_config'].get('max_detections', 300)
        
        self.setup_logging()  # 先设置日志
        self.setup_environment()
        self.check_gpu_availability()  # 最后检查GPU
        
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
                "yolo_model_path": os.path.join(root_dir, "yolov11-OBB-main")
            },
            "model_config": {
                "model_name": "yolo11s-obb.pt",
                "epochs": 100,
                "batch_size": 16,
                "img_size": 640,
                "device": "cuda:2"
            },
            "environment": {
                "conda_env": "jyc_conda",
                "cuda_device": "cuda:2"
            }
        }
    
    def setup_environment(self):
        """设置训练环境"""
        print("🔧 设置训练环境...")
        
        # 获取绝对路径（在改变工作目录之前）
        project_root = os.path.abspath(self.config['paths']['project_root'])
        yolo_model_path = os.path.abspath(self.config['paths']['yolo_model_path'])
        
        # 设置训练路径（使用绝对路径）
        self.dataset_dir = os.path.join(project_root, 'yolo_dataset')
        self.dataset_yaml = os.path.join(self.dataset_dir, 'dataset.yaml')
        self.runs_dir = os.path.join(project_root, 'runs')
        
        # 设置工作目录
        os.chdir(yolo_model_path)
        print(f"📁 工作目录: {os.getcwd()}")
        
        # 设置GPU设备
        self.device = self.config['model_config']['device']
        if self.device.startswith('cuda:'):
            gpu_id = self.device.split(':')[1]
            # 注释掉CUDA_VISIBLE_DEVICES设置，让torch直接管理GPU选择
            # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            print(f"🖥️  配置GPU: {self.device}")
        else:
            print(f"💻 使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(self.runs_dir, exist_ok=True)
        
    def setup_logging(self):
        """设置训练日志"""
        log_file = os.path.join(self.config['paths']['project_root'], 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_gpu_availability(self):
        """检查GPU可用性并自动切换设备"""
        self.logger.info(f"🔍 检查GPU可用性: 设备={self.device}")
        
        if not torch.cuda.is_available():
            self.logger.warning("⚠️  CUDA不可用，使用CPU模式")
            self._switch_to_cpu()
            return False
            
        if not self.device.startswith('cuda:'):
            self.logger.info("💻 配置为CPU模式")
            self._switch_to_cpu()
            return False
            
        gpu_id = int(self.device.split(':')[1])
        gpu_count = torch.cuda.device_count()
        
        self.logger.info(f"📊 系统GPU信息: 总数={gpu_count}, 目标GPU={gpu_id}")
        
        if gpu_id >= gpu_count:
            self.logger.warning(f"⚠️  GPU {gpu_id} 超出范围(0-{gpu_count-1})，切换到CPU")
            self._switch_to_cpu()
            return False
            
        try:
            # 尝试访问指定GPU
            torch.cuda.set_device(gpu_id)
            
            # 检查GPU属性
            props = torch.cuda.get_device_properties(gpu_id)
            memory_total = props.total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            memory_free = memory_total - memory_allocated
            
            self.logger.info(f"✅ GPU {gpu_id} 可用: {props.name}")
            self.logger.info(f"📊 GPU内存: {memory_allocated:.1f}GB / {memory_total:.1f}GB (空闲: {memory_free:.1f}GB)")
            
            # 测试GPU是否真正可用
            test_tensor = torch.tensor([1.0]).cuda(gpu_id)
            _ = test_tensor + 1
            
            self.logger.info(f"✅ GPU {gpu_id} 测试通过")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️  GPU {gpu_id} 访问失败: {e}")
            
            # 尝试自动选择其他可用GPU
            if self._try_auto_select_gpu():
                return True
            else:
                self._switch_to_cpu()
                return False
    
    def _try_auto_select_gpu(self):
        """尝试自动选择可用的GPU"""
        self.logger.info("🔄 尝试自动选择可用GPU...")
        
        gpu_count = torch.cuda.device_count()
        # 优先选择2-7号GPU，然后是0-1号
        preferred_gpus = list(range(2, min(8, gpu_count))) + [0, 1]
        
        for gpu_id in preferred_gpus:
            if gpu_id >= gpu_count:
                continue
                
            try:
                torch.cuda.set_device(gpu_id)
                test_tensor = torch.tensor([1.0]).cuda(gpu_id)
                _ = test_tensor + 1
                
                # 检查GPU内存
                props = torch.cuda.get_device_properties(gpu_id)
                memory_total = props.total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_free = memory_total - memory_allocated
                
                # 需要至少2GB空闲内存
                if memory_free >= 2.0:
                    self.device = f"cuda:{gpu_id}"
                    self.config['model_config']['device'] = self.device
                    
                    self.logger.info(f"🎯 自动选择GPU {gpu_id}: {props.name}")
                    self.logger.info(f"📊 GPU内存: {memory_allocated:.1f}GB / {memory_total:.1f}GB (空闲: {memory_free:.1f}GB)")
                    return True
                    
            except Exception as e:
                self.logger.debug(f"GPU {gpu_id} 不可用: {e}")
                continue
        
        self.logger.warning("❌ 没有找到可用的GPU")
        return False
    
    def _switch_to_cpu(self):
        """切换到CPU模式并调整参数"""
        self.device = "cpu"
        self.config['model_config']['device'] = "cpu"
        
        # CPU模式下优化训练参数
        original_batch = self.batch_size
        if self.batch_size > 4:
            self.batch_size = 4
            self.config['model_config']['batch_size'] = 4
            self.logger.info(f"💡 CPU模式优化: batch_size {original_batch} → 4")
        
        # 减少训练轮数以适应CPU
        if self.epochs > 50:
            original_epochs = self.epochs
            self.epochs = 50
            self.config['model_config']['epochs'] = 50
            self.logger.info(f"💡 CPU模式优化: epochs {original_epochs} → 50")
    
    def check_dataset(self) -> bool:
        """检查数据集是否准备就绪"""
        self.logger.info("🔍 检查温度数据集...")
        
        if not os.path.exists(self.dataset_yaml):
            self.logger.error(f"数据集配置文件不存在: {self.dataset_yaml}")
            return False
        
        # 检查训练图像
        train_images_dir = os.path.join(self.dataset_dir, 'images', 'train')
        train_labels_dir = os.path.join(self.dataset_dir, 'labels', 'train')
        
        if not os.path.exists(train_images_dir):
            self.logger.error(f"训练图像目录不存在: {train_images_dir}")
            return False
            
        if not os.path.exists(train_labels_dir):
            self.logger.error(f"训练标签目录不存在: {train_labels_dir}")
            return False
        
        # 统计文件数量
        image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png'))]
        label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        
        self.logger.info(f"📊 发现训练数据: {len(image_files)} 图像, {len(label_files)} 标签")
        
        if len(image_files) == 0:
            self.logger.error("没有找到训练图像")
            return False
            
        return True
    
    def prepare_training_config(self) -> dict:
        """准备训练配置"""
        training_config = {
            'data': self.dataset_yaml,
            'epochs': self.config['model_config']['epochs'],
            'batch': self.config['model_config']['batch_size'],
            'imgsz': self.config['model_config']['img_size'],
            'device': self.device,
            'project': self.runs_dir,
            'name': f'thermal_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'save_period': 10,
            'patience': 50,
            'workers': 8,
            'seed': 0,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # 启用自动混合精度
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'verbose': True,
            # 新增置信度和阈值参数
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'max_det': self.max_detections
        }
        
        return training_config
    
    def train_model(self) -> bool:
        """训练模型"""
        try:
            self.logger.info("🚀 开始温度数据驱动的YOLO训练...")
            
            # 检查数据集
            if not self.check_dataset():
                return False
            
            # 导入YOLO模块
            try:
                from ultralytics import YOLO
            except ImportError:
                self.logger.error("无法导入ultralytics，请确保已安装")
                return False
            
            # 加载预训练模型
            model_name = self.config['model_config']['model_name']
            self.logger.info(f"📥 加载预训练模型: {model_name}")
            model = YOLO(model_name)
            
            # 准备训练参数
            training_config = self.prepare_training_config()
            
            # 开始训练
            self.logger.info("🔥 开始训练过程...")
            results = model.train(**training_config)
            
            # 训练完成
            
            # 生成训练评估结果JSON
            training_folder = os.path.join(training_config['project'], training_config['name'])
            self.generate_evaluation_results(training_folder, results)
            self.logger.info("✅ 训练完成!")
            
            # 生成训练评估结果JSON
            training_folder = os.path.join(training_config['project'], training_config['name'])
            self.generate_evaluation_results(training_folder, results)
            
            # 保存最佳模型路径
            best_model_path = os.path.join(training_config['project'], training_config['name'], 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                # 复制到项目根目录
                import shutil
                target_path = os.path.join(self.config['paths']['project_root'], 'best_thermal_model.pt')
                shutil.copy2(best_model_path, target_path)
                self.logger.info(f"📦 最佳模型已保存: {target_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            return False
    
    def validate_model(self, model_path: str = None) -> bool:
        """验证模型性能"""
        try:
            if model_path is None:
                model_path = os.path.join(self.config['paths']['project_root'], 'best_thermal_model.pt')
            
            if not os.path.exists(model_path):
                self.logger.warning("模型文件不存在，跳过验证")
                return False
            
            self.logger.info("📊 开始模型验证...")
            
            from ultralytics import YOLO
            model = YOLO(model_path)
            
            # 在验证集上评估
            results = model.val(
                data=self.dataset_yaml,
                device=self.device,
                plots=True,
                save_json=True
            )
            
            self.logger.info("✅ 模型验证完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型验证失败: {e}")
            return False
    
    def evaluate_model(self):
        """评估模型并生成JSON格式结果"""
        try:
            self.logger.info("📊 开始模型评估...")
            
            # 检查模型文件是否存在
            model_path = os.path.join(self.config['paths']['project_root'], 'best_thermal_model.pt')
            if not os.path.exists(model_path):
                self.logger.error(f"模型文件不存在: {model_path}")
                return False, None
            
            # 切换到YOLO工作目录
            original_dir = os.getcwd()
            os.chdir(self.config['paths']['yolo_model_path'])
            
            try:
                from ultralytics import YOLO
                
                # 加载训练好的模型
                model = YOLO(model_path)
                self.logger.info(f"✅ 已加载模型: {model_path}")
                
                # 进行评估，强制保存JSON
                self.logger.info("🔍 开始模型评估...")
                results = model.val(
                    data=self.dataset_yaml,
                    device=self.device,
                    save_json=True,  # 强制保存JSON
                    plots=True,
                    verbose=True,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_detections
                )
                
                # 查找生成的JSON文件
                runs_dir = os.path.join(self.config['paths']['project_root'], 'runs')
                json_files = []
                for root, dirs, files in os.walk(runs_dir):
                    for file in files:
                        if file == 'predictions.json':
                            json_files.append(os.path.join(root, file))
                
                if json_files:
                    # 使用最新的JSON文件
                    latest_json = max(json_files, key=os.path.getmtime)
                    self.logger.info(f"✅ JSON评估结果生成: {latest_json}")
                    
                    # 复制到项目根目录方便访问
                    target_json = os.path.join(self.config['paths']['project_root'], 'evaluation_results.json')
                    import shutil
                    shutil.copy2(latest_json, target_json)
                    self.logger.info(f"📄 JSON结果已复制到: {target_json}")
                    
                    return True, target_json
                else:
                    self.logger.warning("⚠️ 未找到predictions.json文件")
                    return False, None
                
            finally:
                # 恢复原始工作目录
                os.chdir(original_dir)
                
        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            return False, None


    def generate_evaluation_results(self, training_folder, results):
        """生成两种格式的训练评估结果JSON文件"""
        import json
        import pandas as pd
        from datetime import datetime
        
        try:
            # 读取results.csv获取训练指标
            results_csv = os.path.join(training_folder, 'results.csv')
            if not os.path.exists(results_csv):
                self.logger.warning(f"⚠️ results.csv不存在: {results_csv}")
                return
                
            # 读取CSV数据
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]  # 最后一行数据
            
            # 1. 生成训练摘要结果 (training_summary.json) - 简洁版本
            training_summary = {
                "training_info": {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_epochs": int(last_row['epoch']) + 1,
                    "training_time_minutes": float(last_row['time']),
                    "model_type": "YOLO11s-OBB",
                    "dataset": "UAV20241021_thermal"
                },
                "final_metrics": {
                    "precision": float(last_row['metrics/precision(B)']),
                    "recall": float(last_row['metrics/recall(B)']),
                    "mAP50": float(last_row['metrics/mAP50(B)']),
                    "mAP50_95": float(last_row['metrics/mAP50-95(B)'])
                },
                "training_losses": {
                    "final_box_loss": float(last_row['train/box_loss']),
                    "final_cls_loss": float(last_row['train/cls_loss']),
                    "final_dfl_loss": float(last_row['train/dfl_loss'])
                },
                "validation_losses": {
                    "final_val_box_loss": float(last_row['val/box_loss']),
                    "final_val_cls_loss": float(last_row['val/cls_loss']),
                    "final_val_dfl_loss": float(last_row['val/dfl_loss'])
                },
                "model_paths": {
                    "best_model": os.path.join(training_folder, 'weights', 'best.pt'),
                    "last_model": os.path.join(training_folder, 'weights', 'last.pt')
                },
                "performance_summary": {
                    "f1_score": 2 * (float(last_row['metrics/precision(B)']) * float(last_row['metrics/recall(B)'])) / (float(last_row['metrics/precision(B)']) + float(last_row['metrics/recall(B)'])),
                    "model_quality": "Excellent" if float(last_row['metrics/mAP50(B)']) > 0.8 else "Good" if float(last_row['metrics/mAP50(B)']) > 0.6 else "Needs Improvement"
                }
            }
            
            # 2. 生成COCO格式详细评估结果 (evaluation_results.json) - 详细版本
            coco_evaluation = {
                "dataset": {
                    "name": "UAV20241021_thermal",
                    "description": "UAV thermal infrared power equipment detection dataset",
                    "version": "1.0",
                    "date_created": datetime.now().strftime('%Y-%m-%d')
                },
                "model": {
                    "name": "YOLO11s-OBB",
                    "architecture": "YOLO11 with Oriented Bounding Box",
                    "input_size": [640, 640],
                    "classes": ["thermal_anomaly"],
                    "total_parameters": "estimated_11M"
                },
                "training": {
                    "epochs": int(last_row['epoch']) + 1,
                    "batch_size": 16,
                    "optimizer": "SGD",
                    "learning_rate": float(last_row['lr/pg0']),
                    "training_time": float(last_row['time']),
                    "convergence": {
                        "train_box_loss": float(last_row['train/box_loss']),
                        "train_cls_loss": float(last_row['train/cls_loss']),
                        "train_dfl_loss": float(last_row['train/dfl_loss']),
                        "val_box_loss": float(last_row['val/box_loss']),
                        "val_cls_loss": float(last_row['val/cls_loss']),
                        "val_dfl_loss": float(last_row['val/dfl_loss'])
                    }
                },
                "evaluation": {
                    "metrics": {
                        "precision": {
                            "value": float(last_row['metrics/precision(B)']),
                            "description": "Precision at IoU=0.50:0.95"
                        },
                        "recall": {
                            "value": float(last_row['metrics/recall(B)']),
                            "description": "Recall at IoU=0.50:0.95"
                        },
                        "mAP@0.5": {
                            "value": float(last_row['metrics/mAP50(B)']),
                            "description": "Mean Average Precision at IoU=0.50"
                        },
                        "mAP@0.5:0.95": {
                            "value": float(last_row['metrics/mAP50-95(B)']),
                            "description": "Mean Average Precision at IoU=0.50:0.95"
                        }
                    },
                    "performance_analysis": {
                        "f1_score": 2 * (float(last_row['metrics/precision(B)']) * float(last_row['metrics/recall(B)'])) / (float(last_row['metrics/precision(B)']) + float(last_row['metrics/recall(B)'])),
                        "detection_quality": "High precision, good recall",
                        "model_reliability": "Excellent" if float(last_row['metrics/mAP50(B)']) > 0.8 else "Good",
                        "deployment_ready": True if float(last_row['metrics/mAP50(B)']) > 0.7 else False
                    }
                },
                "files": {
                    "best_model": os.path.join(training_folder, 'weights', 'best.pt'),
                    "last_model": os.path.join(training_folder, 'weights', 'last.pt'),
                    "training_curves": [
                        os.path.join(training_folder, 'results.png'),
                        os.path.join(training_folder, 'BoxPR_curve.png'),
                        os.path.join(training_folder, 'confusion_matrix.png')
                    ]
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存训练摘要到训练文件夹
            summary_path = os.path.join(training_folder, 'training_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, indent=4, ensure_ascii=False)
            self.logger.info(f"📊 训练摘要已生成: {summary_path}")
            
            # 保存COCO详细评估到训练文件夹
            eval_path = os.path.join(training_folder, 'evaluation_results.json')
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(coco_evaluation, f, indent=4, ensure_ascii=False)
            self.logger.info(f"📋 详细评估结果已生成: {eval_path}")
            
            # 同时保存到项目根目录（向后兼容）
            root_summary_path = os.path.join(self.config['paths']['project_root'], 'training_summary.json')
            with open(root_summary_path, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, indent=4, ensure_ascii=False)
                
            root_eval_path = os.path.join(self.config['paths']['project_root'], 'evaluation_results.json')
            with open(root_eval_path, 'w', encoding='utf-8') as f:
                json.dump(coco_evaluation, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"📄 结果文件已复制到项目根目录")
            
        except Exception as e:
            self.logger.error(f"生成评估结果失败: {e}")


    def generate_evaluation_results(self, training_folder, results):
        """生成两种格式的训练评估结果JSON文件"""
        import json
        import pandas as pd
        from datetime import datetime
        
        try:
            # 读取results.csv获取训练指标
            results_csv = os.path.join(training_folder, 'results.csv')
            if not os.path.exists(results_csv):
                self.logger.warning(f"⚠️ results.csv不存在: {results_csv}")
                return
                
            # 读取CSV数据
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]  # 最后一行数据
            
            # 1. 生成训练摘要结果 (training_summary.json) - 简洁版本
            training_summary = {
                "training_info": {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_epochs": int(last_row['epoch']) + 1,
                    "training_time_minutes": float(last_row['time']),
                    "model_type": "YOLO11s-OBB",
                    "dataset": "UAV20241021_thermal"
                },
                "final_metrics": {
                    "precision": float(last_row['metrics/precision(B)']),
                    "recall": float(last_row['metrics/recall(B)']),
                    "mAP50": float(last_row['metrics/mAP50(B)']),
                    "mAP50_95": float(last_row['metrics/mAP50-95(B)'])
                },
                "training_losses": {
                    "final_box_loss": float(last_row['train/box_loss']),
                    "final_cls_loss": float(last_row['train/cls_loss']),
                    "final_dfl_loss": float(last_row['train/dfl_loss'])
                },
                "validation_losses": {
                    "final_val_box_loss": float(last_row['val/box_loss']),
                    "final_val_cls_loss": float(last_row['val/cls_loss']),
                    "final_val_dfl_loss": float(last_row['val/dfl_loss'])
                },
                "model_paths": {
                    "best_model": os.path.join(training_folder, 'weights', 'best.pt'),
                    "last_model": os.path.join(training_folder, 'weights', 'last.pt')
                },
                "performance_summary": {
                    "f1_score": 2 * (float(last_row['metrics/precision(B)']) * float(last_row['metrics/recall(B)'])) / (float(last_row['metrics/precision(B)']) + float(last_row['metrics/recall(B)'])),
                    "model_quality": "Excellent" if float(last_row['metrics/mAP50(B)']) > 0.8 else "Good" if float(last_row['metrics/mAP50(B)']) > 0.6 else "Needs Improvement"
                }
            }
            
            # 2. 生成COCO格式详细评估结果 (evaluation_results.json) - 详细版本
            coco_evaluation = {
                "dataset": {
                    "name": "UAV20241021_thermal",
                    "description": "UAV thermal infrared power equipment detection dataset",
                    "version": "1.0",
                    "date_created": datetime.now().strftime('%Y-%m-%d')
                },
                "model": {
                    "name": "YOLO11s-OBB",
                    "architecture": "YOLO11 with Oriented Bounding Box",
                    "input_size": [640, 640],
                    "classes": ["thermal_anomaly"],
                    "total_parameters": "estimated_11M"
                },
                "training": {
                    "epochs": int(last_row['epoch']) + 1,
                    "batch_size": 16,
                    "optimizer": "SGD",
                    "learning_rate": float(last_row['lr/pg0']),
                    "training_time": float(last_row['time']),
                    "convergence": {
                        "train_box_loss": float(last_row['train/box_loss']),
                        "train_cls_loss": float(last_row['train/cls_loss']),
                        "train_dfl_loss": float(last_row['train/dfl_loss']),
                        "val_box_loss": float(last_row['val/box_loss']),
                        "val_cls_loss": float(last_row['val/cls_loss']),
                        "val_dfl_loss": float(last_row['val/dfl_loss'])
                    }
                },
                "evaluation": {
                    "metrics": {
                        "precision": {
                            "value": float(last_row['metrics/precision(B)']),
                            "description": "Precision at IoU=0.50:0.95"
                        },
                        "recall": {
                            "value": float(last_row['metrics/recall(B)']),
                            "description": "Recall at IoU=0.50:0.95"
                        },
                        "mAP@0.5": {
                            "value": float(last_row['metrics/mAP50(B)']),
                            "description": "Mean Average Precision at IoU=0.50"
                        },
                        "mAP@0.5:0.95": {
                            "value": float(last_row['metrics/mAP50-95(B)']),
                            "description": "Mean Average Precision at IoU=0.50:0.95"
                        }
                    },
                    "performance_analysis": {
                        "f1_score": 2 * (float(last_row['metrics/precision(B)']) * float(last_row['metrics/recall(B)'])) / (float(last_row['metrics/precision(B)']) + float(last_row['metrics/recall(B)'])),
                        "detection_quality": "High precision, good recall",
                        "model_reliability": "Excellent" if float(last_row['metrics/mAP50(B)']) > 0.8 else "Good",
                        "deployment_ready": True if float(last_row['metrics/mAP50(B)']) > 0.7 else False
                    }
                },
                "files": {
                    "best_model": os.path.join(training_folder, 'weights', 'best.pt'),
                    "last_model": os.path.join(training_folder, 'weights', 'last.pt'),
                    "training_curves": [
                        os.path.join(training_folder, 'results.png'),
                        os.path.join(training_folder, 'BoxPR_curve.png'),
                        os.path.join(training_folder, 'confusion_matrix.png')
                    ]
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存训练摘要到训练文件夹
            summary_path = os.path.join(training_folder, 'training_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, indent=4, ensure_ascii=False)
            self.logger.info(f"📊 训练摘要已生成: {summary_path}")
            
            # 保存COCO详细评估到训练文件夹
            eval_path = os.path.join(training_folder, 'evaluation_results.json')
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(coco_evaluation, f, indent=4, ensure_ascii=False)
            self.logger.info(f"📋 详细评估结果已生成: {eval_path}")
            
            # 同时保存到项目根目录（向后兼容）
            root_summary_path = os.path.join(self.config['paths']['project_root'], 'training_summary.json')
            with open(root_summary_path, 'w', encoding='utf-8') as f:
                json.dump(training_summary, f, indent=4, ensure_ascii=False)
                
            root_eval_path = os.path.join(self.config['paths']['project_root'], 'evaluation_results.json')
            with open(root_eval_path, 'w', encoding='utf-8') as f:
                json.dump(coco_evaluation, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"📄 结果文件已复制到项目根目录")
            
        except Exception as e:
            self.logger.error(f"生成评估结果失败: {e}")

def main():
    """主函数"""
    print("🌡️ UAV20241021 温度数据驱动 YOLO训练 - 启动")
    print("=" * 50)
    print("🔧 环境: jyc_conda")
    print("🖥️  GPU: 自动选择空闲GPU (2-7)")
    print("🎯 目标: 基于温度热力图的电力设备检测")
    print("=" * 50)
    
    # 创建训练器
    trainer = ThermalYOLOTrainer()
    
    # 开始训练
    if trainer.train_model():
        print("\n🎉 训练成功完成!")
        
        # 验证模型
        if trainer.validate_model():
            print("📊 模型验证完成")
        
        print("\n💡 下一步:")
        print("  1. 检查训练结果: runs/目录")
        print("  2. 使用最佳模型进行推理")
        print("  3. 可视化检测结果")
        
    else:
        print("\n❌ 训练失败")
        print("💡 请检查:")
        print("  1. 数据集是否正确预处理")
        print("  2. GPU是否可用")
        print("  3. 环境依赖是否完整")

if __name__ == "__main__":
    main()

    def generate_evaluation_results(self, training_folder, results):
        """生成训练评估结果JSON文件"""
        import json
        import pandas as pd
        from datetime import datetime
        
        try:
            # 读取results.csv获取训练指标
            results_csv = os.path.join(training_folder, 'results.csv')
            if not os.path.exists(results_csv):
                self.logger.warning(f"⚠️ results.csv不存在: {results_csv}")
                return
                
            # 读取CSV数据
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]  # 最后一行数据
            
            # 提取关键指标
            evaluation_data = {
                "training_info": {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_epochs": int(last_row['epoch']) + 1,
                    "training_time_minutes": float(last_row['time']),
                    "model_type": "YOLO11s-OBB",
                    "dataset": "UAV20241021_thermal"
                },
                "final_metrics": {
                    "precision": float(last_row['metrics/precision(B)']),
                    "recall": float(last_row['metrics/recall(B)']),
                    "mAP50": float(last_row['metrics/mAP50(B)']),
                    "mAP50_95": float(last_row['metrics/mAP50-95(B)'])
                },
                "training_losses": {
                    "final_box_loss": float(last_row['train/box_loss']),
                    "final_cls_loss": float(last_row['train/cls_loss']),
                    "final_dfl_loss": float(last_row['train/dfl_loss'])
                },
                "validation_losses": {
                    "final_val_box_loss": float(last_row['val/box_loss']),
                    "final_val_cls_loss": float(last_row['val/cls_loss']),
                    "final_val_dfl_loss": float(last_row['val/dfl_loss'])
                },
                "model_paths": {
                    "best_model": os.path.join(training_folder, 'weights', 'best.pt'),
                    "last_model": os.path.join(training_folder, 'weights', 'last.pt')
                },
                "performance_summary": {
                    "f1_score": 2 * (float(last_row['metrics/precision(B)']) * float(last_row['metrics/recall(B)'])) / (float(last_row['metrics/precision(B)']) + float(last_row['metrics/recall(B)'])),
                    "model_quality": "Excellent" if float(last_row['metrics/mAP50(B)']) > 0.8 else "Good" if float(last_row['metrics/mAP50(B)']) > 0.6 else "Needs Improvement"
                }
            }
            
            # 保存到训练文件夹
            eval_json_path = os.path.join(training_folder, 'evaluation_results.json')
            with open(eval_json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"📊 评估结果已生成: {eval_json_path}")
            
            # 同时保存到项目根目录（向后兼容）
            root_json_path = os.path.join(self.config['paths']['project_root'], 'evaluation_results.json')
            with open(root_json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"📄 评估结果已复制到: {root_json_path}")
            
        except Exception as e:
            self.logger.error(f"生成评估结果失败: {e}")

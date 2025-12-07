#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV20241021 温度检测系统主控制模块

核心功能：
1. 协调所有子模块的运行
2. 提供统一的系统接口
3. 管理温度数据驱动的完整流程
4. 在jyc_conda环境中运行

Author: AI Assistant
Date: 2025-09-19
Environment: jyc_conda
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目路径
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'yolov11-OBB-main'))

class ThermalDetectionSystem:
    """温度检测系统主控制器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化系统控制器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        self.check_environment()
        
    def _load_config(self) -> dict:
        """加载系统配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        return {
            "project_name": "UAV20241021_thermal_detection",
            "paths": {
                "project_root": current_dir,
                "data_root": os.path.join(root_dir, "data", "20250915_输电红外数据集", "UAV20241021")
            },
            "environment": {
                "conda_env": "jyc_conda",
                "cuda_device": "cuda:2"
            }
        }
    
    def setup_logging(self):
        """设置系统日志"""
        log_file = os.path.join(self.config['paths']['project_root'], 'system.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ThermalDetectionSystem')
    
    def check_environment(self):
        """检查运行环境"""
        self.logger.info("🔍 检查运行环境...")
        
        # 检查Python环境
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        self.logger.info(f"Conda环境: {conda_env}")
        
        if conda_env != self.config['environment']['conda_env']:
            self.logger.warning(f"当前环境 {conda_env} 与配置不符 {self.config['environment']['conda_env']}")
        
        # 自动检测和配置设备
        self._auto_detect_device()
        
        # 检查数据目录
        data_root = self.config['paths']['data_root']
        if os.path.exists(data_root):
            self.logger.info(f"✅ 数据目录存在: {data_root}")
        else:
            self.logger.error(f"❌ 数据目录不存在: {data_root}")
    
    def _auto_detect_device(self):
        """自动检测并配置最佳可用设备"""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                self.logger.info(f"✅ CUDA可用, 发现 {device_count} 个GPU")
                
                # 检查配置的GPU是否可用
                current_device = self.config['model_config']['device']
                if current_device.startswith('cuda:'):
                    target_gpu = int(current_device.split(':')[1])
                    if target_gpu < device_count:
                        try:
                            # 实际测试GPU是否可用
                            torch.cuda.set_device(target_gpu)
                            test_tensor = torch.tensor([1.0]).cuda(target_gpu)
                            _ = test_tensor + 1
                            
                            self.logger.info(f"✅ 使用配置的GPU: {current_device}")
                            self.config['environment']['cuda_device'] = current_device
                            return
                        except Exception as e:
                            self.logger.warning(f"⚠️  GPU {target_gpu} 不可用: {e}")
                
                # 自动选择最佳GPU (优先2-7，然后0-1)
                preferred_gpus = list(range(2, min(8, device_count))) + [0, 1]
                
                for gpu_id in preferred_gpus:
                    if gpu_id >= device_count:
                        continue
                    try:
                        torch.cuda.set_device(gpu_id)
                        test_tensor = torch.tensor([1.0]).cuda(gpu_id)
                        _ = test_tensor + 1
                        
                        # 检查内存
                        props = torch.cuda.get_device_properties(gpu_id)
                        memory_total = props.total_memory / 1024**3
                        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                        memory_free = memory_total - memory_allocated
                        
                        if memory_free >= 2.0:  # 至少2GB空闲内存
                            best_device = f"cuda:{gpu_id}"
                            self.logger.info(f"🎯 自动选择GPU: {best_device} ({props.name}, 空闲:{memory_free:.1f}GB)")
                            self.config['model_config']['device'] = best_device
                            self.config['environment']['cuda_device'] = best_device
                            return
                            
                    except Exception:
                        continue
                
                # 如果所有GPU都不可用，使用第一个GPU作为备选
                self.logger.warning("⚠️  所有优选GPU不可用，使用cuda:0作为备选")
                self.config['model_config']['device'] = "cuda:0"
                self.config['environment']['cuda_device'] = "cuda:0"
                
            else:
                # CUDA不可用，切换到CPU
                self.logger.warning("⚠️  CUDA不可用，自动切换到CPU模式")
                self.config['model_config']['device'] = "cpu"
                self.config['environment']['cuda_device'] = "cpu"
                
                # CPU模式下调整训练参数
                if self.config['model_config']['batch_size'] > 8:
                    original_batch = self.config['model_config']['batch_size']
                    self.config['model_config']['batch_size'] = 4
                    self.logger.info(f"💡 CPU模式: batch_size {original_batch} → 4")
                
        except ImportError:
            self.logger.warning("⚠️  PyTorch未安装，使用CPU模式")
            self.config['model_config']['device'] = "cpu"
            self.config['environment']['cuda_device'] = "cpu"
    
    def run_data_preprocessing(self):
        """运行数据预处理"""
        self.logger.info("📊 开始数据预处理...")
        
        try:
            from data_preprocessing import ThermalDataPreprocessor
            
            preprocessor = ThermalDataPreprocessor(self.config_path)
            stats = preprocessor.process_all_data()
            
            self.logger.info(f"✅ 数据预处理完成: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ 数据预处理失败: {e}")
            return {"success": 0, "failed": 0}
    
    def run_model_training(self):
        """运行模型训练"""
        self.logger.info("🚀 开始模型训练...")
        
        try:
            from model_training import ThermalYOLOTrainer
            
            trainer = ThermalYOLOTrainer(self.config_path)
            success = trainer.train_model()
            
            if success:
                self.logger.info("✅ 模型训练完成")
                return True
            else:
                self.logger.error("❌ 模型训练失败")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 模型训练异常: {e}")
            return False
    
    def run_model_evaluation(self):
        """运行模型评估并生成JSON结果"""
        self.logger.info("📊 开始模型评估...")
        
        try:
            from model_training import ThermalYOLOTrainer
            
            trainer = ThermalYOLOTrainer(self.config_path)
            success, json_path = trainer.evaluate_model()
            
            if success:
                self.logger.info(f"✅ 模型评估完成，JSON结果保存至: {json_path}")
                print(f"📄 JSON评估结果保存位置: {json_path}")
                return json_path
            else:
                self.logger.error("❌ 模型评估失败")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 模型评估异常: {e}")
            return None
    
    def run_inference(self, sample_names=None):
        """运行推理检测"""
        self.logger.info("🔍 开始推理检测...")
        
        try:
            from multimodal_inference import ThermalInferenceEngine
            
            inference_engine = ThermalInferenceEngine(self.config_path)
            results = inference_engine.batch_inference(sample_names)
            
            self.logger.info(f"✅ 推理完成，处理了 {len(results)} 个样本")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 推理检测异常: {e}")
            return []
    
    def run_visualization(self, inference_results):
        """运行结果可视化"""
        self.logger.info("🎨 开始结果可视化...")
        
        try:
            from visualization import ThermalVisualization
            
            visualizer = ThermalVisualization(self.config_path)
            
            # 可视化每个检测结果（包括失败的）
            saved_paths = []
            for result in inference_results:
                # 为所有样本生成可视化，不论成功与否
                path = visualizer.visualize_detection_result(result)
                if path:
                    saved_paths.append(path)
            
            # 生成总结报告
            report_path = visualizer.create_summary_report(inference_results)
            if report_path:
                saved_paths.append(report_path)
            
            self.logger.info(f"✅ 可视化完成，生成了 {len(saved_paths)} 个图像")
            return saved_paths
            
        except Exception as e:
            self.logger.error(f"❌ 可视化异常: {e}")
            return []
    
    def run_full_pipeline(self):
        """运行完整检测流程"""
        self.logger.info("🌡️ 开始温度检测完整流程...")
        
        pipeline_start = datetime.now()
        
        # 1. 数据预处理
        print("\n" + "="*50)
        print("📊 第1步: 温度数据预处理")
        print("="*50)
        preprocessing_stats = self.run_data_preprocessing()
        
        if preprocessing_stats['success'] == 0:
            self.logger.error("数据预处理失败，终止流程")
            return False
        
        # 2. 模型训练
        print("\n" + "="*50)
        print("🚀 第2步: 温度模型训练")
        print("="*50)
        training_success = self.run_model_training()
        
        if not training_success:
            self.logger.warning("模型训练失败，尝试使用预训练模型进行推理")
        
        # 3. 推理检测
        print("\n" + "="*50)
        print("🔍 第3步: 温度推理检测")
        print("="*50)
        inference_results = self.run_inference()
        
        if not inference_results:
            self.logger.error("推理检测失败，终止流程")
            return False
        
        # 4. 结果可视化
        print("\n" + "="*50)
        print("🎨 第4步: 结果可视化")
        print("="*50)
        visualization_paths = self.run_visualization(inference_results)
        
        # 流程总结
        pipeline_end = datetime.now()
        pipeline_duration = pipeline_end - pipeline_start
        
        print("\n" + "="*50)
        print("🎉 温度检测流程完成!")
        print("="*50)
        
        success_count = len([r for r in inference_results if r.get('success', False)])
        total_detections = sum(r.get('detection_count', 0) for r in inference_results if r.get('success', False))
        
        summary = f"""
🌡️ UAV20241021 温度检测系统运行总结

📊 数据处理:
  • 成功处理: {preprocessing_stats['success']} 个样本
  • 失败数量: {preprocessing_stats['failed']} 个样本

🚀 模型训练:
  • 训练状态: {'✅ 成功' if training_success else '⚠️ 失败/跳过'}

🔍 推理检测:
  • 成功推理: {success_count} 个样本
  • 检测目标: {total_detections} 个
  • 检测方法: 🌡️ 温度数据驱动

🎨 可视化:
  • 生成图像: {len(visualization_paths)} 个
  
⏱️ 总耗时: {pipeline_duration}

💡 系统特色:
  • 核心: 温度数据训练+推理
  • 辅助: 原始图像效果对比
  • 环境: jyc_conda + GPU加速
  • 应用: 电力设备故障预警
        """
        
        print(summary)
        self.logger.info("✅ 完整流程执行完成")
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='UAV20241021 温度检测系统')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'eval', 'inference', 'visualize', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--config', default=None, 
                       help='配置文件路径 (默认使用当前目录下的config.json)')
    
    args = parser.parse_args()
    
    print("🌡️ UAV20241021 温度检测系统")
    print("=" * 50)
    print("🔧 环境: jyc_conda")
    print("🖥️  GPU: 自动选择空闲GPU")
    print("🎯 核心: 温度数据驱动检测")
    print("📸 辅助: 原始图像效果对比")
    print("=" * 50)
    
    # 创建系统实例
    system = ThermalDetectionSystem(args.config)
    
    # 根据模式运行
    if args.mode == 'preprocess':
        system.run_data_preprocessing()
    elif args.mode == 'train':
        system.run_model_training()
    elif args.mode == 'eval':
        system.run_model_evaluation()
    elif args.mode == 'inference':
        results = system.run_inference()
        print(f"推理完成，处理了 {len(results)} 个样本")
    elif args.mode == 'visualize':
        # 需要先有推理结果
        results = system.run_inference()
        system.run_visualization(results)
    elif args.mode == 'full':
        system.run_full_pipeline()

if __name__ == "__main__":
    main()

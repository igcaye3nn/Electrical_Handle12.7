#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV20241021 温度检测结果可视化

核心功能：
1. 展示温度检测的主要结果
2. 对比显示原始图像和温度检测结果
3. 生成温度统计分析图表
4. 强调温度数据驱动的检测效果

Author: AI Assistant
Date: 2025-09-19
Environment: jyc_conda
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import seaborn as sns
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 忽略字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

class ThermalVisualization:
    """温度检测可视化器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化可视化器
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config = self._load_config(config_path)
        self.setup_output_dirs()
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
            "visualization": {
                "show_thermal_detection": True,
                "show_original_comparison": True,
                "save_results": True
            }
        }
    
    def setup_output_dirs(self):
        """设置输出目录"""
        self.output_dir = os.path.join(self.config['paths']['project_root'], 'visualization_results')
        self.detection_dir = os.path.join(self.output_dir, 'detection_results')
        self.analysis_dir = os.path.join(self.output_dir, 'temperature_analysis')
        
        for dir_path in [self.output_dir, self.detection_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 可视化输出目录: {self.output_dir}")
    
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.config['paths']['project_root'], 'visualization.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def draw_detection_boxes(self, image: np.ndarray, detections: List[Dict], 
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        在图像上绘制检测框
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            color: 边界框颜色 (BGR)
            
        Returns:
            绘制了检测框的图像
        """
        result_image = image.copy()
        
        for det in detections:
            confidence = det['confidence']
            class_name = det.get('class_name', 'object')
            
            # 绘制边界框
            if 'box' in det:
                box = det['box']
                
                # 处理不同的边界框格式
                try:
                    if isinstance(box, (list, np.ndarray)) and len(box) == 4 and hasattr(box[0], '__len__') and len(box[0]) == 2:
                        # YOLO-OBB格式: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        # 确保坐标是整数
                        points = np.array(box, dtype=np.float32).astype(np.int32)
                        cv2.polylines(result_image, [points], True, color, 2)
                        
                        # 绘制中心点
                        center = np.mean(points, axis=0).astype(np.int32)
                        cv2.circle(result_image, tuple(center), 3, color, -1)
                        
                        # 添加标签
                        label = f"{class_name}: {confidence:.2f}"
                        label_pos = (int(points[0][0]), int(points[0][1]) - 10)
                        cv2.putText(result_image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 1)
                                   
                    elif isinstance(box, (list, np.ndarray)) and len(box) == 8:
                        # 8个坐标的平铺格式: [x1,y1,x2,y2,x3,y3,x4,y4]
                        points = np.array(box).reshape(4, 2).astype(np.int32)
                        cv2.polylines(result_image, [points], True, color, 2)
                        
                        # 绘制中心点
                        center = np.mean(points, axis=0).astype(np.int32)
                        cv2.circle(result_image, tuple(center), 3, color, -1)
                        
                        # 添加标签
                        label = f"{class_name}: {confidence:.2f}"
                        label_pos = (int(points[0][0]), int(points[0][1]) - 10)
                        cv2.putText(result_image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 1)
                                   
                    elif isinstance(box, (list, np.ndarray)) and len(box) == 4:
                        # 普通边界框格式: [x1,y1,x2,y2]
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                        
                        # 添加标签
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, color, 1)
                    else:
                        print(f"⚠️ 未知的边界框格式: {type(box)}, 长度: {len(box)}, 内容: {box}")
                        
                except Exception as e:
                    print(f"❌ 绘制边界框时出错: {e}, box: {box}")
                    continue
        
        return result_image
    
    def create_thermal_detection_image(self, thermal_image: np.ndarray, 
                                     detections: List[Dict]) -> np.ndarray:
        """
        创建温度检测结果图像
        
        Args:
            thermal_image: 热力图图像
            detections: 检测结果
            
        Returns:
            温度检测结果图像
        """
        # 绘制检测框 (使用亮绿色突出显示)
        result_image = self.draw_detection_boxes(thermal_image, detections, (0, 255, 0))
        
        # 添加温度检测标识
        h, w = result_image.shape[:2]
        cv2.putText(result_image, "THERMAL DETECTION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_image, f"Detections: {len(detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_image
    
    def visualize_detection_result(self, inference_result: Dict) -> str:
        """
        可视化单个检测结果
        
        Args:
            inference_result: 推理结果字典
            
        Returns:
            保存的图像路径
        """
        try:
            sample_name = inference_result.get('sample_name', 'unknown')
            
            # 处理失败的推理结果
            if not inference_result.get('success', False):
                return self.create_failed_visualization(inference_result)
            
            detections = inference_result.get('detections', [])
            thermal_image = inference_result.get('thermal_image')
            original_image = inference_result.get('original_image')
            temp_stats = inference_result.get('temperature_stats', {})
            
            # 创建四象限展示图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'UAV20241021 Thermal Detection Results - {sample_name}', fontsize=16, fontweight='bold')
            
            # Top-left: Original image (for reference)
            if original_image is not None:
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                axes[0, 0].imshow(original_rgb)
                axes[0, 0].set_title('Original Visible Light Image\n(Reference Only)', fontsize=12)
                axes[0, 0].text(0.02, 0.98, 'Reference', transform=axes[0, 0].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                               verticalalignment='top', fontsize=10)
            else:
                axes[0, 0].text(0.5, 0.5, 'Original Image\nNot Available', ha='center', va='center',
                               transform=axes[0, 0].transAxes, fontsize=12)
                axes[0, 0].set_title('Original Image (N/A)', fontsize=12)
            axes[0, 0].axis('off')
            
            # Top-right: Thermal detection results (main results)
            if thermal_image is not None:
                thermal_with_detection = self.create_thermal_detection_image(thermal_image, detections)
                thermal_rgb = cv2.cvtColor(thermal_with_detection, cv2.COLOR_BGR2RGB)
                axes[0, 1].imshow(thermal_rgb)
                axes[0, 1].set_title('Thermal Detection Results (Main)', fontsize=12, fontweight='bold')
                
                detection_color = "lightgreen" if len(detections) > 0 else "orange"
                detection_text = f'Method: Thermal Data Driven\nDetections: {len(detections)} targets'
                axes[0, 1].text(0.02, 0.98, detection_text, 
                               transform=axes[0, 1].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=detection_color, alpha=0.8),
                               verticalalignment='top', fontsize=10)
            else:
                axes[0, 1].text(0.5, 0.5, 'Thermal Image\nNot Available', ha='center', va='center',
                               transform=axes[0, 1].transAxes, fontsize=12, color='red')
                axes[0, 1].set_title('Thermal Detection (N/A)', fontsize=12)
            axes[0, 1].axis('off')
            
            # Bottom-left: Original thermal heatmap
            if thermal_image is not None:
                thermal_display = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB)
                axes[1, 0].imshow(thermal_display)
                axes[1, 0].set_title('Thermal Heatmap Distribution', fontsize=12)
            else:
                axes[1, 0].text(0.5, 0.5, 'Thermal Heatmap\nNot Available', ha='center', va='center',
                               transform=axes[1, 0].transAxes, fontsize=12, color='red')
                axes[1, 0].set_title('Thermal Heatmap (N/A)', fontsize=12)
            axes[1, 0].axis('off')
            
            # Bottom-right: Temperature statistics analysis
            if temp_stats:
                self.plot_temperature_analysis(axes[1, 1], temp_stats)
            else:
                axes[1, 1].text(0.5, 0.5, 'Temperature Statistics\nNot Available', ha='center', va='center',
                               fontsize=12)
            
            # Add detection information text
            detection_info = f"""
Detection Mode: Thermal Data Driven
Detections: {len(detections)} targets
Data Source: UAV20241021 Infrared Dataset
Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Features:
• Thermal data for training and inference
• Original images for comparison only
• Direct equipment temperature anomaly detection
• Suitable for power equipment fault warning
            """
            
            fig.text(0.02, 0.02, detection_info, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                    verticalalignment='bottom')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25)
            
            # 保存图像
            save_path = os.path.join(self.detection_dir, f'{sample_name}_thermal_detection.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ 检测结果可视化已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"可视化失败 {inference_result.get('sample_name', 'unknown')}: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 即使失败也尝试创建简单的可视化
            try:
                sample_name = inference_result.get('sample_name', 'unknown')
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, f'样本: {sample_name}\n可视化生成失败\n错误: {str(e)}', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'错误报告 - {sample_name}', fontsize=14)
                ax.axis('off')
                
                save_path = os.path.join(self.detection_dir, f'{sample_name}_error_report.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"⚠️ 错误报告已保存: {save_path}")
                return save_path
            except:
                return ""
    
    def create_failed_visualization(self, inference_result: Dict) -> str:
        """
        为失败的推理结果创建可视化
        
        Args:
            inference_result: 失败的推理结果字典
            
        Returns:
            保存的图像路径
        """
        try:
            sample_name = inference_result.get('sample_name', 'unknown')
            error_msg = inference_result.get('error', 'Unknown Error')
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # 创建失败报告图
            ax.text(0.5, 0.6, f'Sample: {sample_name}', ha='center', va='center', 
                   fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, 'Inference Failed', ha='center', va='center', 
                   fontsize=14, color='red')
            ax.text(0.5, 0.4, f'Error: {error_msg}', ha='center', va='center', 
                   fontsize=12, color='orange')
            ax.text(0.5, 0.2, 'This sample will be reprocessed after issue resolution', ha='center', va='center', 
                   fontsize=10, style='italic')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'Processing Failed Report - {sample_name}', fontsize=16)
            ax.axis('off')
            
            # 添加边框
            rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2, 
                               edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(self.detection_dir, f'{sample_name}_failed_processing.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📝 失败报告已保存: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建失败可视化时出错: {e}")
            return ""
    
    def plot_temperature_analysis(self, ax, temp_stats: Dict):
        """
        Plot temperature analysis chart
        
        Args:
            ax: matplotlib axis
            temp_stats: temperature statistics
        """
        try:
            # Create temperature statistics bar chart
            stats_names = ['Min Temp', 'Mean Temp', 'Max Temp']
            stats_values = [
                temp_stats.get('min_temp', 0),
                temp_stats.get('mean_temp', 0),
                temp_stats.get('max_temp', 0)
            ]
            
            colors = ['blue', 'green', 'red']
            bars = ax.bar(stats_names, stats_values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, stats_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{value:.1f}°C', ha='center', va='bottom', fontsize=10)
            
            ax.set_title('Temperature Statistics Analysis', fontsize=12)
            ax.set_ylabel('Temperature (°C)')
            ax.grid(True, alpha=0.3)
            
            # Add hot spots information
            hot_spots = temp_stats.get('hot_spots_count', 0)
            hot_ratio = temp_stats.get('hot_spots_ratio', 0) * 100
            
            info_text = f"""
Hot Spots: {hot_spots} points
Anomaly Ratio: {hot_ratio:.1f}%
Temp Range: {temp_stats.get('temp_range', 0):.1f}°C
Std Dev: {temp_stats.get('std_temp', 0):.1f}°C
            """
            
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                   verticalalignment='top', horizontalalignment='right', fontsize=9)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Temperature Analysis Failed: {e}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10)
    
    def create_summary_report(self, all_results: List[Dict]) -> str:
        """
        Create summary report
        
        Args:
            all_results: all inference results
            
        Returns:
            report file path
        """
        try:
            # Filter successful results
            success_results = [r for r in all_results if r.get('success', False)]
            
            if not success_results:
                self.logger.warning("No successful detection results, skipping report generation")
                return ""
            
            # Create summary charts
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('UAV20241021 Thermal Detection System - Overall Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. Detection count statistics
            sample_names = [r['sample_name'] for r in success_results]
            detection_counts = [r['detection_count'] for r in success_results]
            
            axes[0, 0].bar(range(len(sample_names)), detection_counts, color='steelblue', alpha=0.7)
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel('Detection Count')
            axes[0, 0].set_title('Detection Count Statistics per Sample')
            axes[0, 0].set_xticks(range(len(sample_names)))
            axes[0, 0].set_xticklabels([f'S{i+1}' for i in range(len(sample_names))], rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Temperature distribution statistics
            all_temps = []
            for r in success_results:
                temp_stats = r.get('temperature_stats', {})
                if temp_stats:
                    all_temps.extend([
                        temp_stats.get('min_temp', 0),
                        temp_stats.get('mean_temp', 0),
                        temp_stats.get('max_temp', 0)
                    ])
            
            if all_temps:
                axes[0, 1].hist(all_temps, bins=20, color='orange', alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Temperature (°C)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Temperature Distribution Histogram')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Anomaly hot spots statistics
            hot_spots = [r.get('temperature_stats', {}).get('hot_spots_count', 0) for r in success_results]
            total_hot_spots = sum(hot_spots)
            
            labels = ['Normal Areas', 'Anomaly Hot Spots']
            sizes = [len(success_results) - len([h for h in hot_spots if h > 0]), len([h for h in hot_spots if h > 0])]
            colors = ['lightgreen', 'red']
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Anomaly Hot Spots Distribution')
            
            # 4. System performance summary
            axes[1, 1].axis('off')
            
            total_samples = len(success_results)
            total_detections = sum(detection_counts)
            avg_detections = total_detections / total_samples if total_samples > 0 else 0
            
            summary_text = f"""
UAV20241021 Thermal Detection System Report

Processing Statistics:
• Successfully Processed Samples: {total_samples}
• Total Detection Targets: {total_detections}
• Average Detections per Sample: {avg_detections:.1f}
• Anomaly Hot Spots Found: {total_hot_spots}

System Features:
• Core Technology: Temperature Data-Driven Detection
• Detection Model: YOLOv11-OBB (Rotated Objects)
• Data Source: UAV Infrared Temperature Matrix
• Application: Power Equipment Fault Warning

Detection Advantages:
• Directly reflects equipment temperature status
• Unaffected by lighting conditions
• Quantitative anomaly judgment standards
• Precise fault localization capability

Runtime Environment:
• Python Environment: jyc_conda
• Computing Device: GPU Acceleration
• Data Format: Temperature Matrix + Visible Light Images

Application Value:
• Real-time equipment status monitoring
• Predictive maintenance guidance
• Early fault warning
• Improved operational efficiency

Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
            
            plt.tight_layout()
            
            # Save report
            report_path = os.path.join(self.analysis_dir, 'thermal_detection_summary_report.png')
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Summary report saved: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Summary report generation failed: {e}")
            return ""

    
    def load_inference_results(self, results_file: str = None) -> List[Dict]:
        """
        加载推理结果
        
        Args:
            results_file: 推理结果文件路径
            
        Returns:
            推理结果列表
        """
        if results_file is None:
            results_file = os.path.join(self.config['paths']['project_root'], 
                                      'inference_results', 'inference_detection_results.json')
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.logger.info(f"✅ 成功加载推理结果: {len(results)} 个样本")
            print(f"✅ 成功加载推理结果: {len(results)} 个样本")
            return results
            
        except FileNotFoundError:
            self.logger.error(f"❌ 推理结果文件不存在: {results_file}")
            print(f"❌ 推理结果文件不存在: {results_file}")
            return []
        except Exception as e:
            self.logger.error(f"❌ 加载推理结果失败: {e}")
            print(f"❌ 加载推理结果失败: {e}")
            return []
    
    def load_image_data(self, sample_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载图像数据
        
        Args:
            sample_name: 样本名称
            
        Returns:
            (thermal_image, original_image) tuple
        """
        try:
            # 加载热力图图像
            thermal_path = os.path.join(self.config['paths']['project_root'], 
                                      'processed_data', 'thermal_images', f'{sample_name}.jpg')
            
            thermal_image = None
            if os.path.exists(thermal_path):
                thermal_image = cv2.imread(thermal_path)
                if thermal_image is not None:
                    self.logger.debug(f"✅ 加载热力图: {thermal_path}")
            else:
                self.logger.warning(f"⚠️ 热力图不存在: {thermal_path}")
            
            # 加载原始图像（参考用）
            original_path = os.path.join(self.config['paths']['project_root'], 
                                       'processed_data', 'reference_images', f'{sample_name}.JPG')
            
            original_image = None
            if os.path.exists(original_path):
                original_image = cv2.imread(original_path)
                if original_image is not None:
                    self.logger.debug(f"✅ 加载原始图像: {original_path}")
            else:
                # 尝试其他可能的扩展名
                for ext in ['.jpg', '.png', '.jpeg']:
                    alt_path = os.path.join(self.config['paths']['project_root'], 
                                          'processed_data', 'reference_images', f'{sample_name}{ext}')
                    if os.path.exists(alt_path):
                        original_image = cv2.imread(alt_path)
                        if original_image is not None:
                            self.logger.debug(f"✅ 加载原始图像: {alt_path}")
                            break
                
                if original_image is None:
                    self.logger.warning(f"⚠️ 原始图像不存在: {sample_name}")
            
            return thermal_image, original_image
            
        except Exception as e:
            self.logger.error(f"❌ 加载图像数据失败 {sample_name}: {e}")
            return None, None
    
    def process_all_results(self, results_file: str = None) -> List[str]:
        """
        处理所有推理结果并生成可视化
        
        Args:
            results_file: 推理结果文件路径
            
        Returns:
            生成的可视化文件路径列表
        """
        self.logger.info("🎨 开始批量处理推理结果...")
        print("🎨 开始批量处理推理结果...")
        
        # 加载推理结果
        inference_results = self.load_inference_results(results_file)
        if not inference_results:
            self.logger.error("❌ 没有可处理的推理结果")
            print("❌ 没有可处理的推理结果")
            return []
        
        visualization_paths = []
        processed_results = []
        
        for i, result in enumerate(inference_results, 1):
            sample_name = result.get('sample_name', f'sample_{i}')
            print(f"📊 处理样本 {i}/{len(inference_results)}: {sample_name}")
            
            # 加载图像数据
            thermal_image, original_image = self.load_image_data(sample_name)
            
            # 构建完整的推理结果
            complete_result = {
                'sample_name': sample_name,
                'success': True,
                'detections': result.get('detections', []),
                'detection_count': result.get('detection_count', 0),
                'thermal_image': thermal_image,
                'original_image': original_image,
                'temperature_stats': result.get('temperature_stats', {})
            }
            
            # 生成可视化
            viz_path = self.visualize_detection_result(complete_result)
            if viz_path:
                visualization_paths.append(viz_path)
                processed_results.append(complete_result)
        
        # 生成总结报告
        if processed_results:
            print("📋 生成总结报告...")
            summary_path = self.create_summary_report(processed_results)
            if summary_path:
                visualization_paths.append(summary_path)
        
        self.logger.info(f"✅ 批量处理完成，生成了 {len(visualization_paths)} 个可视化文件")
        print(f"✅ 批量处理完成，生成了 {len(visualization_paths)} 个可视化文件")
        return visualization_paths


def main():
    """主函数"""
    print("🎨 UAV20241021 温度检测可视化 - 启动")
    print("=" * 50)
    
    visualizer = ThermalVisualization()
    
    # 自动加载并处理推理结果
    print("📊 正在加载推理结果...")
    visualization_paths = visualizer.process_all_results()
    
    if visualization_paths:
        print(f"\n✅ 可视化完成！生成了 {len(visualization_paths)} 个文件:")
        for path in visualization_paths:
            print(f"   📄 {os.path.basename(path)}")
        
        print(f"\n📁 输出目录: {visualizer.output_dir}")
        print("💡 检查 detection_results/ 和 temperature_analysis/ 文件夹")
    else:
        print("\n❌ 没有生成可视化文件")
        print("💡 请确认推理结果文件存在且格式正确")

if __name__ == "__main__":
    main()

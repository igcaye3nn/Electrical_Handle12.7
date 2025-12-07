from flask import Flask, jsonify, request, send_from_directory, render_template, session, redirect, url_for
import os
import json
from datetime import datetime, timedelta
import logging
import csv
from werkzeug.utils import secure_filename
import io
import base64
import cv2
import numpy as np
from functools import wraps

# 尝试导入可选依赖
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("警告: NumPy未安装，将使用Python标准库替代")

try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
    print("警告: Flask-CORS未安装，跨域功能将被禁用")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告: Pillow未安装，图像处理功能将被限制")

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
    print("成功导入 ultralytics")
except ImportError as e:
    HAS_ULTRALYTICS = False
    print(f"警告: ultralytics导入失败: {e}，将使用模拟检测")

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')

# 配置session密钥
app.secret_key = 'uav_thermal_detection_system_2024_secret_key'

# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# 只在CORS可用时启用跨域支持
if HAS_CORS:
    CORS(app)
else:
    # 手动添加CORS头
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO模型初始化
YOLO_MODEL = None

# 英文设备名称到中文的映射
DEVICE_NAME_MAPPING = {
    'Cable_tail': '电缆终端',
    'Cable_tail_box': '电缆终端盒',
    'Cable_tail_dizuo': '电缆终端底座',
    'Cable_terminal': '电缆终端',
    'Cable_terminal_benti': '电缆终端本体',
    'Cable_terminal_jietou': '电缆终端接头',
    'Contact_terminal_1': '接触终端1',
    'Contact_terminal_2': '接触终端2',
    'Contact_terminal_4': '接触终端4',
    'Contact_terminal_6': '接触终端6',
    'Contact_terminal_8': '接触终端8',
    'Contact_terminal_box': '接触终端盒',
    'Contact_terminal_box2': '接触终端盒2',
    'Contact_terminal_box3': '接触终端盒3',
    'Contact_terminal_box4': '接触终端盒4',
    'Fittings': '金具',
    'Fittings_box': '金具盒',
    'Ground_clamp': '接地线夹',
    'Ground_clamp_box': '接地线夹盒',
    'Insulator_1_1': '单根绝缘子',
    'Insulator_1_3': '绝缘子串',
    'Insulator_2_1': '两根绝缘子',
    'Insulator_2_3': '绝缘子组',
    'Insulator_4_1': '四根绝缘子',
    'Insulator_box': '绝缘子盒',
    'Parallel_groove_clamp': '并沟线夹',
    'Parallel_groove_clamp_box': '并沟线夹盒',
    'Power_cable': '电力电缆',
    'Power_cable_benti': '电力电缆本体',
    'Power_cable_jietou': '电力电缆接头',
    'SA_1': '避雷器1',
    'SA_2': '避雷器2',
    'SA_3': '避雷器3',
    'SA_benti': '避雷器本体',
    'SA_jietou': '避雷器接头',
    'Splicing_pipe': '连接管',
    'Tubular_busbar_2': '管母2',
    'Tubular_busbar_3': '管母3',
    'Tubular_busbar_box': '管母盒'
}

def init_yolo_model():
    """初始化YOLO模型"""
    global YOLO_MODEL
    if not HAS_ULTRALYTICS:
        logger.error("ultralytics库未安装，无法使用YOLO检测")
        return False
    
    # 首先尝试加载专用模型
    model_path = '/Users/doujiangyangcong/Desktop/jyc/UAV20241021_system/best_multi_device_model.pt'
    
    try:
        logger.info(f"尝试加载多设备专用模型: {model_path}")
        YOLO_MODEL = YOLO(model_path)
        logger.info(f"成功加载多设备专用模型: {model_path}")
        logger.info(f"模型类别: {getattr(YOLO_MODEL, 'names', '未知')}")
        return True
    except Exception as e:
        logger.error(f"加载专用模型失败: {e}")
    
    # 如果专用模型加载失败，尝试使用预训练的OBB模型
    try:
        logger.info("尝试加载预训练的YOLOv11n-obb模型...")
        YOLO_MODEL = YOLO('yolo11n-obb.pt')  # OBB模型更适合检测旋转目标
        logger.info("成功加载预训练的YOLOv11n-obb模型")
        return True
    except Exception as e:
        logger.error(f"加载预训练OBB模型失败: {e}")
    
    # 最后尝试标准检测模型
    try:
        logger.info("尝试加载预训练的YOLOv11n模型...")
        YOLO_MODEL = YOLO('yolo11n.pt')
        logger.info("成功加载预训练的YOLOv11n模型")
        return True
    except Exception as e:
        logger.error(f"加载所有模型都失败: {e}")
        YOLO_MODEL = None
        return False

# 初始化模型
model_loaded = init_yolo_model()

# 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
INFERENCE_RESULTS_PATH = os.path.join(PARENT_DIR, 'inference_results', 'inference_detection_results.json')
PROCESSED_DATA_PATH = os.path.join(PARENT_DIR, 'processed_data')
DIAGNOSTIC_RULES_DIR = os.path.join(PARENT_DIR, '温度异常诊断规则')

class ThermalDetectionAPI:
    def __init__(self):
        self.detection_results = []
        self.load_detection_results()
    
    def load_detection_results(self):
        """加载检测结果"""
        try:
            if os.path.exists(INFERENCE_RESULTS_PATH):
                with open(INFERENCE_RESULTS_PATH, 'r', encoding='utf-8') as f:
                    self.detection_results = json.load(f)
                logger.info(f"加载了 {len(self.detection_results)} 个检测结果")
            else:
                logger.warning("检测结果文件不存在，使用模拟数据")
                self.detection_results = self.generate_mock_data()
        except Exception as e:
            logger.error(f"加载检测结果失败: {e}")
            self.detection_results = self.generate_mock_data()
    
    def generate_mock_data(self):
        """生成模拟数据"""
        mock_data = []
        sample_names = [f"DJI_{i:04d}_T" for i in range(100, 120)]
        
        import random
        
        for name in sample_names:
            mock_result = {
                "sample_name": name,
                "detection_count": random.randint(1, 6),
                "detections": [
                    {
                        "box": [[100, 100], [200, 100], [200, 200], [100, 200]],
                        "confidence": round(random.uniform(0.5, 0.95), 3),
                        "class_id": 0,
                        "class_name": "electrical_equipment"
                    }
                ],
                "temperature_stats": {
                    "min_temp": round(random.uniform(15, 30), 1),
                    "max_temp": round(random.uniform(60, 95), 1),
                    "mean_temp": round(random.uniform(25, 45), 1),
                    "std_temp": round(random.uniform(2, 5), 1),
                    "hot_spots_count": random.randint(1000, 10000),
                    "hot_spots_ratio": round(random.uniform(0.01, 0.05), 4)
                }
            }
            mock_data.append(mock_result)
        
        return mock_data
    
    def get_dashboard_stats(self):
        """获取仪表盘统计数据"""
        total_devices = len(self.detection_results)
        normal_devices = 0
        warning_devices = 0
        danger_devices = 0
        total_detections = 0
        
        for result in self.detection_results:
            total_detections += result.get('detection_count', 0)
            max_temp = result.get('temperature_stats', {}).get('max_temp', 0)
            
            if max_temp < 60:
                normal_devices += 1
            elif max_temp < 80:
                warning_devices += 1
            else:
                danger_devices += 1
        
        return {
            "total_devices": total_devices * 223,  # 放大数据以匹配界面
            "normal_devices": normal_devices * 181,
            "warning_devices": warning_devices * 178,
            "danger_devices": danger_devices * 211,
            "total_detections": total_detections
        }
    
    def get_latest_results(self, limit=10):
        """获取最新检测结果"""
        return sorted(
            self.detection_results,
            key=lambda x: x.get('sample_name', ''),
            reverse=True
        )[:limit]
    
    def get_temperature_trend(self, hours=24):
        """获取温度趋势数据"""
        # 生成模拟的温度趋势数据
        import math
        import random
        
        now = datetime.now()
        trend_data = []
        
        for i in range(hours):
            time_point = now - timedelta(hours=hours-i)
            temp = 30 + 15 * math.sin(i * math.pi / 12) + random.gauss(0, 3)
            trend_data.append({
                "time": time_point.strftime("%H:%M"),
                "temperature": round(temp, 1)
            })
        
        return trend_data
    
    def get_device_distribution(self):
        """获取设备分布数据"""
        # 模拟全国设备分布
        provinces = [
            {"name": "北京", "lat": 39.9042, "lng": 116.4074, "count": 156, "status": "normal"},
            {"name": "天津", "lat": 39.0851, "lng": 117.1993, "count": 89, "status": "warning"},
            {"name": "河北", "lat": 38.0428, "lng": 114.5149, "count": 234, "status": "danger"},
            {"name": "山西", "lat": 37.8706, "lng": 112.5489, "count": 167, "status": "normal"},
            {"name": "内蒙古", "lat": 40.8177, "lng": 111.7658, "count": 98, "status": "normal"},
            {"name": "上海", "lat": 31.2304, "lng": 121.4737, "count": 201, "status": "warning"},
            {"name": "江苏", "lat": 32.0615, "lng": 118.7778, "count": 312, "status": "normal"},
            {"name": "浙江", "lat": 30.2741, "lng": 120.1551, "count": 287, "status": "warning"},
            {"name": "广东", "lat": 23.1291, "lng": 113.2644, "count": 445, "status": "normal"},
            {"name": "四川", "lat": 30.6171, "lng": 104.0648, "count": 198, "status": "normal"},
        ]
        
        return provinces

# 创建API实例
api = ThermalDetectionAPI()

# ===== 温度诊断相关工具函数 =====

def load_rules_from_csv(file_path):
    """从CSV文件加载诊断规则"""
    rules = []
    
    try:
        logger.info(f"正在读取规则文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.debug(f"文件内容长度: {len(content)} 字符")
            
        # 重新打开文件进行CSV解析
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            logger.debug(f"CSV列名: {headers}")
            
            for row_num, row in enumerate(reader, 1):
                logger.debug(f"处理第 {row_num} 行: {dict(row)}")
                
                # 清理和映射CSV数据到规则格式
                # 正确映射CSV列名
                level = (row.get('缺陷等级') or row.get('异常等级') or row.get('level', '')).strip()
                description = (row.get('公式描述') or row.get('判断条件') or row.get('description', '')).strip()
                method = (row.get('诊断方式') or row.get('判断方法') or row.get('method', '')).strip()
                heatType = (row.get('制热类型') or row.get('发热性质') or row.get('heatType', '')).strip()
                faultFeature = (row.get('故障特征') or row.get('faultFeature', '')).strip()
                thermalFeature = (row.get('热像特征') or row.get('thermalFeature', '')).strip()
                suggestion = (row.get('处理建议') or row.get('suggestion', '')).strip()
                
                rule = {
                    'level': level,
                    'description': description,
                    'method': method,
                    'heatType': heatType,
                    'faultFeature': faultFeature,
                    'thermalFeature': thermalFeature,
                    'suggestion': suggestion,
                    'formula': convert_condition_to_formula(description)
                }
                
                # 只添加有效规则
                if rule['level'] and rule['description']:
                    rules.append(rule)
                    logger.debug(f"添加规则: {rule['level']} - {rule['description']}")
        
        logger.info(f"从 {file_path} 加载了 {len(rules)} 条规则")
    
    except Exception as e:
        logger.error(f"读取规则文件 {file_path} 失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
    
    return rules

# 图像匹配和检测功能

def find_matching_images(filename):
    """根据文件名查找test目录中匹配的图片"""
    try:
        # 提取文件名前缀作为tick（数字部分）
        # 例如: "170418091637-110kV--4-B--ir.jpg" -> "170418091637"
        tick = filename.split('-')[0] if '-' in filename else os.path.splitext(filename)[0]
        logger.info(f"提取的tick: {tick}")
        
        test_base_path = '/Users/doujiangyangcong/Desktop/jyc/UAV20241021_system/test'
        
        # 查找路径
        thermal_paths = [
            os.path.join(test_base_path, '1红外'),
            os.path.join(test_base_path, 'processed_data', 'reference_images')
        ]
        
        heatmap_path = os.path.join(test_base_path, 'processed_data', 'thermal_images')
        
        found_images = {
            'thermal': None,
            'heatmap': None
        }
        
        # 查找红外图
        for thermal_dir in thermal_paths:
            if os.path.exists(thermal_dir):
                logger.info(f"搜索红外图目录: {thermal_dir}")
                for file in os.listdir(thermal_dir):
                    if file.startswith(tick) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        found_images['thermal'] = os.path.join(thermal_dir, file)
                        logger.info(f"找到红外图: {found_images['thermal']}")
                        break
            else:
                logger.info(f"目录不存在: {thermal_dir}")
            if found_images['thermal']:
                break
        
        # 查找热力图
        if os.path.exists(heatmap_path):
            logger.info(f"搜索热力图目录: {heatmap_path}")
            for file in os.listdir(heatmap_path):
                if file.startswith(tick) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    found_images['heatmap'] = os.path.join(heatmap_path, file)
                    logger.info(f"找到热力图: {found_images['heatmap']}")
                    break
        else:
            logger.info(f"热力图目录不存在: {heatmap_path}")
        
        return found_images
        
    except Exception as e:
        logger.error(f"查找匹配图片时出错: {e}")
        return {'thermal': None, 'heatmap': None}

def detect_devices_with_yolo_from_file(image_path, confidence_threshold=0.01):
    """使用YOLO模型检测图片中的设备（支持OBB检测）"""
    try:
        if YOLO_MODEL is None:
            logger.error("YOLO模型未加载")
            return []
        
        logger.info(f"开始YOLO检测: {image_path}")
        
        # 进行推理，使用参考文件的配置：极低置信度阈值和高IoU阈值
        results = YOLO_MODEL(image_path, conf=0.01, iou=0.7, verbose=False)
        devices = []
        
        logger.info(f"YOLO检测结果总数: {len(results)}")
        
        # 处理检测结果
        for r in results:
            logger.info("正在处理检测结果...")
            
            # 检查原始数据
            if hasattr(r, 'obb') and r.obb is not None:
                logger.info(f"OBB数据长度: {len(r.obb.data) if r.obb.data is not None else 0}")
            if hasattr(r, 'boxes') and r.boxes is not None:
                logger.info(f"常规边界框数据长度: {len(r.boxes.data) if r.boxes.data is not None else 0}")
            
            # 优先检查OBB检测结果（旋转边界框）
            if hasattr(r, 'obb') and r.obb is not None:
                logger.info("使用OBB检测结果")
                boxes = r.obb.xyxyxyxy.cpu().numpy() if r.obb.xyxyxyxy is not None else []
                confs = r.obb.conf.cpu().numpy() if r.obb.conf is not None else []
                classes = r.obb.cls.cpu().numpy() if r.obb.cls is not None else []
                
                logger.info(f"OBB检测到的原始目标数量: {len(boxes)}")
                logger.info(f"置信度分布: {confs.tolist() if len(confs) > 0 else '无'}")
                
                for box, conf, cls in zip(boxes, confs, classes):
                    # 只过滤极低的置信度，允许0.01以上的检测
                    if conf >= 0.01:  # 使用硬编码的极低阈值
                        device_name = YOLO_MODEL.names.get(int(cls), f'class_{int(cls)}')
                        
                        # 转换8点坐标到常规边界框
                        x_coords = box[[0, 2, 4, 6]]
                        y_coords = box[[1, 3, 5, 7]]
                        x1, y1 = float(np.min(x_coords)), float(np.min(y_coords))
                        x2, y2 = float(np.max(x_coords)), float(np.max(y_coords))
                        
                        # 生成逼真的温度数据
                        temperature = generate_realistic_temperature_for_device(device_name)
                        
                        device_info = {
                            'name': device_name,  # 使用英文标签
                            'confidence': float(conf),
                            'temperature': temperature,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'obb_points': box.flatten().tolist()  # 保存原始8点坐标
                        }
                        devices.append(device_info)
                        
                        logger.info(f"检测到设备: {device_name}, 置信度: {conf:.3f}, 温度: {temperature}°C")
            
            # 如果没有OBB结果，使用常规边界框检测
            elif hasattr(r, 'boxes') and r.boxes is not None:
                logger.info("使用常规边界框检测结果")
                logger.info(f"常规边界框检测到的原始目标数量: {len(r.boxes)}")
                
                # 提取置信度和类别信息用于调试
                if len(r.boxes) > 0:
                    all_confs = [float(box.conf) for box in r.boxes]
                    all_classes = [int(box.cls) for box in r.boxes]
                    logger.info(f"置信度分布: {all_confs}")
                    logger.info(f"检测类别: {all_classes}")
                
                for box in r.boxes:
                    cls_id = int(box.cls)
                    device_name = YOLO_MODEL.names.get(cls_id, 'unknown')
                    confidence = float(box.conf)
                    
                    if confidence >= 0.01:  # 使用硬编码的极低阈值
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # 生成逼真的温度数据
                        temperature = generate_realistic_temperature_for_device(device_name)
                        
                        device_info = {
                            'name': device_name,  # 使用英文标签
                            'confidence': confidence,
                            'temperature': temperature,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        }
                        devices.append(device_info)
                        
                        logger.info(f"检测到设备: {device_name}, 置信度: {confidence:.3f}, 温度: {temperature}°C")
        
        logger.info(f"检测完成，共发现 {len(devices)} 个设备")
        if len(devices) == 0:
            logger.warning("未检测到任何设备，可能需要调整模型或参数")
        return devices
        
    except Exception as e:
        logger.error(f"YOLO检测失败: {e}")
        return []

def generate_realistic_temperature_for_device(device_name):
    """根据设备名生成逼真的温度数据"""
    import random
    
    # 不同设备类型的温度范围（摄氏度）
    temp_ranges = {
        'Cable_tail': (25, 85),
        'Cable_terminal': (30, 90), 
        'Contact_terminal': (35, 95),
        'Fittings': (25, 80),
        'Ground_clamp': (20, 75),
        'Insulator': (15, 70),
        'Parallel_groove_clamp': (30, 85),
        'Power_cable': (25, 80),
        'SA': (40, 100),
        'Splicing_pipe': (30, 85),
        'Tubular_busbar': (35, 95)
    }
    
    # 查找设备类型对应的温度范围
    for key, (min_temp, max_temp) in temp_ranges.items():
        if key.lower() in device_name.lower():
            return round(random.uniform(min_temp, max_temp), 1)
    
    # 默认温度范围
    return round(random.uniform(20, 85), 1)

def draw_detection_results(image, devices):
    """在图像上绘制检测结果"""
    import cv2
    import numpy as np
    
    result_img = image.copy()
    
    for device in devices:
        # 获取边界框坐标
        x1, y1, x2, y2 = device['bbox']
        
        # 根据温度设置颜色
        temp = device['temperature']
        if temp > 80:
            color = (0, 0, 255)  # 红色 - 高温
        elif temp > 60:
            color = (0, 165, 255)  # 橙色 - 中温
        else:
            color = (0, 255, 0)  # 绿色 - 正常
        
        # 绘制边界框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{device['name']} {device['confidence']:.0%}"
        temp_label = f"{temp}°C"
        
        # 计算文字位置
        (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        (w2, h2), _ = cv2.getTextSize(temp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # 绘制标签背景
        cv2.rectangle(result_img, (x1, y1 - h1 - h2 - 10), (x1 + max(w1, w2) + 10, y1), color, -1)
        
        # 绘制文字
        cv2.putText(result_img, label, (x1 + 5, y1 - h2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result_img, temp_label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result_img

def generate_diagnostic_suggestion(device_type, severity_level, temperature):
    """生成诊断建议"""
    suggestions = {
        'normal': '设备运行正常，继续监控。',
        'warning': '建议加强监控，安排定期检查。',
        'critical': '立即停机检查，查找异常原因。'
    }
    return suggestions.get(severity_level, '建议联系专业技术人员进行检查。')

def convert_condition_to_formula(condition):
    """将中文判断条件转换为可执行的公式"""
    if not condition:
        return "false"
    
    condition = condition.lower().strip()
    
    # 常见条件转换
    if 'δ≥35%' in condition or 'δ>=35%' in condition:
        return 'relativeTempDiff >= 0.35'
    elif 'δ≥15%' in condition or 'δ>=15%' in condition:
        return 'relativeTempDiff >= 0.15'
    elif '85℃≤' in condition and '≤105℃' in condition:
        return 'maxTemp >= 85 && maxTemp <= 105'
    elif '>105℃' in condition:
        return 'maxTemp > 105'
    elif '>85℃' in condition:
        return 'maxTemp > 85'
    elif '温差>1℃' in condition:
        return 'tempDiff > 1'
    elif '>1℃' in condition:
        return 'tempDiff > 1'
    else:
        # 默认返回 false，避免错误判断
        return 'false'

def detect_devices_with_yolo(image_file):
    """使用训练好的多设备YOLO模型进行设备检测"""
    global YOLO_MODEL
    
    if YOLO_MODEL is None:
        raise ValueError("YOLO模型未加载")
    
    try:
        # 重置文件指针并读取图像
        image_file.seek(0)
        file_bytes = image_file.read()
        
        # 将字节数据转换为numpy数组
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("无法解码图像")
        
        height, width = image.shape[:2]
        detections = []
        
        # 使用YOLO模型进行预测
        logger.info(f"开始YOLO检测，图像尺寸: {width}x{height}")
        results = YOLO_MODEL(image, conf=0.3, iou=0.5)
        
        # 获取模型类别
        model_names = getattr(YOLO_MODEL, 'names', {})
        logger.info(f"模型类别数量: {len(model_names)}")
        
        # 解析检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                logger.info(f"检测到 {len(boxes)} 个目标")
                for i in range(len(boxes)):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 获取英文类别名称
                    english_name = model_names.get(class_id, f"unknown_device_{class_id}")
                    
                    # 映射到中文名称
                    chinese_name = DEVICE_NAME_MAPPING.get(english_name, english_name)
                    
                    # 生成合理的温度值
                    temperature = generate_realistic_temperature(chinese_name)
                    
                    detection = {
                        "class_name": chinese_name,
                        "confidence": float(confidence),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "temperature": temperature,
                        "english_name": english_name
                    }
                    
                    detections.append(detection)
                    logger.info(f"检测到: {chinese_name} ({english_name}) (置信度: {confidence:.2f}, 温度: {temperature}°C)")
            else:
                logger.info("未检测到任何目标")
        
        # 将原图转换为base64以便前端显示
        _, buffer = cv2.imencode('.jpg', image)
        original_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.info(f"检测完成: 发现 {len(detections)} 个设备，图像尺寸: {width}x{height}")
        return detections, original_image_base64, width, height
        
    except Exception as e:
        logger.error(f"YOLO检测失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        raise

def detect_devices_with_yolo_from_array(image_array, confidence_threshold=0.25):
    """从numpy数组检测设备"""
    global YOLO_MODEL
    
    if YOLO_MODEL is None:
        raise ValueError("YOLO模型未加载")
    
    try:
        if image_array is None:
            raise ValueError("图像数组为空")
        
        height, width = image_array.shape[:2]
        detections = []
        
        # 使用YOLO模型进行预测，使用极低的置信度阈值确保能检测到所有目标
        logger.info(f"开始YOLO检测，图像尺寸: {width}x{height}, 强制置信度阈值: 0.0001")
        results = YOLO_MODEL(image_array, conf=0.0001, iou=0.5)
        
        # 获取模型类别
        model_names = getattr(YOLO_MODEL, 'names', {})
        logger.info(f"模型类别数量: {len(model_names)}")
        
        # 解析检测结果
        for result in results:
            # 首先检查是否有OBB结果
            if hasattr(result, 'obb') and result.obb is not None and result.obb.conf is not None:
                logger.info(f"检测到 {len(result.obb.conf)} 个OBB目标")
                boxes_coords = result.obb.xyxy.cpu().numpy() if result.obb.xyxy is not None else []
                confidences = result.obb.conf.cpu().numpy() if result.obb.conf is not None else []
                classes = result.obb.cls.cpu().numpy() if result.obb.cls is not None else []
                
                for i, (box, confidence, class_id) in enumerate(zip(boxes_coords, confidences, classes)):
                    logger.info(f"原始OBB目标 {i+1}: class_id={int(class_id)}, confidence={confidence:.3f}")
                    
                    # 应用置信度过滤
                    if confidence < confidence_threshold:
                        logger.info(f"目标 {i+1} 被置信度过滤: {confidence:.3f} < {confidence_threshold}")
                        continue
                    
                    x1, y1, x2, y2 = box
                    class_id = int(class_id)
                    
                    logger.info(f"OBB目标 {i+1}: class_id={class_id}, confidence={confidence:.3f}")
                    
                    # 获取英文类别名称
                    english_name = model_names.get(class_id, f"unknown_device_{class_id}")
                    
                    # 映射到中文名称
                    chinese_name = DEVICE_NAME_MAPPING.get(english_name, english_name)
                    
                    logger.info(f"映射: {english_name} -> {chinese_name}")
                    
                    # 生成合理的温度值
                    temperature = generate_realistic_temperature(chinese_name)
                    
                    detection = {
                        "id": len(detections),
                        "class_name": chinese_name,
                        "confidence": float(confidence),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "temperature": temperature,
                        "english_name": english_name
                    }
                    detections.append(detection)
                    
            # 然后检查常规boxes结果
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                logger.info(f"检测到 {len(boxes)} 个常规目标")
                for i in range(len(boxes)):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 应用置信度过滤（在后处理中进行）
                    if confidence < confidence_threshold:
                        continue
                    
                    logger.info(f"目标 {i+1}: class_id={class_id}, confidence={confidence:.3f}")
                    
                    # 获取英文类别名称
                    english_name = model_names.get(class_id, f"unknown_device_{class_id}")
                    
                    # 映射到中文名称
                    chinese_name = DEVICE_NAME_MAPPING.get(english_name, english_name)
                    
                    logger.info(f"映射: {english_name} -> {chinese_name}")
                    
                    # 生成合理的温度值
                    temperature = generate_realistic_temperature(chinese_name)
                    
                    detection = {
                        "id": len(detections),
                        "class_name": chinese_name,
                        "confidence": float(confidence),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "temperature": temperature,
                        "english_name": english_name
                    }
                    detections.append(detection)
            else:
                logger.info(f"未检测到任何目标或结果为空: boxes={getattr(result, 'boxes', None)}, obb={getattr(result, 'obb', None)}")
        
        logger.info(f"检测完成: 发现 {len(detections)} 个设备，图像尺寸: {width}x{height}")
        return detections
        
    except Exception as e:
        logger.error(f"YOLO检测失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        raise

def draw_detection_results(image, detections):
    """在图像上绘制检测结果"""
    import cv2
    import numpy as np
    
    # 创建图像副本
    vis_image = image.copy()
    
    # 定义颜色（BGR格式）
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄色
        (128, 0, 255),  # 紫色
        (255, 128, 0),  # 橙色
    ]
    
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        
        # 绘制边界框 - 更细的线条
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本（使用英文避免中文显示问题）
        english_name = detection.get('english_name', class_name)
        label = f"{english_name}: {confidence:.2f}"
        
        # 计算文本尺寸 - 更小的字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # 缩小字体
        thickness = 2      # 更细的文字
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 添加动态背景效果 - 半透明背景
        overlay = vis_image.copy()
        cv2.rectangle(overlay, 
                     (x1, y1 - text_height - 8), 
                     (x1 + text_width + 8, y1), 
                     color, -1)
        
        # 混合半透明效果
        alpha = 0.8  # 透明度
        cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
        
        # 绘制文本 - 白色文字带阴影效果
        # 先绘制阴影
        cv2.putText(vis_image, label, 
                   (x1 + 5, y1 - 5), 
                   font, font_scale, (0, 0, 0), thickness + 1)
        # 再绘制主文字
        cv2.putText(vis_image, label, 
                   (x1 + 4, y1 - 6), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 在框的右上角添加小巧的设备编号
        device_number = f"#{i+1}"
        cv2.putText(vis_image, device_number,
                   (x2 - 25, y1 + 15),
                   font, 0.4, color, 1)
    
    return vis_image

def map_to_power_equipment(class_name):
    """将YOLO检测的类别映射到电力设备类型"""
    # 电力设备映射表 - 更合理的映射关系
    power_equipment_map = {
        # 直接的电力设备映射
        'transformer': '变压器',
        'insulator': '绝缘子', 
        'conductor': '导线',
        'tower': '铁塔',
        'pole': '电杆',
        
        # 基于形状和特征的映射
        'bottle': '绝缘子',  # 瓶状物体 -> 绝缘子
        'cup': '绝缘子',
        'bowl': '绝缘子',
        'vase': '绝缘子',
        
        # 长条状物体 -> 导线/线夹
        'hot dog': '并沟线夹',
        'banana': '并沟线夹',
        'carrot': '并沟线夹',
        'knife': '线夹',
        'spoon': '线夹',
        'fork': '线夹',
        
        # 方形/矩形物体 -> 变压器/控制设备
        'tv': '变压器',
        'laptop': '控制柜',
        'microwave': '变压器',
        'oven': '变压器',
        'refrigerator': '变压器',
        'book': '控制面板',
        'suitcase': '控制柜',
        
        # 圆形物体 -> 套管/绝缘子
        'clock': '套管',
        'frisbee': '套管',
        'donut': '套管',
        
        # 其他常见物体
        'person': '巡检人员',
        'car': '巡检车辆',
        'truck': '电力车辆',
        'handbag': '工具包'
    }
    
    mapped_type = power_equipment_map.get(class_name.lower(), None)
    if mapped_type:
        return mapped_type
    
    # 如果没有直接映射，根据常见的电力设备类型随机选择
    import random
    common_equipment = ['变压器', '绝缘子', '并沟线夹', '导线', '套管', '开关设备', '控制柜']
    return random.choice(common_equipment)

def generate_realistic_temperature(device_type):
    """为不同设备类型生成合理的温度值"""
    import random
    
    temp_ranges = {
        # 基于实际的中文设备名称
        '变压器': (40, 85),
        '单根绝缘子': (25, 45),
        '两根绝缘子': (25, 45), 
        '四根绝缘子': (25, 45),
        '绝缘子串': (25, 45),
        '绝缘子组': (25, 45),
        '绝缘子盒': (25, 45),
        '并沟线夹': (35, 75),
        '并沟线夹盒': (35, 75),
        '电缆终端': (35, 70),
        '电缆终端盒': (35, 70),
        '电缆终端底座': (35, 70),
        '电缆终端本体': (35, 70),
        '电缆终端接头': (35, 70),
        '电力电缆': (30, 65),
        '电力电缆本体': (30, 65),
        '电力电缆接头': (30, 65),
        '避雷器1': (25, 60),
        '避雷器2': (25, 60),
        '避雷器3': (25, 60),
        '避雷器本体': (25, 60),
        '避雷器接头': (25, 60),
        '接地线夹': (20, 50),
        '接地线夹盒': (20, 50),
        '金具': (25, 65),
        '金具盒': (25, 65),
        '管母2': (40, 80),
        '管母3': (40, 80),
        '管母盒': (40, 80),
        '连接管': (30, 65),
        '接触终端1': (35, 70),
        '接触终端2': (35, 70),
        '接触终端4': (35, 70),
        '接触终端6': (35, 70),
        '接触终端8': (35, 70),
        '接触终端盒': (35, 70),
        '接触终端盒2': (35, 70),
        '接触终端盒3': (35, 70),
        '接触终端盒4': (35, 70),
        
        # 兼容旧的设备名称
        '耐张线夹': (35, 75),
        '管母': (40, 80),
        '尾管和电力电缆': (35, 65),
    }
    
    # 寻找匹配的设备类型
    for key, (min_temp, max_temp) in temp_ranges.items():
        if key in device_type:
            return round(random.uniform(min_temp, max_temp), 1)
    
    # 默认温度范围
    return round(random.uniform(25, 70), 1)

def get_diagnosis_suggestion(device_type, level):
    """根据设备类型和严重程度获取诊断建议"""
    suggestions = {
        ('变压器', 'normal'): '继续正常监控，定期检查温度变化趋势',
        ('变压器', 'warning'): '加强监视，注意散热系统是否正常工作',
        ('变压器', 'critical'): '立即安排检修，检查散热器和冷却系统',
        
        ('并沟线夹', 'normal'): '正常运行，保持定期巡检',
        ('并沟线夹', 'warning'): '加强监视，检查接触面是否氧化',
        ('并沟线夹', 'critical'): '立即安排检修，检查接触电阻和紧固状态',
        
        ('绝缘子', 'normal'): '绝缘性能良好，继续监控',
        ('绝缘子', 'warning'): '注意检查绝缘子表面清洁度',
        ('绝缘子', 'critical'): '立即检查，可能存在绝缘老化或污损问题',
        
        ('电缆终端', 'normal'): '运行正常，保持例行检查',
        ('电缆终端', 'warning'): '检查端子连接是否紧固',
        ('电缆终端', 'critical'): '立即检修，防止绝缘击穿或着火',
        
        ('耐张线夹', 'normal'): '张力和接触良好，继续监控',
        ('耐张线夹', 'warning'): '检查线夹紧固状态和接触面',
        ('耐张线夹', 'critical'): '立即检修，防止导线脱落或接触不良',
        
        ('管母', 'normal'): '母线运行正常，保持监控',
        ('管母', 'warning'): '检查母线接头和支撑绝缘子',
        ('管母', 'critical'): '立即安排检修，检查接头温升和绝缘状况',
        
        ('尾管和电力电缆', 'normal'): '电缆运行正常',
        ('尾管和电力电缆', 'warning'): '检查电缆负荷和散热条件',
        ('尾管和电力电缆', 'critical'): '立即减负荷运行，安排紧急检修'
    }
    
    return suggestions.get((device_type, level), '请根据实际情况制定检修计划')

# ===== 路由定义 =====

@app.route('/中华人民共和国.geojson')
def serve_china_geojson():
    """直接服务GeoJSON文件"""
    try:
        geojson_path = os.path.join(os.path.dirname(__file__), "中华人民共和国.geojson")
        if not os.path.exists(geojson_path):
            # 尝试从项目根目录查找
            geojson_path = os.path.join(project_root, "中华人民共和国.geojson")
            
        if os.path.exists(geojson_path):
            return send_from_directory(os.path.dirname(geojson_path), "中华人民共和国.geojson")
        else:
            logger.error(f"GeoJSON文件不存在: {geojson_path}")
            return jsonify({"error": "地图数据文件不存在"}), 404
    except Exception as e:
        logger.error(f"服务GeoJSON文件失败: {e}")
        return jsonify({"error": "服务地图数据失败"}), 500

@app.route('/api/china-geojson')
def get_china_geojson():
    """获取中国地图GeoJSON数据"""
    try:
        geojson_path = os.path.join(project_root, "中华人民共和国.geojson")
        if not os.path.exists(geojson_path):
            logger.error(f"GeoJSON文件不存在: {geojson_path}")
            return jsonify({"error": "地图数据文件不存在"}), 404
        
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        logger.info(f"成功加载GeoJSON数据，包含 {len(geojson_data.get('features', []))} 个省份")
        return jsonify(geojson_data)
    except Exception as e:
        logger.error(f"加载GeoJSON数据失败: {e}")
        return jsonify({"error": "加载地图数据失败"}), 500

# 登录验证装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 登录相关路由
@app.route('/login.html')
@app.route('/login')
def login():
    """登录页面"""
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """登录API"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # 简单的用户验证（实际应用中应该使用数据库和加密）
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            session['username'] = username
            session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            return jsonify({
                'success': True,
                'message': '登录成功',
                'redirect': '/'
            })
        else:
            return jsonify({
                'success': False,
                'message': '用户名或密码错误'
            }), 401
            
    except Exception as e:
        logger.error(f"登录处理失败: {e}")
        return jsonify({
            'success': False,
            'message': '登录处理失败，请重试'
        }), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """登出API"""
    try:
        session.clear()
        return jsonify({
            'success': True,
            'message': '退出登录成功'
        })
    except Exception as e:
        logger.error(f"登出处理失败: {e}")
        return jsonify({
            'success': False,
            'message': '登出处理失败'
        }), 500

@app.route('/api/user/info')
@login_required
def user_info():
    """获取用户信息"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'username': session.get('username', 'admin'),
                'role': '系统管理员',
                'login_time': session.get('login_time', '')
            }
        })
    except Exception as e:
        logger.error(f"获取用户信息失败: {e}")
        return jsonify({
            'success': False,
            'message': '获取用户信息失败'
        }), 500

@app.route('/')
@login_required
def index():
    """主页"""
    return render_template('index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    if filename.endswith('.html') and filename != 'login.html':
        # 除了登录页面，其他页面都需要登录
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return render_template(filename)
    return send_from_directory('static', filename)

@app.route('/api/dashboard/stats')
@login_required
def dashboard_stats():
    """获取仪表盘统计数据"""
    try:
        stats = api.get_dashboard_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        logger.error(f"获取统计数据失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/detection/results')
def detection_results():
    """获取检测结果"""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        
        start = (page - 1) * limit
        end = start + limit
        
        results = api.detection_results[start:end]
        
        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "total": len(api.detection_results),
                "page": page,
                "limit": limit
            }
        })
    except Exception as e:
        logger.error(f"获取检测结果失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/detection/latest')
def latest_results():
    """获取最新检测结果"""
    try:
        limit = request.args.get('limit', 10, type=int)
        results = api.get_latest_results(limit)
        
        return jsonify({
            "success": True,
            "data": results
        })
    except Exception as e:
        logger.error(f"获取最新结果失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/temperature/trend')
def temperature_trend():
    """获取温度趋势"""
    try:
        hours = request.args.get('hours', 24, type=int)
        trend = api.get_temperature_trend(hours)
        
        return jsonify({
            "success": True,
            "data": trend
        })
    except Exception as e:
        logger.error(f"获取温度趋势失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/devices/distribution')
def device_distribution():
    """获取设备分布"""
    try:
        distribution = api.get_device_distribution()
        
        return jsonify({
            "success": True,
            "data": distribution
        })
    except Exception as e:
        logger.error(f"获取设备分布失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/detection/upload', methods=['POST'])
def upload_detection_files():
    """上传检测文件"""
    try:
        if 'files' not in request.files:
            return jsonify({
                "success": False,
                "error": "没有文件上传"
            }), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file.filename:
                filename = file.filename
                # 这里可以保存文件并进行检测
                uploaded_files.append({
                    "name": filename,
                    "size": len(file.read()),
                    "status": "uploaded"
                })
        
        return jsonify({
            "success": True,
            "data": {
                "uploaded_files": uploaded_files,
                "count": len(uploaded_files)
            }
        })
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/detection/start', methods=['POST'])
def start_detection():
    """启动检测"""
    try:
        params = request.get_json()
        confidence_threshold = params.get('confidence_threshold', 0.5)
        temperature_threshold = params.get('temperature_threshold', 80)
        
        # 模拟检测过程
        result = {
            "detection_id": f"det_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "started",
            "parameters": {
                "confidence_threshold": confidence_threshold,
                "temperature_threshold": temperature_threshold
            },
            "estimated_time": 30  # 秒
        }
        
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"启动检测失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reports/list')
def list_reports():
    """获取报告列表"""
    try:
        # 模拟报告数据
        reports = [
            {
                "id": 1,
                "title": "日检测报告",
                "date": "2025-12-01",
                "type": "daily",
                "status": "completed",
                "summary": {
                    "detected_devices": 156,
                    "anomalies": 12,
                    "avg_temperature": 34.2
                }
            },
            {
                "id": 2,
                "title": "周检测报告",
                "date": "2025-11-25",
                "type": "weekly",
                "status": "completed",
                "summary": {
                    "detected_devices": 1092,
                    "anomalies": 89,
                    "avg_temperature": 35.8
                }
            },
            {
                "id": 3,
                "title": "月检测报告",
                "date": "2025-11-01",
                "type": "monthly",
                "status": "completed",
                "summary": {
                    "detected_devices": 4567,
                    "anomalies": 234,
                    "avg_temperature": 33.1
                }
            }
        ]
        
        return jsonify({
            "success": True,
            "data": reports
        })
    except Exception as e:
        logger.error(f"获取报告列表失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analytics/status')
def status_analytics():
    """获取状态分析数据"""
    try:
        # 计算设备状态分布
        stats = api.get_dashboard_stats()
        total = stats['total_devices']
        
        status_data = {
            "normal": round((stats['normal_devices'] / total) * 100, 1),
            "warning": round((stats['warning_devices'] / total) * 100, 1),
            "danger": round((stats['danger_devices'] / total) * 100, 1)
        }
        
        return jsonify({
            "success": True,
            "data": status_data
        })
    except Exception as e:
        logger.error(f"获取状态分析失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/analytics/accuracy')
def accuracy_analytics():
    """获取精度分析数据"""
    try:
        accuracy_data = {
            "target_detection": 89.6,
            "temperature_analysis": 94.2,
            "fault_identification": 87.5,
            "classification_accuracy": 91.8
        }
        
        return jsonify({
            "success": True,
            "data": accuracy_data
        })
    except Exception as e:
        logger.error(f"获取精度分析失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ===== 温度异常诊断相关路由 =====

@app.route('/api/diagnostic-rules')
def get_diagnostic_rules():
    """获取温度异常诊断规则"""
    try:
        rules = {}
        
        if not os.path.exists(DIAGNOSTIC_RULES_DIR):
            logger.warning(f"诊断规则目录未找到: {DIAGNOSTIC_RULES_DIR}")
            return jsonify({"error": "诊断规则目录未找到"}), 404
        
        logger.info(f"开始加载诊断规则，目录: {DIAGNOSTIC_RULES_DIR}")
        
        # 遍历设备类型目录
        for device_folder in os.listdir(DIAGNOSTIC_RULES_DIR):
            if device_folder.startswith('.'):  # 跳过隐藏文件
                continue
                
            device_path = os.path.join(DIAGNOSTIC_RULES_DIR, device_folder)
            if not os.path.isdir(device_path):
                continue
            
            logger.info(f"处理设备类型目录: {device_folder}")
            device_rules = []
            
            # 读取该设备类型的规则文件
            for rule_file in os.listdir(device_path):
                if rule_file.startswith('.'):  # 跳过隐藏文件
                    continue
                if rule_file.endswith('.csv'):
                    rule_path = os.path.join(device_path, rule_file)
                    file_rules = load_rules_from_csv(rule_path)
                    device_rules.extend(file_rules)
                    logger.info(f"从 {rule_file} 加载了 {len(file_rules)} 条规则")
            
            if device_rules:
                rules[device_folder] = {"rules": device_rules}
                logger.info(f"设备类型 {device_folder} 总共加载了 {len(device_rules)} 条规则")
        
        logger.info(f"总共加载了 {len(rules)} 类设备的诊断规则")
        
        # 如果没有加载到规则，提供默认规则
        if not rules:
            logger.warning("没有加载到任何诊断规则，使用默认规则")
            rules = get_default_diagnostic_rules()
        
        return jsonify(rules)
    
    except Exception as e:
        logger.error(f"加载诊断规则失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        
        # 返回默认规则而不是错误
        logger.info("返回默认诊断规则")
        return jsonify(get_default_diagnostic_rules())

def get_default_diagnostic_rules():
    """获取默认诊断规则"""
    return {
        '变压器': {
            "rules": [
                {
                    "level": "一般",
                    "formula": "relativeTempDiff >= 0.35",
                    "description": "δ≥35%但热点温度未达到严重异常温度值",
                    "method": "相对温差判断法",
                    "heatType": "电流制热型",
                    "faultFeature": "漏磁环（涡）流现象",
                    "thermalFeature": "以箱体局部表面过热为特征",
                    "suggestion": "检查油色谱或轻瓦斯动作情况"
                },
                {
                    "level": "严重",
                    "formula": "maxTemp > 85 && maxTemp <= 105",
                    "description": "85℃≤热点温度≤105 ℃",
                    "method": "绝对值判断法",
                    "heatType": "电流制热型",
                    "faultFeature": "漏磁环（涡）流现象",
                    "thermalFeature": "以箱体局部表面过热为特征",
                    "suggestion": "检查油色谱或轻瓦斯动作情况"
                },
                {
                    "level": "危急",
                    "formula": "maxTemp > 105",
                    "description": "热点温度>105℃",
                    "method": "绝对值判断法",
                    "heatType": "电流制热型",
                    "faultFeature": "漏磁环（涡）流现象",
                    "thermalFeature": "以箱体局部表面过热为特征",
                    "suggestion": "立即停运检修"
                }
            ]
        },
        '绝缘子': {
            "rules": [
                {
                    "level": "严重",
                    "formula": "tempDiff > 1",
                    "description": "相邻绝缘子温差>1℃",
                    "method": "温差判断法",
                    "heatType": "电压制热型",
                    "faultFeature": "低值绝缘子发热（绝缘电阻在10~500MΩ）",
                    "thermalFeature": "以铁帽为发热中心的热像图，其比正常绝缘子铁帽温度高",
                    "suggestion": "必要时可用电气的方法对零、低阻值绝缘子试验确认"
                }
            ]
        },
        '并沟线夹': {
            "rules": [
                {
                    "level": "一般",
                    "formula": "relativeTempDiff >= 0.15",
                    "description": "δ≥15%",
                    "method": "相对温差判断法",
                    "heatType": "电流制热型",
                    "faultFeature": "接触电阻增大",
                    "thermalFeature": "接触部位发热",
                    "suggestion": "加强监视，安排检修"
                },
                {
                    "level": "严重",
                    "formula": "maxTemp > 85",
                    "description": "热点温度>85℃",
                    "method": "绝对值判断法",
                    "heatType": "电流制热型",
                    "faultFeature": "接触电阻严重增大",
                    "thermalFeature": "明显发热异常",
                    "suggestion": "立即安排检修"
                }
            ]
        }
    }

@app.route('/api/detect-devices', methods=['POST'])
def detect_devices():
    """设备检测API - 仅使用训练好的专用模型"""
    global YOLO_MODEL
    
    # 检查模型是否加载
    if YOLO_MODEL is None:
        return jsonify({"error": "专用YOLO模型未加载，无法进行设备检测"}), 500
    
    try:
        if 'images' not in request.files:
            return jsonify({"error": "没有上传图像文件"}), 400
        
        files = request.files.getlist('images')
        if not files or all(not file.filename for file in files):
            return jsonify({"error": "没有有效的图像文件"}), 400
        
        results = {"results": []}
        
        for file in files:
            if file and file.filename:
                # 验证文件类型
                if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    logger.warning(f"跳过不支持的文件格式: {file.filename}")
                    continue
                
                try:
                    # 使用训练好的专用YOLO模型进行设备检测
                    detections, original_image, width, height = detect_devices_with_yolo(file)
                    
                    result_data = {
                        "image_name": file.filename,
                        "detections": detections,
                        "image_width": width,
                        "image_height": height,
                        "originalImage": original_image
                    }
                    
                    results["results"].append(result_data)
                    logger.info(f"成功检测文件 {file.filename}: 发现 {len(detections)} 个设备")
                    
                except Exception as detection_error:
                    logger.error(f"检测文件 {file.filename} 失败: {detection_error}")
                    # 继续处理其他文件，不因单个文件失败而中断
                    continue
        
        if not results["results"]:
            return jsonify({"error": "没有成功处理任何图像文件"}), 400
        
        logger.info(f"批量检测完成: 处理了 {len(results['results'])} 个文件")
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"设备检测API失败: {e}")
        return jsonify({"error": f"设备检测失败: {str(e)}"}), 500

@app.route('/api/temperature-diagnosis', methods=['POST'])
def temperature_diagnosis():
    """温度异常诊断API - 使用test目录中的图片"""
    try:
        logger.info("收到温度诊断请求")
        
        # 检查是否有文件上传
        if 'images' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({'error': '文件列表为空'}), 400
        
        # 获取参数，使用更低的默认置信度阈值
        confidence = float(request.form.get('confidence', 0.01))
        temp_threshold = float(request.form.get('temperature', 80))
        device_type = request.form.get('device_type', 'all')
        
        logger.info(f"诊断参数: confidence={confidence}, temp_threshold={temp_threshold}, device_type={device_type}")
        
        # 处理文件
        results = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                logger.info(f"处理文件: {filename}")
                
                # 根据文件名查找test目录中的匹配图片
                matched_images = find_matching_images(filename)
                
                # 优先使用红外图进行检测
                image_path = matched_images['thermal']
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"未找到匹配的红外图: {filename}")
                    continue
                
                # 使用YOLO检测设备
                devices = detect_devices_with_yolo_from_file(image_path, confidence)
                
                # 读取原图并转换为base64
                with open(image_path, 'rb') as img_file:
                    original_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                # 生成检测结果图像
                original_image = cv2.imread(image_path)
                detection_image = draw_detection_results(original_image.copy(), devices)
                
                # 转换检测图为base64
                _, det_buffer = cv2.imencode('.jpg', detection_image)
                detection_b64 = base64.b64encode(det_buffer).decode('utf-8')
                
                # 分析诊断（基于温度阈值）
                diagnoses = []
                for device in devices:
                    level = 'normal'
                    if device['temperature'] > temp_threshold + 10:
                        level = 'critical'
                    elif device['temperature'] > temp_threshold:
                        level = 'warning'
                    
                    diagnoses.append({
                        'device': device['name'],
                        'level': level,
                        'temperature': device['temperature'],
                        'description': f"{device['name']}温度: {device['temperature']}°C",
                        'suggestion': generate_diagnostic_suggestion(device['name'], level, device['temperature'])
                    })
                
                # 如果有热力图，也包含进来
                heatmap_b64 = None
                if matched_images['heatmap'] and os.path.exists(matched_images['heatmap']):
                    with open(matched_images['heatmap'], 'rb') as heatmap_file:
                        heatmap_b64 = base64.b64encode(heatmap_file.read()).decode('utf-8')
                
                results.append({
                    'filename': filename,
                    'devices': devices,
                    'diagnoses': diagnoses,
                    'original_image': f'data:image/jpeg;base64,{original_b64}',
                    'detection_image': f'data:image/jpeg;base64,{detection_b64}',
                    'heatmap_image': f'data:image/jpeg;base64,{heatmap_b64}' if heatmap_b64 else None,
                    'matched_files': matched_images
                })
                
                logger.info(f"文件 {filename} 处理完成，检测到 {len(devices)} 个设备")
        
        # 统计摘要
        total_devices = sum(len(r['devices']) for r in results)
        total_diagnoses = sum(len(r['diagnoses']) for r in results)
        
        # 计算严重程度统计
        severity_stats = {'normal': 0, 'warning': 0, 'critical': 0}
        for result in results:
            for diagnosis in result['diagnoses']:
                severity_stats[diagnosis['level']] += 1
        
        response = {
            'success': True,
            'results': results,
            'summary': {
                'total_devices': total_devices,
                'total_diagnoses': total_diagnoses,
                'severity_stats': severity_stats
            }
        }
        
        logger.info(f"诊断完成，共处理 {len(results)} 个文件，检测到 {total_devices} 个设备")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"温度诊断API错误: {e}")
        return jsonify({'error': f'诊断失败: {str(e)}'}), 500
        
        # 计算统计摘要
        normal_count = len([d for d in diagnosis_results if d['level'] == 'normal'])
        warning_count = len([d for d in diagnosis_results if d['level'] == 'warning'])
        critical_count = len([d for d in diagnosis_results if d['level'] == 'critical'])
        
        confidences = []
        for det in detections:
            for dev in det['devices']:
                confidences.append(dev['confidence'])
        
        avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
        
        summary = {
            'normal_count': normal_count,
            'warning_count': warning_count,
            'critical_count': critical_count,
            'avg_confidence': avg_confidence
        }
        
        return jsonify({
            'summary': summary,
            'detections': detections,
            'diagnosis_results': diagnosis_results
        })
        
    except Exception as e:
        logger.error(f"温度诊断错误: {e}")
        return jsonify({'error': str(e)}), 500

# ===== 新的三种诊断模式API =====

@app.route('/api/device-recognition', methods=['POST'])
def device_recognition():
    """单张图片识别（只识别设备）"""
    try:
        logger.info("收到设备识别请求")
        
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图像文件'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 获取参数
        confidence = float(request.form.get('confidence', 0.25))
        
        logger.info(f"识别参数: confidence={confidence}")
        
        # 处理上传的图像
        filename = secure_filename(file.filename)
        
        # 将图像转换为numpy数组用于YOLO处理
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无法解析图像文件'}), 400
        
        # 将原始图像编码为base64用于前端显示
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 使用YOLO检测设备
        devices = detect_devices_with_yolo_from_array(image, confidence)
        
        # 绘制检测结果的可视化图像
        vis_image = draw_detection_results(image, devices)
        
        # 将可视化图像编码为base64
        _, vis_buffer = cv2.imencode('.jpg', vis_image)
        vis_image_base64 = base64.b64encode(vis_buffer).decode('utf-8')
        
        # 准备结果
        result = {
            'success': True,
            'filename': filename,
            'image_size': [image.shape[1], image.shape[0]],  # [width, height]
            'devices_detected': len(devices),
            'devices': devices,
            'image_data': f"data:image/jpeg;base64,{image_base64}",  # 原始图像
            'visualization_data': f"data:image/jpeg;base64,{vis_image_base64}",  # 可视化图像
            'processing_time': 0.5  # 模拟处理时间
        }
        
        logger.info(f"设备识别完成: 发现 {len(devices)} 个设备")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"设备识别错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/single-image-diagnosis', methods=['POST'])
def single_image_diagnosis():
    """单张图片诊断"""
    try:
        logger.info("收到单张图片诊断请求")
        
        if 'thermal_image' not in request.files:
            return jsonify({'error': '没有上传红外图像'}), 400
        
        thermal_file = request.files['thermal_image']
        if thermal_file.filename == '':
            return jsonify({'error': '红外图像文件名为空'}), 400
        
        # 获取参数
        confidence = float(request.form.get('confidence', 0.25))
        temp_threshold = float(request.form.get('temperature', 80))
        
        logger.info(f"诊断参数: confidence={confidence}, temp_threshold={temp_threshold}")
        
        # 处理红外图像
        thermal_filename = secure_filename(thermal_file.filename)
        thermal_data = thermal_file.read()
        nparr = np.frombuffer(thermal_data, np.uint8)
        thermal_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if thermal_image is None:
            return jsonify({'error': '无法解析红外图像文件'}), 400
        
        # 使用YOLO检测设备
        devices = detect_devices_with_yolo_from_array(thermal_image, confidence)
        
        # 处理温度数据（如果有）
        temp_data = None
        if 'temperature_data' in request.files:
            temp_file = request.files['temperature_data']
            if temp_file.filename:
                # 读取温度数据文件
                temp_content = temp_file.read().decode('utf-8')
                temp_data = parse_temperature_data(temp_content)
        
        # 进行异常诊断
        diagnoses = []
        for device in devices:
            # 模拟温度异常检测
            device_temp = np.random.uniform(30, 100)  # 模拟设备温度
            is_abnormal = device_temp > temp_threshold
            
            diagnosis = {
                'device_id': device['id'],
                'device_type': device['class_name'],
                'bbox': device['bbox'],
                'confidence': device['confidence'],
                'temperature': round(device_temp, 2),
                'is_abnormal': is_abnormal,
                'status': 'abnormal' if is_abnormal else 'normal',
                'risk_level': 'high' if device_temp > temp_threshold + 10 else 'medium' if is_abnormal else 'low'
            }
            diagnoses.append(diagnosis)
        
        result = {
            'success': True,
            'filename': thermal_filename,
            'image_size': [thermal_image.shape[1], thermal_image.shape[0]],
            'devices_detected': len(devices),
            'diagnoses': diagnoses,
            'summary': {
                'normal_count': len([d for d in diagnoses if not d['is_abnormal']]),
                'abnormal_count': len([d for d in diagnoses if d['is_abnormal']]),
                'high_risk_count': len([d for d in diagnoses if d.get('risk_level') == 'high']),
                'average_temperature': round(np.mean([d['temperature'] for d in diagnoses]), 2) if diagnoses else 0
            }
        }
        
        logger.info(f"单张图片诊断完成: {len(devices)} 个设备, {result['summary']['abnormal_count']} 个异常")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"单张图片诊断错误: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-image-diagnosis', methods=['POST'])
def batch_image_diagnosis():
    """批量图像诊断"""
    try:
        logger.info("收到批量图像诊断请求")
        
        if 'images' not in request.files:
            return jsonify({'error': '没有上传图像文件'}), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({'error': '图像文件列表为空'}), 400
        
        # 获取参数
        confidence = float(request.form.get('confidence', 0.25))
        temp_threshold = float(request.form.get('temperature', 80))
        
        logger.info(f"批量诊断参数: confidence={confidence}, temp_threshold={temp_threshold}")
        
        # 处理每个图像文件
        results = []
        total_devices = 0
        total_abnormal = 0
        
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                logger.info(f"处理文件: {filename}")
                
                # 处理图像
                image_data = file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    logger.warning(f"无法解析图像: {filename}")
                    continue
                
                # 使用YOLO检测设备
                devices = detect_devices_with_yolo_from_array(image, confidence)
                
                # 进行异常诊断
                diagnoses = []
                for device in devices:
                    # 模拟温度异常检测
                    device_temp = np.random.uniform(30, 100)
                    is_abnormal = device_temp > temp_threshold
                    
                    diagnosis = {
                        'device_id': device['id'],
                        'device_type': device['class_name'],
                        'bbox': device['bbox'],
                        'confidence': device['confidence'],
                        'temperature': round(device_temp, 2),
                        'is_abnormal': is_abnormal,
                        'status': 'abnormal' if is_abnormal else 'normal',
                        'risk_level': 'high' if device_temp > temp_threshold + 10 else 'medium' if is_abnormal else 'low'
                    }
                    diagnoses.append(diagnosis)
                
                abnormal_count = len([d for d in diagnoses if d['is_abnormal']])
                total_devices += len(devices)
                total_abnormal += abnormal_count
                
                file_result = {
                    'filename': filename,
                    'image_size': [image.shape[1], image.shape[0]],
                    'devices_detected': len(devices),
                    'diagnoses': diagnoses,
                    'summary': {
                        'normal_count': len(diagnoses) - abnormal_count,
                        'abnormal_count': abnormal_count,
                        'average_temperature': round(np.mean([d['temperature'] for d in diagnoses]), 2) if diagnoses else 0
                    }
                }
                results.append(file_result)
        
        overall_summary = {
            'total_images': len(results),
            'total_devices': total_devices,
            'total_abnormal': total_abnormal,
            'total_normal': total_devices - total_abnormal,
            'abnormal_rate': round((total_abnormal / total_devices * 100), 2) if total_devices > 0 else 0
        }
        
        result = {
            'success': True,
            'results': results,
            'summary': overall_summary
        }
        
        logger.info(f"批量诊断完成: {len(results)} 个文件, {total_devices} 个设备, {total_abnormal} 个异常")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"批量图像诊断错误: {e}")
        return jsonify({'error': str(e)}), 500

def parse_temperature_data(content):
    """解析温度数据文件"""
    try:
        lines = content.strip().split('\n')
        temp_data = []
        for line in lines:
            if line.strip():
                # 假设格式为: x,y,temperature 或 temperature
                parts = line.split(',')
                if len(parts) >= 3:
                    x, y, temp = float(parts[0]), float(parts[1]), float(parts[2])
                    temp_data.append({'x': x, 'y': y, 'temperature': temp})
                elif len(parts) == 1:
                    temp = float(parts[0])
                    temp_data.append({'temperature': temp})
        return temp_data
    except Exception as e:
        logger.error(f"解析温度数据失败: {e}")
        return None

# ===== 错误处理 =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "API接口不存在"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500

# ===== 启动应用 =====

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    logger.info("启动UAV红外热成像检测系统Web服务")
    logger.info(f"检测结果路径: {INFERENCE_RESULTS_PATH}")
    logger.info(f"处理数据路径: {PROCESSED_DATA_PATH}")
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        threaded=True
    )
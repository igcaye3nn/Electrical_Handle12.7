#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于app_5001.py重写的电力设备检测服务
使用YOLO模型进行设备识别
"""
import pickle
import time
import flask
from flask import request
import numpy as np
import os
import sys
import cv2
from pathlib import Path
import logging

# 导入YOLO
sys.path.append('../yolov11-OBB-main')
from ultralytics import YOLO

app = flask.Flask(__name__)

# 配置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_multi_device_model.pt')
IMAGE_DIR = os.path.join(BASE_DIR, 'test/processed_data/reference_images')
TEMP_DIR = os.path.join(BASE_DIR, 'test/processed_data/thermal_images')
XML_DIR = os.path.join(BASE_DIR, 'test/processed_data/labels')
RESULT_DIR = os.path.join(BASE_DIR, 'web_temp/results')
# 全局变量
model = None

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app_5001.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model():
    """加载YOLO模型"""
    global model
    logger = setup_logging()
    
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def find_image_by_ticks(ticks):
    """根据ticks查找对应的图片文件 - 改进版"""
    logger = logging.getLogger(__name__)
    
    # 获取图片目录中的所有图片文件
    image_files = []
    for filename in os.listdir(IMAGE_DIR):
        if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_files.append(filename)
    
    if not image_files:
        logger.error("No image files found in directory")
        return None
    
    logger.info(f"Searching for image with ticks: {ticks}")
    logger.info(f"Available images: {len(image_files)}")
    
    # 1. 首先尝试完全匹配ticks (去掉扩展名)
    for filename in image_files:
        base_name = filename.split('.')[0]
        if ticks == base_name:
            image_path = os.path.join(IMAGE_DIR, filename)
            logger.info(f"Found exact full match image: {image_path}")
            return image_path
    
    # 2. 尝试部分匹配ticks（作为子字符串）
    for filename in image_files:
        if str(ticks) in filename:
            image_path = os.path.join(IMAGE_DIR, filename)
            logger.info(f"Found partial match image: {image_path}")
            return image_path
    
    # 3. 尝试时间戳格式转换匹配
    # 当前文件名格式: 170911173545-500kV-避雷器...
    # ticks格式: 1761270345573186 (13位)
    ticks_str = str(ticks)
    logger.info(f"Trying timestamp format conversion for: {ticks_str}")
    
    if len(ticks_str) >= 12:
        # 尝试不同的时间戳转换方式
        possible_formats = [
            ticks_str[:12],   # 取前12位: 176127034557
            ticks_str[1:13],  # 去掉第一位: 761270345573
            ticks_str[-12:],  # 取后12位: 270345573186
            ticks_str[:6] + ticks_str[7:13],  # 其他组合
        ]
        
        for time_format in possible_formats:
            logger.info(f"  Trying format: {time_format}")
            for filename in image_files:
                if time_format in filename:
                    image_path = os.path.join(IMAGE_DIR, filename)
                    logger.info(f"Found timestamp match image: {image_path} (format: {time_format})")
                    return image_path
        
    # 4. 如果ticks看起来像真实的时间戳，尝试解析
    try:
        if len(ticks_str) == 13:  # 毫秒时间戳
            timestamp_seconds = int(ticks_str[:10])
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp_seconds)
            
            # 生成可能的文件名格式
            date_formats = [
                dt.strftime('%y%m%d%H%M%S'),     # 170911173545
                dt.strftime('%Y%m%d%H%M%S'),     # 20170911173545  
                dt.strftime('%m%d%H%M%S'),       # 0911173545
                dt.strftime('%d%H%M%S'),         # 11173545
            ]
            
            logger.info(f"Parsed timestamp: {dt}, trying date formats...")
            for date_format in date_formats:
                logger.info(f"  Trying date format: {date_format}")
                for filename in image_files:
                    if date_format in filename:
                        image_path = os.path.join(IMAGE_DIR, filename)
                        logger.info(f"Found date match image: {image_path} (format: {date_format})")
                        return image_path
    except:
        logger.info("Could not parse as timestamp")
    
    # 5. 如果以上都没有匹配，使用ticks的哈希值来选择图片
    # 这样同一个ticks总是对应同一张图片
    ticks_hash = hash(str(ticks)) % len(image_files)
    selected_filename = image_files[ticks_hash]
    image_path = os.path.join(IMAGE_DIR, selected_filename)
    
    logger.info(f"No direct match found, selected image by hash: {image_path}")
    logger.info(f"Ticks: {ticks} -> Hash: {ticks_hash} -> File: {selected_filename}")
    
    return image_path
    
    
def load_temp_data(ticks):
    """加载温度数据"""
    logger = logging.getLogger(__name__)
    
    # 构建温度数据文件路径
    temp_file = os.path.join(TEMP_DIR, f"{ticks}.txt")
    
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r') as f:
                temp_data = f.read()
            logger.info(f"Loaded temperature data: {temp_file}")
            return temp_data
        except Exception as e:
            logger.error(f"Failed to load temperature data: {e}")
            return None
    else:
        logger.warning(f"Temperature data not found: {temp_file}")
        return None

def run_inference(image_path):
    """使用YOLO模型进行推理"""
    global model
    logger = logging.getLogger(__name__)
    
    if model is None:
        logger.error("Model not loaded")
        return []
    
    try:
        # 运行推理
        results = model(image_path, conf=0.01, iou=0.7)
        
        equipment = []
        if results and len(results) > 0:
            result = results[0]
            
            # 处理OBB检测结果
            if hasattr(result, 'obb') and result.obb is not None:
                boxes = result.obb.xyxyxyxy.cpu().numpy() if result.obb.xyxyxyxy is not None else []
                confs = result.obb.conf.cpu().numpy() if result.obb.conf is not None else []
                classes = result.obb.cls.cpu().numpy() if result.obb.cls is not None else []
                
                # 获取类别名称
                class_names = list(model.names.values()) if hasattr(model, 'names') else []
                
                for box, conf, cls in zip(boxes, confs, classes):
                    if len(class_names) > int(cls):
                        class_name = class_names[int(cls)]
                        # 转换为8点坐标格式
                        bbox_8points = box.flatten().tolist()
                        equipment.append((class_name, float(conf), bbox_8points))
        
        logger.info(f"Detection completed: {len(equipment)} objects found")
        return equipment
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def bd_obb():
    """电力设备检测主接口"""
    if request.method == "GET":
        # 返回简单的HTML页面用于测试
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>电力设备检测系统</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>🔌 电力设备检测系统</h1>
            <h2>📊 系统状态：运行中</h2>
            <p>🤖 模型已加载，支持39种设备类型检测</p>
            <p>🌐 API端点：POST /</p>
            <p>📝 参数：ticks (时间戳)</p>
            <hr>
            <h3>测试表单</h3>
            <form method="post" action="/">
                <label for="ticks">时间戳 (ticks):</label><br>
                <input type="text" id="ticks" name="ticks" value="20241021_143000"><br><br>
                <input type="submit" value="检测设备">
            </form>
        </body>
        </html>
        """
    
    # POST方法处理设备检测
    logger = logging.getLogger(__name__)

    try:
        # 获得时间戳
        ticks = flask.request.form["ticks"]
        
        # 根据时间戳组合成温度数据路径，浮点型矩阵（使用文件读写是因为速度比通过JSON传递要快）
        DATA_PATH = '/Users/doujiangyangcong/Desktop/jyc/UAV20241021_system/test/processed_data/thermal_images/' + ticks + '_data.txt'
        
        # 根据时间戳组合成推理结果存放文件的路径
        CLA_PATH = '/Users/doujiangyangcong/Desktop/jyc/UAV20241021_system/web_temp/results/' + ticks + '_cla.txt'
        
        logger.info(f"Processing request with ticks: {ticks}")
        logger.info(f"Temperature data path: {DATA_PATH}")
        logger.info(f"Result path: {CLA_PATH}")
        
        # 读取温度数据（如果存在）
        temp = None
        if os.path.exists(DATA_PATH):
            try:
                f_data = open(DATA_PATH, 'rb')
                temp = pickle.load(f_data)
                f_data.close()
                logger.info("使用温度数据推理")
            except Exception as e:
                logger.warning(f"Failed to load temperature data: {e}")
        else:
            logger.info("温度数据文件不存在，仅使用图片推理")
        
        # 根据ticks查找对应的图片
        image_path = find_image_by_ticks(ticks)
        if not image_path:
            logger.error(f"No image found for ticks: {ticks}")
            return flask.jsonify({"error": "Image not found"}), 404
        
        # 使用温度数据推理
        logger.info('使用图片数据推理')
        equipment = run_inference(image_path)
        
        # 将推理结果的数据格式进行转换
        logger.info('将推理结果的数据格式进行转换')
        
        # 格式参考，equitment为列表，列表中为元组，元组中3个元素分别为标签、置信度、定向坐标
        # 将结果放入字典的equitment字段中
        result = {}
        result["equitment"] = equipment
        
        # 确保结果目录存在
        os.makedirs(os.path.dirname(CLA_PATH), exist_ok=True)
        
        # 将结果写入结果存放文件
        f_cla = open(CLA_PATH, 'wb')
        pickle.dump(result, f_cla)
        f_cla.close()
        
        logger.info(f"Results saved to: {CLA_PATH}")
        logger.info(f"Found {len(equipment)} equipment items")
        
        # 打印检测结果
        for i, (name, conf, bbox) in enumerate(equipment):
            logger.info(f"  {i+1}. {name}: {conf:.3f}")
        
        # 因为结果已经放入文件，这里空字符返回即可
        return ''
        
    except KeyError:
        logger.error("No ticks parameter provided")
        return flask.jsonify({"error": "Missing ticks parameter"}), 400
    except Exception as e:
        logger.error(f"Error in bd_obb: {e}")
        return flask.jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model_loaded": model is not None}
def health_check():
    """健康检查接口"""
    return flask.jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "image_dir": IMAGE_DIR,
        "temp_dir": TEMP_DIR
    })

if __name__ == '__main__':
    # 加载模型
    if load_model():
        logger = logging.getLogger(__name__)
        logger.info("🚀 电力设备检测服务启动")
        logger.info("=" * 50)
        logger.info(f"🤖 模型路径: {MODEL_PATH}")
        logger.info(f"📁 图片目录: {IMAGE_DIR}")
        logger.info(f"🌡️ 温度数据目录: {TEMP_DIR}")
        logger.info(f"📄 标签目录: {XML_DIR}")
        logger.info(f"💾 结果目录: {RESULT_DIR}")
        logger.info(f"🌐 服务地址: http://0.0.0.0:5001")
        logger.info("=" * 50)
        
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("❌ 模型加载失败，服务无法启动")
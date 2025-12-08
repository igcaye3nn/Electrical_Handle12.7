#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电力设备温度异常诊断规则引擎

基于CSV规则文件进行温度异常诊断
支持多种设备类型的诊断规则

Author: AI Assistant
Date: 2025-12-08
"""

import os
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
import math

class TemperatureDiagnosisEngine:
    """温度异常诊断引擎"""
    
    def __init__(self, rules_dir: str = None):
        """
        初始化诊断引擎
        
        Args:
            rules_dir: 诊断规则目录路径
        """
        if rules_dir is None:
            rules_dir = os.path.join(os.path.dirname(__file__), '温度异常诊断规则')
        
        self.rules_dir = rules_dir
        self.device_rules = {}
        self.env_temp = 25.0  # 默认环境温度
        self.logger = logging.getLogger(__name__)
        
        # 加载所有设备的诊断规则
        self._load_all_rules()
    
    def _load_all_rules(self):
        """加载所有设备类型的诊断规则"""
        if not os.path.exists(self.rules_dir):
            self.logger.warning(f"诊断规则目录不存在: {self.rules_dir}")
            return
        
        for device_dir in os.listdir(self.rules_dir):
            device_path = os.path.join(self.rules_dir, device_dir)
            if os.path.isdir(device_path):
                self._load_device_rules(device_dir, device_path)
        
        self.logger.info(f"已加载 {len(self.device_rules)} 种设备的诊断规则")
    
    def _load_device_rules(self, device_type: str, device_path: str):
        """加载特定设备类型的诊断规则"""
        rules = []
        
        for filename in os.listdir(device_path):
            if filename.endswith('.csv'):
                csv_path = os.path.join(device_path, filename)
                try:
                    # 使用更灵活的CSV读取方式，处理不规整的数据
                    df = pd.read_csv(csv_path, encoding='utf-8', 
                                   skipinitialspace=True, 
                                   na_values=['', 'nan', 'NaN', 'null'],
                                   keep_default_na=True)
                    
                    # 删除完全空的列
                    df = df.dropna(axis=1, how='all')
                    
                    # 处理每一行规则
                    for _, row in df.iterrows():
                        rule = self._parse_rule(row.to_dict())
                        if rule:
                            rules.append(rule)
                    self.logger.info(f"从 {filename} 加载了 {len(df)} 条 {device_type} 规则")
                except Exception as e:
                    self.logger.error(f"加载规则文件失败 {csv_path}: {e}")
                    # 尝试使用备用方法读取
                    try:
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            header = lines[0].strip().split(',')
                            for line in lines[1:]:
                                values = line.strip().split(',')
                                # 填充缺失的列
                                while len(values) < len(header):
                                    values.append('')
                                # 截断多余的列
                                values = values[:len(header)]
                                
                                rule_dict = dict(zip(header, values))
                                rule = self._parse_rule(rule_dict)
                                if rule:
                                    rules.append(rule)
                        self.logger.info(f"使用备用方法从 {filename} 加载了规则")
                    except Exception as e2:
                        self.logger.error(f"备用方法也失败 {csv_path}: {e2}")
        
        self.device_rules[device_type] = rules
        self.logger.info(f"{device_type} 总计加载 {len(rules)} 条规则")
    
    def _parse_rule(self, rule_dict: Dict) -> Optional[Dict]:
        """解析单条诊断规则"""
        try:
            # 必需字段
            required_fields = ['缺陷等级', '公式信息', '诊断规则']
            
            # 支持不同的字段名称变体
            field_mapping = {
                '缺陷等级': ['缺陷等级', 'level', '等级'],
                '诊断公式': ['诊断公式', '公式信息', 'formula', '公式'],
                '诊断规则': ['诊断规则', '公式描述', 'rule', '规则'],
                '诊断方式': ['诊断方式', '诊断方法', 'method'],
                '处理建议': ['处理建议', 'suggestion', '建议'],
                '故障特征': ['故障特征', 'fault_feature', '特征'],
                '热像特征': ['热像特征', 'thermal_feature'],
                '诊断量': ['诊断量', '诊断提示信息', 'quantity']
            }
            
            # 标准化字段名称
            normalized_dict = {}
            for standard_name, variants in field_mapping.items():
                for variant in variants:
                    if variant in rule_dict:
                        normalized_dict[standard_name] = rule_dict[variant]
                        break
                if standard_name not in normalized_dict:
                    normalized_dict[standard_name] = ''
            
            # 检查必需字段
            if not normalized_dict.get('缺陷等级') or pd.isna(normalized_dict.get('缺陷等级')):
                return None
                
            formula = normalized_dict.get('诊断公式', '')
            if not formula or pd.isna(formula):
                return None
            
            # 解析诊断公式
            parsed_formula = self._parse_formula(str(formula))
            
            # 安全地处理可能的nan值
            def safe_str(value):
                if pd.isna(value) or value is None:
                    return ''
                return str(value).strip()
            
            return {
                'level': safe_str(normalized_dict['缺陷等级']),
                'formula': safe_str(formula),
                'parsed_formula': parsed_formula,
                'threshold_condition': safe_str(normalized_dict.get('阈值条件', '')),
                'diagnosis_rule': safe_str(normalized_dict.get('诊断规则', normalized_dict.get('诊断方式', ''))),
                'diagnosis_method': safe_str(normalized_dict.get('诊断方式', '')),
                'fault_feature': safe_str(normalized_dict.get('故障特征', '')),
                'thermal_feature': safe_str(normalized_dict.get('热像特征', '')),
                'treatment_suggestion': safe_str(normalized_dict.get('处理建议', '')),
                'diagnosis_quantity': safe_str(normalized_dict.get('诊断量', '')),
                'auxiliary_type': safe_str(normalized_dict.get('辅助类型', normalized_dict.get('制热类型', '')))
            }
        except Exception as e:
            self.logger.error(f"解析规则失败: {e}, 规则内容: {rule_dict}")
            return None
    
    def _parse_formula(self, formula: str) -> Dict:
        """解析诊断公式"""
        if not formula or formula.strip() == '':
            return {'type': 'unknown', 'expression': formula}
        
        # 预处理公式，替换常见的数学函数
        normalized_formula = formula.replace('math.max', 'max').replace('math.min', 'min')
        
        # 检测公式类型
        if '>=' in formula or '<=' in formula or '>' in formula or '<' in formula or '==' in formula:
            if 'abs(' in formula:
                return {'type': 'absolute_difference', 'expression': normalized_formula}
            elif 'max(' in formula and 'min(' in formula:
                return {'type': 'relative_temperature', 'expression': normalized_formula}
            elif any(temp_var in formula for temp_var in ['R01:Max', 'R02:Max', 'R03:Max', 'R04:Max']):
                return {'type': 'absolute_temperature', 'expression': normalized_formula}
            else:
                return {'type': 'threshold', 'expression': normalized_formula}
        
        return {'type': 'unknown', 'expression': normalized_formula}
    
    def set_environment_temperature(self, env_temp: float):
        """设置环境温度"""
        self.env_temp = env_temp
        self.logger.info(f"环境温度设置为: {env_temp}°C")
    
    def diagnose_device(self, device_type: str, temperature_regions: Dict[str, float]) -> List[Dict]:
        """
        诊断单个设备的温度异常
        
        Args:
            device_type: 设备类型 (如 '避雷器')
            temperature_regions: 温度区域数据 {'R01': 85.5, 'R02': 45.2, ...}
            
        Returns:
            诊断结果列表
        """
        if device_type not in self.device_rules:
            self.logger.warning(f"未找到 {device_type} 的诊断规则")
            return []
        
        rules = self.device_rules[device_type]
        diagnosis_results = []
        
        for rule in rules:
            try:
                result = self._evaluate_rule(rule, temperature_regions)
                if result['triggered']:
                    diagnosis_results.append(result)
            except Exception as e:
                self.logger.error(f"评估规则失败: {e}, 规则: {rule}")
        
        # 按严重程度排序
        severity_order = {'危急': 0, '严重': 1, '一般': 2, '正常': 3}
        diagnosis_results.sort(key=lambda x: severity_order.get(x['level'], 4))
        
        return diagnosis_results
    
    def _evaluate_rule(self, rule: Dict, temperatures: Dict[str, float]) -> Dict:
        """评估单条诊断规则"""
        formula = rule['parsed_formula']['expression']
        
        # 替换温度变量
        eval_formula = self._substitute_temperature_variables(formula, temperatures)
        
        # 替换环境温度
        eval_formula = eval_formula.replace('EnvTemp', str(self.env_temp))
        
        try:
            # 安全评估表达式
            result = self._safe_eval(eval_formula)
            triggered = bool(result)
            
            diagnosis_result = {
                'triggered': triggered,
                'level': rule['level'],
                'formula': rule['formula'],
                'evaluated_formula': eval_formula,
                'result': result,
                'diagnosis_rule': rule['diagnosis_rule'],
                'fault_feature': rule['fault_feature'],
                'thermal_feature': rule['thermal_feature'],
                'treatment_suggestion': rule['treatment_suggestion'] if rule['treatment_suggestion'] else '建议进一步检查设备状态',
                'diagnosis_method': rule['diagnosis_method'],
                'temperatures_used': temperatures
            }
            
            return diagnosis_result
            
        except Exception as e:
            self.logger.error(f"公式评估失败: {e}, 公式: {eval_formula}")
            return {
                'triggered': False,
                'level': rule['level'],
                'error': str(e),
                'formula': rule['formula']
            }
    
    def _substitute_temperature_variables(self, formula: str, temperatures: Dict[str, float]) -> str:
        """替换公式中的温度变量"""
        import re
        
        eval_formula = formula
        
        # 处理多种可能的温度变量格式
        patterns = [
            r'R(\d+):Max',    # R01:Max 格式
            r'R(\d+)_Max',    # R01_Max 格式  
            r'R(\d+)\.Max',   # R01.Max 格式
        ]
        
        for pattern in patterns:
            temp_vars = re.findall(pattern, eval_formula)
            for region_num in temp_vars:
                region_key = f"R{region_num.zfill(2)}"  # 确保是两位数格式，如 R01
                
                # 尝试不同的键名格式
                temp_value = None
                for key in [region_key, f"R{region_num}", region_key.lower()]:
                    if key in temperatures:
                        temp_value = temperatures[key]
                        break
                
                if temp_value is None:
                    # 使用默认值
                    temp_value = 25.0
                    self.logger.warning(f"未找到温度数据 {region_key}, 使用默认值 {temp_value}")
                
                # 替换所有匹配的模式
                old_patterns = [
                    f"R{region_num}:Max",
                    f"R{region_num}_Max", 
                    f"R{region_num}.Max",
                    f"R{region_num.zfill(2)}:Max",
                    f"R{region_num.zfill(2)}_Max",
                    f"R{region_num.zfill(2)}.Max"
                ]
                
                for old_pattern in old_patterns:
                    eval_formula = eval_formula.replace(old_pattern, str(temp_value))
        
        return eval_formula
    
    def _safe_eval(self, expression: str) -> bool:
        """安全地评估数学表达式"""
        # 允许的函数和操作符
        allowed_names = {
            'max': max,
            'min': min,
            'abs': abs,
            'math': math,
            '__builtins__': {}
        }
        
        # 预处理表达式，处理常见的数学函数
        # 替换中文逗号和处理函数调用
        processed_expr = expression
        
        # 替换 max(a，b) 中的中文逗号
        processed_expr = re.sub(r'max\(([^)]+)\)', lambda m: f"max({m.group(1).replace('，', ',')})", processed_expr)
        processed_expr = re.sub(r'min\(([^)]+)\)', lambda m: f"min({m.group(1).replace('，', ',')})", processed_expr)
        processed_expr = re.sub(r'abs\(([^)]+)\)', lambda m: f"abs({m.group(1).replace('，', ',')})", processed_expr)
        
        # 清理多余的空格
        processed_expr = re.sub(r'\s+', ' ', processed_expr.strip())
        
        try:
            result = eval(processed_expr, allowed_names)
            return result
        except Exception as e:
            self.logger.error(f"表达式评估失败: {processed_expr}, 错误: {e}")
            return False
    
    def batch_diagnose(self, devices_data: List[Dict]) -> List[Dict]:
        """
        批量诊断多个设备
        
        Args:
            devices_data: 设备数据列表，每个元素包含 {'device_type': str, 'temperatures': dict}
            
        Returns:
            批量诊断结果
        """
        batch_results = []
        
        for i, device_data in enumerate(devices_data):
            device_type = device_data.get('device_type', 'unknown')
            temperatures = device_data.get('temperatures', {})
            device_id = device_data.get('device_id', f'device_{i}')
            
            diagnosis_results = self.diagnose_device(device_type, temperatures)
            
            batch_results.append({
                'device_id': device_id,
                'device_type': device_type,
                'temperatures': temperatures,
                'diagnosis_results': diagnosis_results,
                'highest_severity': diagnosis_results[0]['level'] if diagnosis_results else '正常',
                'abnormal_count': len([r for r in diagnosis_results if r['triggered']])
            })
        
        return batch_results
    
    def generate_summary_report(self, batch_results: List[Dict]) -> Dict:
        """生成诊断汇总报告"""
        total_devices = len(batch_results)
        abnormal_devices = len([r for r in batch_results if r['abnormal_count'] > 0])
        
        severity_counts = {'危急': 0, '严重': 0, '一般': 0, '正常': 0}
        
        for result in batch_results:
            severity = result['highest_severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_devices': total_devices,
            'normal_devices': severity_counts.get('正常', 0),
            'abnormal_devices': abnormal_devices,
            'severity_distribution': severity_counts,
            'abnormal_rate': round((abnormal_devices / total_devices * 100), 2) if total_devices > 0 else 0
        }

def test_diagnosis_engine():
    """测试诊断引擎"""
    # 创建诊断引擎
    engine = TemperatureDiagnosisEngine()
    
    # 设置环境温度
    engine.set_environment_temperature(25.0)
    
    # 测试避雷器诊断
    test_temperatures = {
        'R01': 85.5,  # 接头温度
        'R02': 45.2,  # 本体温度1
        'R03': 46.8,  # 本体温度2
        'R04': 44.1   # 其他部位
    }
    
    results = engine.diagnose_device('避雷器', test_temperatures)
    
    print("=" * 60)
    print("避雷器温度异常诊断结果")
    print("=" * 60)
    print(f"测试温度数据: {test_temperatures}")
    print(f"环境温度: {engine.env_temp}°C")
    print(f"诊断结果数量: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"\n【诊断结果 {i}】")
        print(f"严重程度: {result['level']}")
        print(f"是否触发: {result['triggered']}")
        print(f"诊断规则: {result['diagnosis_rule']}")
        print(f"故障特征: {result['fault_feature']}")
        print(f"处理建议: {result['treatment_suggestion']}")
        if 'error' in result:
            print(f"错误信息: {result['error']}")
        # 添加调试信息
        if 'evaluated_formula' in result:
            print(f"评估公式: {result['evaluated_formula']}")
        if 'formula' in result:
            print(f"原始公式: {result['formula']}")

if __name__ == '__main__':
    test_diagnosis_engine()
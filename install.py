#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV20241021 温度检测系统环境配置

专门为jyc_conda环境配置所需依赖
不创建新环境，使用现有的jyc_conda环境

Author: AI Assistant
Date: 2025-09-19
Environment: jyc_conda
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

class JycCondaConfigurator:
    """jyc_conda环境配置器"""
    
    def __init__(self):
        self.setup_logging()
        self.conda_env = "jyc_conda"
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_conda_env(self):
        """检查conda环境"""
        current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        self.logger.info(f"当前环境: {current_env}")
        
        if current_env != self.conda_env:
            self.logger.warning(f"建议在 {self.conda_env} 环境中运行")
            return False
        
        self.logger.info(f"✅ 正在 {self.conda_env} 环境中运行")
        return True
    
    def run_command(self, command, description=""):
        """运行命令"""
        if description:
            self.logger.info(f"📦 {description}")
        
        self.logger.info(f"执行: {command}")
        
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            if result.stdout:
                self.logger.info(f"输出: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"命令失败: {e}")
            if e.stderr:
                self.logger.error(f"错误: {e.stderr}")
            return False
    
    def install_pytorch_cuda(self):
        """安装PyTorch CUDA支持"""
        self.logger.info("🔥 安装PyTorch CUDA支持...")
        
        # 检查CUDA版本
        cuda_check = "nvidia-smi | grep -oP 'CUDA Version: \\K[0-9]+\\.[0-9]+'"
        
        # 安装PyTorch with CUDA支持 (适用于CUDA 12.x)
        pytorch_install = (
            "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 "
            "-c pytorch -c nvidia -y"
        )
        
        if not self.run_command(pytorch_install, "安装PyTorch CUDA版本"):
            self.logger.warning("PyTorch CUDA安装失败，尝试CPU版本...")
            cpu_install = "conda install pytorch torchvision torchaudio cpuonly -c pytorch -y"
            return self.run_command(cpu_install, "安装PyTorch CPU版本")
        
        return True
    
    def install_ultralytics(self):
        """安装Ultralytics YOLO"""
        commands = [
            ("pip install ultralytics", "安装Ultralytics YOLO"),
            ("pip install ultralytics[export]", "安装YOLO导出功能")
        ]
        
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
        return True
    
    def install_opencv(self):
        """安装OpenCV"""
        commands = [
            ("conda install opencv -c conda-forge -y", "安装OpenCV"),
            ("pip install opencv-python", "确保OpenCV Python绑定")
        ]
        
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
        return True
    
    def install_data_science_packages(self):
        """安装数据科学包"""
        commands = [
            ("conda install numpy pandas matplotlib seaborn -y", "安装基础数据科学包"),
            ("conda install scikit-learn -y", "安装机器学习包"),
            ("pip install pillow", "安装图像处理包")
        ]
        
        for cmd, desc in commands:
            if not self.run_command(cmd, desc):
                return False
        return True
    
    def install_other_dependencies(self):
        """安装其他依赖"""
        commands = [
            ("pip install tqdm", "安装进度条库"),
            ("pip install pyyaml", "安装YAML支持"),
            ("pip install psutil", "安装系统监控"),
            ("conda install jupyter -y", "安装Jupyter支持")
        ]
        
        for cmd, desc in commands:
            self.run_command(cmd, desc)  # 这些不是必需的，失败也继续
        
        return True
    
    def verify_installation(self):
        """验证安装"""
        self.logger.info("🔍 验证安装...")
        
        verification_script = '''
import sys
print(f"Python版本: {sys.version}")

# 检查核心包
packages = ["torch", "torchvision", "ultralytics", "cv2", "numpy", "matplotlib"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}: 已安装")
    except ImportError:
        print(f"❌ {pkg}: 未安装")

# 检查CUDA
try:
    import torch
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
except:
    print("CUDA检查失败")

# 检查YOLO
try:
    from ultralytics import YOLO
    print("✅ YOLO: 可用")
except:
    print("❌ YOLO: 不可用")
'''
        
        # 将验证脚本写入临时文件
        verify_file = "/tmp/verify_jyc_conda.py"
        with open(verify_file, 'w') as f:
            f.write(verification_script)
        
        # 运行验证
        self.run_command(f"python {verify_file}", "运行环境验证")
        
        # 清理临时文件
        os.remove(verify_file)
    
    def configure_gpu_settings(self):
        """配置GPU设置"""
        self.logger.info("🖥️  配置GPU设置...")
        
        gpu_config = f'''
# GPU配置文件
# 自动生成时间: {self.get_timestamp()}

# GPU管理交给程序内部自动选择，避免索引冲突
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7  # 注释掉，让PyTorch直接管理GPU选择

# PyTorch设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 显存优化
export CUDA_LAUNCH_BLOCKING=1
'''
        
        gpu_config_file = os.path.join(self.project_root, 'gpu_config.sh')
        with open(gpu_config_file, 'w') as f:
            f.write(gpu_config)
        
        self.logger.info(f"✅ GPU配置已保存: {gpu_config_file}")
        
    def get_timestamp(self):
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def create_conda_activation_script(self):
        """创建conda环境激活脚本"""
        script_content = f'''#!/bin/bash
# UAV20241021 温度检测系统启动脚本
# 自动激活jyc_conda环境

echo "🌡️ UAV20241021 温度检测系统"
echo "=" * 50

# 激活conda环境
echo "🔧 激活jyc_conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {self.conda_env}

# 检查环境
echo "📊 当前环境: $CONDA_DEFAULT_ENV"

# 设置GPU
source {self.project_root}/gpu_config.sh

# 进入项目目录
cd {self.project_root}

echo "✅ 环境配置完成"
echo "💡 可以运行: python main.py --mode full"
'''
        
        script_path = os.path.join(self.project_root, 'activate_env.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 添加执行权限
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"✅ 环境激活脚本已创建: {script_path}")
    
    def run_configuration(self):
        """运行完整配置过程"""
        self.logger.info("🔧 开始配置jyc_conda环境...")
        
        # 检查环境
        if not self.check_conda_env():
            self.logger.warning("环境检查失败，继续配置...")
        
        # 创建项目目录
        os.makedirs(self.project_root, exist_ok=True)
        
        # 配置步骤
        steps = [
            (self.install_pytorch_cuda, "安装PyTorch CUDA"),
            (self.install_ultralytics, "安装Ultralytics YOLO"),
            (self.install_opencv, "安装OpenCV"), 
            (self.install_data_science_packages, "安装数据科学包"),
            (self.install_other_dependencies, "安装其他依赖"),
            (self.configure_gpu_settings, "配置GPU设置"),
            (self.create_conda_activation_script, "创建启动脚本"),
            (self.verify_installation, "验证安装")
        ]
        
        success_count = 0
        for step_func, step_name in steps:
            try:
                self.logger.info(f"\n{'='*20} {step_name} {'='*20}")
                if step_func():
                    success_count += 1
                    self.logger.info(f"✅ {step_name} 完成")
                else:
                    self.logger.warning(f"⚠️  {step_name} 失败")
            except Exception as e:
                self.logger.error(f"❌ {step_name} 异常: {e}")
        
        # 总结
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🎉 配置完成!")
        self.logger.info(f"✅ 成功步骤: {success_count}/{len(steps)}")
        
        if success_count >= len(steps) - 2:  # 允许少数非关键步骤失败
            self.logger.info("🌡️ jyc_conda环境配置成功!")
            self.logger.info(f"💡 使用方法:")
            self.logger.info(f"   cd {self.project_root}")
            self.logger.info(f"   python main.py --mode full")
            return True
        else:
            self.logger.warning("⚠️  配置过程中存在问题，请检查错误信息")
            return False

def main():
    """主函数"""
    print("🔧 UAV20241021 jyc_conda环境配置器")
    print("="*50)
    print("📦 目标: 在现有jyc_conda环境中安装所需依赖")
    print("🚫 不会: 创建新的conda环境")
    print("✅ 特色: 温度数据驱动检测系统专用配置")
    print("="*50)
    
    configurator = JycCondaConfigurator()
    success = configurator.run_configuration()
    
    if success:
        print("\n🎉 恭喜! jyc_conda环境配置完成")
        print("🌡️ 温度检测系统已就绪")
    else:
        print("\n⚠️  配置过程存在问题")
        print("💡 请检查错误信息并重试")

if __name__ == "__main__":
    main()

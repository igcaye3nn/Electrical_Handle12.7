// 用户管理类
class UserManager {
    constructor() {
        this.currentUser = null;
        this.init();
    }

    init() {
        this.checkLoginStatus();
        this.bindUserMenuEvents();
    }

    // 检查登录状态
    checkLoginStatus() {
        const isLoggedIn = localStorage.getItem('isLoggedIn');
        const username = localStorage.getItem('username');
        
        if (isLoggedIn === 'true' && username) {
            this.currentUser = {
                username: username,
                role: '系统管理员',
                loginTime: localStorage.getItem('loginTime') || new Date().toLocaleString()
            };
            this.updateUserInterface();
        } else {
            // 如果未登录，重定向到登录页
            window.location.href = 'login.html';
        }
    }

    // 更新用户界面
    updateUserInterface() {
        if (this.currentUser) {
            document.getElementById('currentUsername').textContent = this.currentUser.username;
            document.getElementById('dropdownUsername').textContent = this.currentUser.username;
            document.getElementById('dropdownUserRole').textContent = this.currentUser.role;
            document.getElementById('loginTime').textContent = `登录时间: ${this.currentUser.loginTime}`;
        }
    }

    // 绑定用户菜单事件
    bindUserMenuEvents() {
        const userProfile = document.getElementById('userProfile');
        const userDropdown = document.getElementById('userDropdown');
        const logoutBtn = document.getElementById('logoutBtn');

        // 用户资料点击事件
        if (userProfile) {
            userProfile.addEventListener('click', (e) => {
                e.stopPropagation();
                userProfile.classList.toggle('active');
                userDropdown.classList.toggle('active');
            });
        }

        // 点击页面其他地方关闭下拉菜单
        document.addEventListener('click', (e) => {
            if (!userProfile?.contains(e.target)) {
                userProfile?.classList.remove('active');
                userDropdown?.classList.remove('active');
            }
        });

        // 菜单项点击事件
        document.getElementById('profileSettings')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showMessage('个人设置功能开发中...', 'info');
        });

        document.getElementById('systemSettings')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showMessage('系统设置功能开发中...', 'info');
        });

        document.getElementById('changePassword')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showChangePasswordModal();
        });

        // 退出登录
        if (logoutBtn) {
            logoutBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.logout();
            });
        }
    }

    // 显示修改密码模态框
    showChangePasswordModal() {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <h3>修改密码</h3>
                <form id="changePasswordForm">
                    <div class="form-group">
                        <label>当前密码</label>
                        <input type="password" id="currentPassword" required>
                    </div>
                    <div class="form-group">
                        <label>新密码</label>
                        <input type="password" id="newPassword" required>
                    </div>
                    <div class="form-group">
                        <label>确认新密码</label>
                        <input type="password" id="confirmPassword" required>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="btn-cancel">取消</button>
                        <button type="submit" class="btn-confirm">确认修改</button>
                    </div>
                </form>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // 事件处理
        modal.querySelector('.btn-cancel').onclick = () => modal.remove();
        modal.onclick = (e) => {
            if (e.target === modal) modal.remove();
        };
        
        modal.querySelector('#changePasswordForm').onsubmit = (e) => {
            e.preventDefault();
            // 这里可以添加密码修改逻辑
            this.showMessage('密码修改功能开发中...', 'info');
            modal.remove();
        };
    }

    // 退出登录
    logout() {
        // 调用服务器登出API清除session
        fetch('/api/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            // 清除前端存储
            localStorage.removeItem('isLoggedIn');
            localStorage.removeItem('username');
            localStorage.removeItem('loginTime');
            
            this.showMessage('退出登录成功', 'success');
            setTimeout(() => {
                window.location.href = '/login';
            }, 1000);
        })
        .catch(error => {
            console.error('登出失败:', error);
            // 即使API调用失败，也强制跳转到登录页面
            localStorage.clear();
            window.location.href = '/login';
        });
    }

    // 显示消息
    showMessage(message, type = 'info') {
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${type}`;
        messageEl.textContent = message;
        messageEl.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            background: ${type === 'success' ? '#52c41a' : type === 'error' ? '#f5222d' : '#1890ff'};
            color: white;
            border-radius: 6px;
            z-index: 10000;
            transform: translateX(300px);
            transition: transform 0.3s ease;
        `;
        
        document.body.appendChild(messageEl);
        
        setTimeout(() => {
            messageEl.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            messageEl.style.transform = 'translateX(300px)';
            setTimeout(() => messageEl.remove(), 300);
        }, 3000);
    }
}

// 主应用程序
class UAVDetectionApp {
    constructor() {
        this.currentPage = 'dashboard';
        this.charts = {};
        this.userManager = new UserManager();
        this.init();
    }

    // 获取状态对应的颜色
    getStatusColor(status) {
        const colors = {
            'normal': '#52c41a',   // 绿色
            'warning': '#faad14',  // 橙色
            'danger': '#f5222d'    // 红色
        };
        return colors[status] || '#d9d9d9';
    }

    init() {
        this.bindEvents();
        this.initCharts();
        this.loadDashboard();
        this.startDataUpdates();
    }

    // 事件绑定
    bindEvents() {
        // 导航切换
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const target = e.currentTarget.getAttribute('href').substring(1);
                this.switchPage(target);
            });
        });

        // 地图控制按钮
        document.querySelectorAll('.btn-map-control').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.btn-map-control').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                const type = e.currentTarget.dataset.type;
                this.updateMapData(type);
            });
        });

        // 设备点击
        document.querySelectorAll('.device-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectDevice(item);
            });
        });

        // 上传按钮
        document.querySelector('.btn-upload')?.addEventListener('click', () => {
            this.switchPage('detection');
        });

        // 模式切换
        document.querySelectorAll('.mode-card').forEach(card => {
            card.addEventListener('click', (e) => {
                // 移除所有卡片的选中状态
                document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
                // 选中当前卡片
                e.currentTarget.classList.add('selected');
                
                const mode = e.currentTarget.dataset.mode;
                this.switchDiagnosisMode(mode);
            });
        });

        // 诊断按钮点击
        document.getElementById('start-diagnosis')?.addEventListener('click', () => {
            this.startDiagnosis();
        });

        // 重置按钮点击
        document.getElementById('reset-form')?.addEventListener('click', () => {
            this.resetForm();
        });

        // 置信度滑动条和输入框联动
        const confidenceSlider = document.getElementById('confidence-slider');
        const confidenceInput = document.getElementById('confidence-input');
        
        if (confidenceSlider && confidenceInput) {
            confidenceSlider.addEventListener('input', (e) => {
                confidenceInput.value = e.target.value;
            });
            
            confidenceInput.addEventListener('input', (e) => {
                confidenceSlider.value = e.target.value;
            });
        }

        // 文件上传事件监听器
        this.setupFileUploadListeners();
    }

    // 设置文件上传监听器
    setupFileUploadListeners() {
        // 设备识别文件上传
        const deviceRecognitionFile = document.getElementById('device-recognition-file');
        if (deviceRecognitionFile) {
            deviceRecognitionFile.addEventListener('change', (e) => {
                this.handleFileSelection(e.target, 'device-recognition-file-list');
            });
        }

        // 单张图片诊断 - 红外图像
        const singleThermal = document.getElementById('single-thermal');
        if (singleThermal) {
            singleThermal.addEventListener('change', (e) => {
                this.handleFileSelection(e.target, 'single-thermal-file-list');
            });
        }

        // 单张图片诊断 - 温度数据
        const singleTempData = document.getElementById('single-temp-data');
        if (singleTempData) {
            singleTempData.addEventListener('change', (e) => {
                this.handleFileSelection(e.target, 'single-temp-data-file-list');
            });
        }

        // 批量诊断 - 红外图像
        const batchThermal = document.getElementById('batch-thermal');
        if (batchThermal) {
            batchThermal.addEventListener('change', (e) => {
                this.handleFileSelection(e.target, 'batch-thermal-file-list');
            });
        }

        // 批量诊断 - 温度数据
        const batchTempData = document.getElementById('batch-temp-data');
        if (batchTempData) {
            batchTempData.addEventListener('change', (e) => {
                this.handleFileSelection(e.target, 'batch-temp-data-file-list');
            });
        }
    }

    // 处理文件选择
    handleFileSelection(input, listId) {
        const fileList = document.getElementById(listId);
        if (!fileList) return;

        fileList.innerHTML = '';

        if (input.files && input.files.length > 0) {
            Array.from(input.files).forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                // 创建图片预览
                if (file.type.startsWith('image/')) {
                    const imagePreview = document.createElement('div');
                    imagePreview.className = 'image-preview';
                    
                    const img = document.createElement('img');
                    img.className = 'preview-img';
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        img.src = e.target.result;
                    };
                    reader.readAsDataURL(file);
                    
                    imagePreview.appendChild(img);
                    fileItem.appendChild(imagePreview);
                }
                
                const fileInfo = document.createElement('div');
                fileInfo.className = 'file-info';
                
                const fileName = document.createElement('span');
                fileName.className = 'file-name';
                fileName.textContent = file.name;
                
                const fileSize = document.createElement('span');
                fileSize.className = 'file-size';
                fileSize.textContent = this.formatFileSize(file.size);
                
                const removeBtn = document.createElement('button');
                removeBtn.className = 'file-remove';
                removeBtn.innerHTML = '×';
                removeBtn.onclick = () => {
                    fileItem.remove();
                    // 如果是单文件输入，清空input
                    if (!input.multiple) {
                        input.value = '';
                    }
                };
                
                fileInfo.appendChild(fileName);
                fileInfo.appendChild(fileSize);
                fileInfo.appendChild(removeBtn);
                fileItem.appendChild(fileInfo);
                
                fileList.appendChild(fileItem);
            });

            console.log(`文件已选择: ${input.files.length} 个文件`);
        }
    }

    // 格式化文件大小
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 页面切换
    switchPage(page) {
        // 更新导航
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[href="#${page}"]`)?.classList.add('active');

        // 更新页面
        document.querySelectorAll('.page').forEach(p => {
            p.classList.remove('active');
        });
        document.getElementById(page)?.classList.add('active');

        this.currentPage = page;

        // 页面特殊处理
        if (page === 'map') {
            setTimeout(() => this.initDetailMap(), 100);
        } else if (page === 'reports') {
            setTimeout(() => this.initReportCharts(), 100);
        } else if (page === 'detection') {
            setTimeout(() => this.initDetectionPage(), 100);
        }
    }

    // 初始化检测分析页面
    initDetectionPage() {
        // 默认选择第一个模式
        const firstModeCard = document.querySelector('.mode-card[data-mode="device-recognition"]');
        if (firstModeCard) {
            firstModeCard.classList.add('selected');
            this.switchDiagnosisMode('device-recognition');
        }
    }

    // 初始化详细地图（地图监控页面）
    initDetailMap() {
        const mapDom = document.getElementById('detail-map-chart');
        if (!mapDom) {
            console.log('详细地图容器不存在');
            return;
        }

        // 等待地图数据加载
        if (!isMapDataReady()) {
            console.log('等待地图数据加载完成...');
            document.addEventListener('chinaMapDataReady', () => {
                this.initDetailMap();
            });
            return;
        }

        const myChart = echarts.init(mapDom);
        const mapData = getChinaMapGeoJSON();
        
        if (mapData) {
            echarts.registerMap('china', mapData);
        }

        // 准备设备数据
        const data = Object.keys(equipmentData).map(name => ({
            name: name,
            value: equipmentData[name].value,
            status: equipmentData[name].status,
            itemStyle: {
                color: this.getStatusColor(equipmentData[name].status)
            }
        }));

        const option = {
            title: {
                text: '全国设备分布监控',
                left: 'center',
                top: 10,
                textStyle: {
                    color: '#ffffff',
                    fontSize: 18,
                    fontWeight: 'bold',
                    fontFamily: 'Arial, "Microsoft YaHei", sans-serif'
                }
            },
            tooltip: {
                trigger: 'item',
                triggerOn: 'mousemove',
                showDelay: 100,
                hideDelay: 100,
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                borderColor: '#333',
                borderWidth: 1,
                textStyle: {
                    color: '#ffffff',
                    fontSize: 14,
                    fontWeight: 'normal'
                },
                formatter: function(params) {
                    // 为地图上的标记点显示工具提示
                    if (params.componentType === 'markPoint' && params.data && params.data.name) {
                        const name = params.data.name;
                        const value = params.data.value;
                        const status = params.data.status;
                        const statusText = {
                            'normal': '正常',
                            'warning': '警告',
                            'danger': '危险'
                        };
                        // 模拟平均温度数据
                        const avgTemp = (Math.random() * 20 + 25).toFixed(1);
                        return `
                            <div style="padding: 12px; color: white; font-family: Arial, sans-serif;">
                                <div style="font-weight: bold; margin-bottom: 8px; font-size: 16px;">${name}</div>
                                <div style="margin-bottom: 4px;">设备数量: ${value.toLocaleString()}</div>
                                <div style="margin-bottom: 4px;">状态: <span style="color: ${status === 'normal' ? '#4caf50' : status === 'warning' ? '#ff9800' : '#f44336'}">${statusText[status]}</span></div>
                                <div>平均温度: ${avgTemp}°C</div>
                            </div>
                        `;
                    }
                    // 其他所有情况都不显示工具提示
                    return false;
                }
            },
            visualMap: {
                min: 0,
                max: Math.max(...Object.values(equipmentData).map(d => d.value)),
                left: 'left',
                top: 'bottom',
                text: ['高密度', '低密度'],
                textStyle: {
                    color: '#ffffff',
                    fontSize: 14,
                    fontWeight: 'bold'
                },
                calculable: true,
                inRange: {
                    color: ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#9c27b0']
                },
                itemWidth: 20,
                itemHeight: 120
            },
            geo: {
                map: 'china',
                roam: true,
                scaleLimit: {
                    min: 0.8,
                    max: 3
                },
                zoom: 1.6,
                center: [104.0, 37.5],
                itemStyle: {
                    areaColor: 'transparent',
                    borderColor: 'transparent'
                }
            },
            series: [
                {
                    name: '设备密度',
                    type: 'map',
                    map: 'china',
                    roam: true,
                    scaleLimit: {
                        min: 0.8,
                        max: 3
                    },
                    zoom: 1.6,
                    center: [104.0, 37.5],
                    data: data,
                    itemStyle: {
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    emphasis: {
                        itemStyle: {
                            borderColor: '#ffffff',
                            borderWidth: 2
                        }
                    },
                    label: {
                        show: true,
                        color: '#ffffff',
                        fontSize: 12,
                        fontWeight: 'bold'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            color: '#ffffff',
                            fontSize: 14,
                            fontWeight: 'bold'
                        }
                    },
                    tooltip: {
                        show: false
                    }
                },
                {
                    name: '设备分布',
                    type: 'map',
                    map: 'china',
                    geoIndex: 0,
                    data: data,
                    itemStyle: {
                        areaColor: 'transparent',
                        borderColor: 'transparent'
                    },
                    emphasis: {
                        itemStyle: {
                            areaColor: 'transparent',
                            borderColor: 'transparent'
                        }
                    },
                    label: {
                        show: false
                    },
                    tooltip: {
                        show: true,
                        formatter: function(params) {
                            if (params.data && params.data.name) {
                                const name = params.data.name;
                                const value = params.data.value;
                                const status = params.data.status;
                                const statusText = {
                                    'normal': '正常',
                                    'warning': '警告',
                                    'danger': '危险'
                                };
                                // 模拟平均温度数据
                                const avgTemp = (Math.random() * 20 + 25).toFixed(1);
                                return `
                                    <div style="padding: 12px; color: white; font-family: Arial, sans-serif;">
                                        <div style="font-weight: bold; margin-bottom: 8px; font-size: 16px;">${name}</div>
                                        <div style="margin-bottom: 4px;">设备数量: ${value.toLocaleString()}</div>
                                        <div style="margin-bottom: 4px;">状态: <span style="color: ${status === 'normal' ? '#4caf50' : status === 'warning' ? '#ff9800' : '#f44336'}">${statusText[status]}</span></div>
                                        <div>平均温度: ${avgTemp}°C</div>
                                    </div>
                                `;
                            }
                            return false;
                        }
                    },
                    markPoint: {
                        symbol: 'circle',
                        symbolSize: function(value, params) {
                            // 根据设备数量调整圆点大小
                            const deviceCount = params.data.value || 0;
                            return Math.max(8, Math.min(20, deviceCount / 200));
                        },
                        itemStyle: {
                            color: function(params) {
                                const status = params.data.status;
                                if (status === 'warning') return '#ff9800';
                                if (status === 'danger') return '#f44336';
                                return '#4caf50'; // normal
                            },
                            borderColor: '#ffffff',
                            borderWidth: 2,
                            shadowBlur: 3,
                            shadowColor: 'rgba(0, 0, 0, 0.3)'
                        },
                        emphasis: {
                            itemStyle: {
                                borderWidth: 3,
                                shadowBlur: 8,
                                shadowColor: 'rgba(0, 0, 0, 0.6)'
                            },
                            scale: 1.3
                        },
                        data: data.filter(item => {
                            return equipmentData[item.name] && equipmentData[item.name].value > 0;
                        }).map(item => ({
                            name: item.name,
                            value: item.value,
                            status: item.status,
                            coord: item.name // 使用省份名称作为坐标，让ECharts自动定位
                        }))
                    }
                }
            ]
        };

        myChart.setOption(option);
        this.charts.detailMap = myChart;

        // 响应式
        window.addEventListener('resize', () => {
            myChart.resize();
        });

        console.log('详细地图初始化完成');
    }

    // 初始化图表
    initCharts() {
        this.initTemperatureChart();
        this.initEquipmentLevelChart();
        // 等待地图数据加载完成后再初始化地图
        this.waitForMapDataAndInit();
    }

    // 等待地图数据并初始化
    waitForMapDataAndInit() {
        if (isMapDataReady()) {
            console.log('地图数据已准备就绪，直接初始化');
            this.initChinaMap();
        } else {
            console.log('等待地图数据加载完成...');
            document.addEventListener('chinaMapDataReady', () => {
                console.log('收到地图数据就绪事件，开始初始化地图');
                this.initChinaMap();
            });
        }
    }

    // 初始化中国地图
    initChinaMap() {
        const mapDom = document.getElementById('china-map-chart');
        if (!mapDom) {
            console.log('地图容器不存在');
            return;
        }

        const geoJSONData = getChinaMapGeoJSON();
        if (!geoJSONData) {
            console.error('GeoJSON数据未加载');
            return;
        }

        console.log('开始初始化中国地图，GeoJSON特征数量:', geoJSONData.features ? geoJSONData.features.length : 0);

        // 注册地图数据
        echarts.registerMap('china', geoJSONData);
        
        const myChart = echarts.init(mapDom);
        
        // 准备设备数据
        const data = Object.keys(equipmentData).map(name => ({
            name: name,
            value: equipmentData[name].value,
            status: equipmentData[name].status
        }));

        const option = {
            title: {
                text: '全国UAV热成像设备分布图',
                left: 'center',
                top: 10,
                textStyle: {
                    color: '#ffffff',
                    fontSize: 18,
                    fontWeight: 'bold',
                    fontFamily: 'Arial, "Microsoft YaHei", sans-serif'
                }
            },
            tooltip: {
                trigger: 'item',
                triggerOn: 'mousemove',
                showDelay: 100,
                hideDelay: 100,
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                borderColor: '#333',
                borderWidth: 1,
                textStyle: {
                    color: '#ffffff',
                    fontSize: 14,
                    fontWeight: 'normal'
                },
                formatter: function(params) {
                    // 只为散点图显示工具提示，且必须是“设备分布”系列
                    if (params.seriesType === 'scatter' && params.seriesName === '设备分布' && params.data && params.data.name) {
                        const name = params.data.name;
                        const value = params.data.value[2];
                        const status = params.data.status;
                        const statusText = {
                            'normal': '正常',
                            'warning': '警告',
                            'danger': '危险'
                        };
                        // 模拟平均温度数据
                        const avgTemp = (Math.random() * 20 + 25).toFixed(1);
                        return `
                            <div style="padding: 12px; color: white; font-family: Arial, sans-serif;">
                                <div style="font-weight: bold; margin-bottom: 8px; font-size: 16px;">${name}</div>
                                <div style="margin-bottom: 4px;">设备数量: ${value.toLocaleString()}</div>
                                <div style="margin-bottom: 4px;">状态: <span style="color: ${status === 'normal' ? '#4caf50' : status === 'warning' ? '#ff9800' : '#f44336'}">${statusText[status]}</span></div>
                                <div>平均温度: ${avgTemp}°C</div>
                            </div>
                        `;
                    }
                    // 其他所有情况都不显示工具提示
                    return false;
                }
            },
            visualMap: {
                min: 0,
                max: Math.max(...Object.values(equipmentData).map(d => d.value)),
                left: 'left',
                top: 'bottom',
                text: ['高密度', '低密度'],
                textStyle: {
                    color: '#ffffff',
                    fontSize: 14,
                    fontWeight: 'bold'
                },
                calculable: true,
                inRange: {
                    color: ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#9c27b0']
                },
                itemWidth: 20,
                itemHeight: 120
            },
            geo: {
                map: 'china',
                roam: true,
                scaleLimit: {
                    min: 0.8,
                    max: 3
                },
                zoom: 1.6,
                center: [104.0, 37.5],
                itemStyle: {
                    areaColor: 'transparent',
                    borderColor: 'transparent'
                }
            },
            series: [
                {
                    name: '设备密度',
                    type: 'map',
                    map: 'china',
                    roam: true,
                    scaleLimit: {
                        min: 0.8,
                        max: 3
                    },
                    zoom: 1.6,
                    center: [104.0, 37.5],
                    data: data,
                    itemStyle: {
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    emphasis: {
                        itemStyle: {
                            borderColor: '#ffffff',
                            borderWidth: 2
                        }
                    },
                    label: {
                        show: true,
                        color: '#ffffff',
                        fontSize: 12,
                        fontWeight: 'bold'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            color: '#ffffff',
                            fontSize: 14,
                            fontWeight: 'bold'
                        }
                    },
                    tooltip: {
                        show: false
                    }
                },
                {
                    name: '设备分布',
                    type: 'scatter',
                    coordinateSystem: 'geo',
                    data: data.map(item => {
                        // 省份名称映射 - 将完整名称映射到简化名称
                        const nameMapping = {
                            '北京': '北京',
                            '天津': '天津', 
                            '河北省': '河北',
                            '山西省': '山西',
                            '内蒙古自治区': '内蒙古',
                            '辽宁省': '辽宁',
                            '吉林省': '吉林',
                            '黑龙江省': '黑龙江',
                            '上海': '上海',
                            '江苏省': '江苏',
                            '浙江省': '浙江',
                            '安徽省': '安徽',
                            '福建省': '福建',
                            '江西省': '江西',
                            '山东省': '山东',
                            '河南省': '河南',
                            '湖北省': '湖北',
                            '湖南省': '湖南',
                            '广东省': '广东',
                            '广西壮族自治区': '广西',
                            '海南省': '海南',
                            '重庆': '重庆',
                            '四川省': '四川',
                            '贵州省': '贵州',
                            '云南省': '云南',
                            '西藏自治区': '西藏',
                            '陕西省': '陕西',
                            '甘肃省': '甘肃',
                            '青海省': '青海',
                            '宁夏回族自治区': '宁夏',
                            '新疆维吾尔自治区': '新疆',
                            '台湾省': '台湾',
                            '香港特别行政区': '香港',
                            '澳门特别行政区': '澳门'
                        };
                        
                        // 精确的省份中心坐标
                        const coords = {
                            '北京': [116.4074, 39.9042],
                            '天津': [117.2008, 39.0842],
                            '河北': [114.5149, 38.0428],
                            '山西': [112.5489, 37.8570],
                            '内蒙古': [111.7658, 40.8176],
                            '辽宁': [123.4315, 41.8057],
                            '吉林': [125.3245, 43.8868],
                            '黑龙江': [126.6420, 45.7576],
                            '上海': [121.4737, 31.2304],
                            '江苏': [118.7633, 32.0615],
                            '浙江': [120.1538, 30.2875],
                            '安徽': [117.2272, 31.8206],
                            '福建': [119.2965, 26.0745],
                            '江西': [115.8921, 28.6765],
                            '山东': [117.0009, 36.6758],
                            '河南': [113.6540, 34.7566],
                            '湖北': [114.2985, 30.5844],
                            '湖南': [112.9823, 28.1941],
                            '广东': [113.2802, 23.1252],
                            '广西': [108.3202, 22.8244],
                            '海南': [110.3312, 20.0311],
                            '重庆': [106.5516, 29.5630],
                            '四川': [104.0665, 30.5722],
                            '贵州': [106.7133, 26.5783],
                            '云南': [102.7123, 25.0406],
                            '西藏': [91.1174, 29.6463],
                            '陕西': [108.9480, 34.2636],
                            '甘肃': [103.8236, 36.0581],
                            '青海': [101.7782, 36.6171],
                            '宁夏': [106.2784, 38.4664],
                            '新疆': [87.6177, 43.7928],
                            '台湾': [120.9605, 23.6978],
                            '香港': [114.1694, 22.3193],
                            '澳门': [113.5439, 22.1987]
                        };
                        
                        // 获取简化名称和对应坐标
                        const simpleName = nameMapping[item.name] || item.name;
                        const coord = coords[simpleName];
                        
                        if (!coord) {
                            console.warn(`省份 ${item.name} (${simpleName}) 没有找到对应坐标`);
                            return {
                                name: item.name,
                                value: [104.0, 37.5, item.value], // 默认中国中心位置
                                status: item.status
                            };
                        }
                        
                        return {
                            name: item.name,
                            value: [...coord, item.value],
                            status: item.status
                        };
                    }),
                    symbolSize: function (val) {
                        // 根据设备数量调整圆点大小
                        return Math.max(8, Math.min(20, val[2] / 200));
                    },
                    itemStyle: {
                        color: function(params) {
                            const status = params.data.status;
                            if (status === 'warning') return '#ff9800';
                            if (status === 'danger') return '#f44336';
                            return '#4caf50'; // normal
                        },
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        shadowBlur: 3,
                        shadowColor: 'rgba(0, 0, 0, 0.3)'
                    },
                    emphasis: {
                        itemStyle: {
                            borderWidth: 3,
                            shadowBlur: 8,
                            shadowColor: 'rgba(0, 0, 0, 0.6)'
                        },
                        scale: 1.3
                    },
                    // 确保散点图与地图同步缩放
                    zlevel: 2
                }
            ]
        };

        myChart.setOption(option);
        this.charts.chinaMap = myChart;

        console.log('中国地图初始化完成');

        // 响应式
        window.addEventListener('resize', () => {
            myChart.resize();
        });
    }

    // 获取状态颜色
    getStatusColor(status) {
        const colors = {
            'normal': 'rgba(76, 175, 80, 0.3)',
            'warning': 'rgba(255, 152, 0, 0.3)',
            'danger': 'rgba(244, 67, 54, 0.3)'
        };
        return colors[status] || colors.normal;
    }

    // 初始化温度趋势图
    initTemperatureChart() {
        const chartDom = document.getElementById('temperature-chart');
        if (!chartDom) return;

        const myChart = echarts.init(chartDom);
        
        // 生成模拟数据
        const hours = [];
        const temperatures = [];
        for (let i = 0; i < 24; i++) {
            hours.push(i + ':00');
            temperatures.push((Math.sin(i / 24 * Math.PI * 2) * 20 + 60 + Math.random() * 10).toFixed(1));
        }

        const option = {
            tooltip: {
                trigger: 'axis',
                formatter: '{b}: {c}°C'
            },
            xAxis: {
                type: 'category',
                data: hours,
                axisLabel: {
                    color: '#666'
                },
                axisLine: {
                    lineStyle: {
                        color: '#e0e0e0'
                    }
                }
            },
            yAxis: {
                type: 'value',
                name: '温度(°C)',
                nameTextStyle: {
                    color: '#666'
                },
                axisLabel: {
                    color: '#666'
                },
                axisLine: {
                    lineStyle: {
                        color: '#e0e0e0'
                    }
                },
                splitLine: {
                    lineStyle: {
                        color: '#f0f0f0'
                    }
                }
            },
            series: [{
                data: temperatures,
                type: 'line',
                smooth: true,
                lineStyle: {
                    color: '#667eea',
                    width: 3
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0,
                        y: 0,
                        x2: 0,
                        y2: 1,
                        colorStops: [{
                            offset: 0, color: 'rgba(102, 126, 234, 0.3)'
                        }, {
                            offset: 1, color: 'rgba(102, 126, 234, 0.05)'
                        }]
                    }
                },
                symbol: 'circle',
                symbolSize: 6,
                itemStyle: {
                    color: '#667eea',
                    borderColor: 'white',
                    borderWidth: 2
                }
            }]
        };

        myChart.setOption(option);
        this.charts.temperatureChart = myChart;

        // 响应式
        window.addEventListener('resize', () => {
            myChart.resize();
        });
    }

    // 初始化设备等级统计柱形图
    initEquipmentLevelChart() {
        const chartDom = document.getElementById('equipment-level-chart');
        if (!chartDom) return;

        const myChart = echarts.init(chartDom);
        
        const option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                },
                formatter: '{b}: {c}台'
            },
            grid: {
                left: '10%',
                right: '5%',
                bottom: '15%',
                top: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: ['正常', '一般', '警告', '危险'],
                axisLabel: {
                    color: '#666',
                    fontSize: 12
                },
                axisLine: {
                    lineStyle: {
                        color: '#e0e0e0'
                    }
                }
            },
            yAxis: {
                type: 'value',
                name: '设备数量',
                nameTextStyle: {
                    color: '#666',
                    fontSize: 11
                },
                axisLabel: {
                    color: '#666',
                    fontSize: 11
                },
                axisLine: {
                    lineStyle: {
                        color: '#e0e0e0'
                    }
                },
                splitLine: {
                    lineStyle: {
                        color: '#f0f0f0'
                    }
                }
            },
            series: [{
                type: 'bar',
                data: [
                    { value: 156, itemStyle: { color: '#52c41a' } },
                    { value: 45, itemStyle: { color: '#1890ff' } },
                    { value: 28, itemStyle: { color: '#faad14' } },
                    { value: 12, itemStyle: { color: '#f5222d' } }
                ],
                barWidth: '50%',
                label: {
                    show: true,
                    position: 'top',
                    color: '#333',
                    fontSize: 12,
                    fontWeight: 'bold',
                    formatter: '{c}台'
                }
            }]
        };

        myChart.setOption(option);
        this.charts.equipmentLevelChart = myChart;

        // 响应式
        window.addEventListener('resize', () => {
            myChart.resize();
        });
    }

    // 初始化详细地图(地图监控页面)
    initDetailMap() {
        const mapDom = document.getElementById('detail-map-chart');
        if (!mapDom) {
            console.warn('地图容器 detail-map-chart 不存在');
            return;
        }

        console.log('开始初始化地图监控页面地图');

        // 检查GeoJSON数据是否已加载
        if (!isMapDataReady()) {
            console.warn('GeoJSON数据未加载完成,无法初始化地图监控');
            return;
        }

        const geoJSON = getChinaMapGeoJSON();
        if (!geoJSON) {
            console.error('无法获取GeoJSON数据');
            return;
        }

        // 注册地图
        echarts.registerMap('china', geoJSON);
        const myChart = echarts.init(mapDom);
        
        // 准备省份数据 - 使用与主页相同的equipmentData
        const data = Object.keys(equipmentData).map(province => ({
            name: province,
            value: equipmentData[province].value,
            status: equipmentData[province].status,
            itemStyle: {
                color: this.getStatusColor(equipmentData[province].status)
            }
        }));

        const option = {
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    // 只在鼠标移到散点上时显示详细信息
                    if (params.seriesType === 'scatter') {
                        return `
                            <div style="padding: 10px; background: rgba(0, 0, 0, 0.8); border-radius: 6px; color: white;">
                                <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">${params.data.name}</div>
                                <div style="margin: 4px 0;">设备数量: <span style="color: #4fc3f7;">${params.data.value[2]}</span> 台</div>
                                <div style="margin: 4px 0;">正常: <span style="color: #4caf50;">${params.data.normal}</span> | 警告: <span style="color: #ff9800;">${params.data.warning}</span> | 危险: <span style="color: #f44336;">${params.data.danger}</span></div>
                                <div style="margin: 4px 0;">平均温度: <span style="color: #ff9800;">${params.data.avgTemp}</span>°C</div>
                            </div>
                        `;
                    }
                    return '';
                }
            },
            visualMap: {
                min: 0,
                max: Math.max(...Object.values(equipmentData).map(d => d.value)),
                left: 'left',
                top: 'bottom',
                text: ['高密度', '低密度'],
                textStyle: {
                    color: '#ffffff',
                    fontSize: 14,
                    fontWeight: 'bold'
                },
                calculable: true,
                inRange: {
                    color: ['#f3e5f5', '#e1bee7', '#ce93d8', '#ba68c8', '#9c27b0']
                },
                itemWidth: 20,
                itemHeight: 120
            },
            geo: {
                map: 'china',
                roam: true,
                scaleLimit: {
                    min: 0.8,
                    max: 3
                },
                zoom: 1.6,
                center: [104.0, 37.5],
                itemStyle: {
                    areaColor: 'transparent',
                    borderColor: 'transparent'
                }
            },
            series: [
                {
                    name: '设备密度',
                    type: 'map',
                    map: 'china',
                    roam: true,
                    scaleLimit: {
                        min: 0.8,
                        max: 3
                    },
                    zoom: 1.6,
                    center: [104.0, 37.5],
                    data: data,
                    itemStyle: {
                        borderColor: '#ffffff',
                        borderWidth: 1
                    },
                    emphasis: {
                        itemStyle: {
                            borderColor: '#ffffff',
                            borderWidth: 2
                        }
                    },
                    label: {
                        show: true,
                        color: '#ffffff',
                        fontSize: 12,
                        fontWeight: 'bold'
                    },
                    emphasis: {
                        label: {
                            show: true,
                            color: '#ffffff',
                            fontSize: 14,
                            fontWeight: 'bold'
                        }
                    },
                    tooltip: {
                        show: false
                    }
                },
                {
                    name: '设备分布',
                    type: 'scatter',
                    coordinateSystem: 'geo',
                    data: data.map(item => {
                        // 省份名称映射
                        const nameMapping = {
                            '北京': '北京',
                            '天津': '天津', 
                            '河北省': '河北',
                            '山西省': '山西',
                            '内蒙古自治区': '内蒙古',
                            '辽宁省': '辽宁',
                            '吉林省': '吉林',
                            '黑龙江省': '黑龙江',
                            '上海': '上海',
                            '江苏省': '江苏',
                            '浙江省': '浙江',
                            '安徽省': '安徽',
                            '福建省': '福建',
                            '江西省': '江西',
                            '山东省': '山东',
                            '河南省': '河南',
                            '湖北省': '湖北',
                            '湖南省': '湖南',
                            '广东省': '广东',
                            '广西壮族自治区': '广西',
                            '海南省': '海南',
                            '重庆': '重庆',
                            '四川省': '四川',
                            '贵州省': '贵州',
                            '云南省': '云南',
                            '西藏自治区': '西藏',
                            '陕西省': '陕西',
                            '甘肃省': '甘肃',
                            '青海省': '青海',
                            '宁夏回族自治区': '宁夏',
                            '新疆维吾尔自治区': '新疆',
                            '台湾省': '台湾',
                            '香港特别行政区': '香港',
                            '澳门特别行政区': '澳门'
                        };
                        
                        // 省份中心坐标
                        const coords = {
                            '北京': [116.4074, 39.9042],
                            '天津': [117.2008, 39.0842],
                            '河北': [114.5149, 38.0428],
                            '山西': [112.5489, 37.8570],
                            '内蒙古': [111.7658, 40.8176],
                            '辽宁': [123.4315, 41.8057],
                            '吉林': [125.3245, 43.8868],
                            '黑龙江': [126.6420, 45.7576],
                            '上海': [121.4737, 31.2304],
                            '江苏': [118.7633, 32.0615],
                            '浙江': [120.1538, 30.2875],
                            '安徽': [117.2272, 31.8206],
                            '福建': [119.2965, 26.0745],
                            '江西': [115.8921, 28.6765],
                            '山东': [117.0009, 36.6758],
                            '河南': [113.6540, 34.7566],
                            '湖北': [114.2985, 30.5844],
                            '湖南': [112.9823, 28.1941],
                            '广东': [113.2802, 23.1252],
                            '广西': [108.3202, 22.8244],
                            '海南': [110.3312, 20.0311],
                            '重庆': [106.5516, 29.5630],
                            '四川': [104.0665, 30.5722],
                            '贵州': [106.7133, 26.5783],
                            '云南': [102.7123, 25.0406],
                            '西藏': [91.1174, 29.6463],
                            '陕西': [108.9480, 34.2636],
                            '甘肃': [103.8236, 36.0581],
                            '青海': [101.7782, 36.6171],
                            '宁夏': [106.2784, 38.4664],
                            '新疆': [87.6177, 43.7928],
                            '台湾': [120.9605, 23.6978],
                            '香港': [114.1694, 22.3193],
                            '澳门': [113.5439, 22.1987]
                        };
                        
                        const simpleName = nameMapping[item.name] || item.name;
                        const coord = coords[simpleName];
                        
                        if (!coord) {
                            console.warn(`省份 ${item.name} 没有找到对应坐标`);
                            return {
                                name: item.name,
                                value: [104.0, 37.5, item.value],
                                status: item.status,
                                normal: item.normal,
                                warning: item.warning,
                                danger: item.danger,
                                avgTemp: item.avgTemp
                            };
                        }
                        
                        return {
                            name: item.name,
                            value: [...coord, item.value],
                            status: item.status,
                            normal: item.normal,
                            warning: item.warning,
                            danger: item.danger,
                            avgTemp: item.avgTemp
                        };
                    }),
                    symbolSize: function (val) {
                        return Math.max(8, Math.min(20, val[2] / 200));
                    },
                    itemStyle: {
                        color: function(params) {
                            const status = params.data.status;
                            if (status === 'warning') return '#ff9800';
                            if (status === 'danger') return '#f44336';
                            return '#4caf50';
                        },
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        shadowBlur: 5,
                        shadowColor: 'rgba(0, 0, 0, 0.5)',
                        opacity: 0.9
                    },
                    emphasis: {
                        itemStyle: {
                            borderWidth: 3,
                            shadowBlur: 10,
                            shadowColor: 'rgba(0, 0, 0, 0.7)',
                            scale: 1.2
                        }
                    },
                    // 确保散点图与地图同步缩放
                    zlevel: 2
                },
                {
                    name: '变电站',
                    type: 'scatter',
                    coordinateSystem: 'geo',
                    data: [
                        // 主要城市的变电站位置
                        { name: '北京变电站1', value: [116.4074, 39.9042, 100], status: 'normal' },
                        { name: '北京变电站2', value: [116.5074, 39.8042, 80], status: 'warning' },
                        { name: '上海变电站1', value: [121.4737, 31.2304, 120], status: 'normal' },
                        { name: '上海变电站2', value: [121.3737, 31.3304, 90], status: 'normal' },
                        { name: '广州变电站1', value: [113.2802, 23.1252, 110], status: 'danger' },
                        { name: '深圳变电站1', value: [114.0579, 22.5431, 85], status: 'normal' },
                        { name: '杭州变电站1', value: [120.1538, 30.2875, 95], status: 'warning' },
                        { name: '南京变电站1', value: [118.7633, 32.0615, 75], status: 'normal' },
                        { name: '天津变电站1', value: [117.2008, 39.0842, 88], status: 'normal' },
                        { name: '武汉变电站1', value: [114.2985, 30.5844, 92], status: 'warning' },
                        { name: '成都变电站1', value: [104.0665, 30.5722, 98], status: 'normal' },
                        { name: '西安变电站1', value: [108.9480, 34.2636, 82], status: 'normal' },
                        { name: '重庆变电站1', value: [106.5516, 29.5630, 87], status: 'warning' },
                        { name: '郑州变电站1', value: [113.6540, 34.7566, 78], status: 'normal' },
                        { name: '济南变电站1', value: [117.0009, 36.6758, 73], status: 'normal' },
                        { name: '石家庄变电站1', value: [114.5149, 38.0428, 69], status: 'normal' },
                        { name: '太原变电站1', value: [112.5489, 37.8570, 65], status: 'warning' },
                        { name: '沈阳变电站1', value: [123.4315, 41.8057, 71], status: 'normal' },
                        { name: '长春变电站1', value: [125.3245, 43.8868, 68], status: 'normal' },
                        { name: '哈尔滨变电站1', value: [126.6420, 45.7576, 66], status: 'normal' }
                    ],
                    symbol: 'pin',  // 使用图钉样式作为变电站图标
                    symbolSize: function(val) {
                        return [25, 35]; // 固定大小的图钉图标
                    },
                    itemStyle: {
                        color: function(params) {
                            const status = params.data.status;
                            if (status === 'warning') return '#ff9800';
                            if (status === 'danger') return '#f44336';
                            return '#2196f3'; // 变电站用蓝色表示
                        },
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        shadowBlur: 8,
                        shadowColor: 'rgba(0, 0, 0, 0.4)'
                    },
                    emphasis: {
                        itemStyle: {
                            borderWidth: 3,
                            shadowBlur: 12,
                            shadowColor: 'rgba(0, 0, 0, 0.6)'
                        },
                        scale: 1.3
                    },
                    tooltip: {
                        formatter: function(params) {
                            const status = params.data.status;
                            const statusText = {
                                'normal': '正常',
                                'warning': '警告', 
                                'danger': '危险'
                            };
                            const avgTemp = (Math.random() * 20 + 25).toFixed(1);
                            return `
                                <div style="padding: 12px; background: rgba(0, 0, 0, 0.8); border-radius: 6px; color: white;">
                                    <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px; color: #2196f3;">🏭 ${params.data.name}</div>
                                    <div style="margin: 4px 0;">类型: 变电站</div>
                                    <div style="margin: 4px 0;">状态: <span style="color: ${status === 'normal' ? '#4caf50' : status === 'warning' ? '#ff9800' : '#f44336'}">${statusText[status]}</span></div>
                                    <div style="margin: 4px 0;">当前负载: ${params.data.value[2]}%</div>
                                    <div>设备温度: ${avgTemp}°C</div>
                                </div>
                            `;
                        }
                    },
                    zlevel: 3  // 确保变电站图标在最上层
                }
            ]
        };

        myChart.setOption(option);
        this.charts.detailMap = myChart;

        window.addEventListener('resize', () => {
            myChart.resize();
        });

        console.log('地图监控页面地图初始化完成');
    }

    // 初始化报告图表
    initReportCharts() {
        this.initStatusPieChart();
        this.initTempTrendChart();
        this.initAccuracyBarChart();
    }

    // 状态分布饼图
    initStatusPieChart() {
        const chartDom = document.getElementById('status-pie-chart');
        if (!chartDom) return;

        const myChart = echarts.init(chartDom);
        
        const data = [
            { value: 3629979, name: '正常设备', itemStyle: { color: '#4caf50' } },
            { value: 3567779, name: '异常设备', itemStyle: { color: '#ff9800' } },
            { value: 42200, name: '高温报警', itemStyle: { color: '#f44336' } }
        ];

        const option = {
            tooltip: {
                trigger: 'item',
                formatter: '{a} <br/>{b}: {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                left: 'left',
                data: data.map(item => item.name)
            },
            series: [{
                name: '设备状态',
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['60%', '50%'],
                data: data,
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        };

        myChart.setOption(option);
        this.charts.statusPieChart = myChart;
    }

    // 温度趋势分析
    initTempTrendChart() {
        const chartDom = document.getElementById('temp-trend-chart');
        if (!chartDom) return;

        const myChart = echarts.init(chartDom);
        
        // 生成最近30天的数据
        const days = [];
        const avgTemp = [];
        const maxTemp = [];
        const minTemp = [];
        
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            days.push(`${date.getMonth() + 1}/${date.getDate()}`);
            
            const base = 45 + Math.sin(i / 30 * Math.PI * 2) * 10;
            avgTemp.push((base + Math.random() * 5).toFixed(1));
            maxTemp.push((base + 15 + Math.random() * 10).toFixed(1));
            minTemp.push((base - 10 + Math.random() * 5).toFixed(1));
        }

        const option = {
            tooltip: {
                trigger: 'axis'
            },
            legend: {
                data: ['平均温度', '最高温度', '最低温度']
            },
            xAxis: {
                type: 'category',
                data: days
            },
            yAxis: {
                type: 'value',
                name: '温度(°C)'
            },
            series: [
                {
                    name: '平均温度',
                    type: 'line',
                    data: avgTemp,
                    smooth: true,
                    lineStyle: { color: '#667eea' }
                },
                {
                    name: '最高温度',
                    type: 'line',
                    data: maxTemp,
                    smooth: true,
                    lineStyle: { color: '#f44336' }
                },
                {
                    name: '最低温度',
                    type: 'line',
                    data: minTemp,
                    smooth: true,
                    lineStyle: { color: '#4caf50' }
                }
            ]
        };

        myChart.setOption(option);
        this.charts.tempTrendChart = myChart;
    }

    // 检测精度统计
    initAccuracyBarChart() {
        const chartDom = document.getElementById('accuracy-bar-chart');
        if (!chartDom) return;

        const myChart = echarts.init(chartDom);
        
        const data = [
            { name: '变压器', accuracy: 92.5, color: '#667eea' },
            { name: '绝缘子', accuracy: 88.3, color: '#4ecdc4' },
            { name: '并沟线夹', accuracy: 85.7, color: '#ffd93d' },
            { name: '电缆终端', accuracy: 90.2, color: '#6bcf7f' },
            { name: '开关设备', accuracy: 87.9, color: '#4d9de0' }
        ];

        const option = {
            tooltip: {
                trigger: 'axis',
                formatter: '{b}: {c}%'
            },
            xAxis: {
                type: 'category',
                data: data.map(item => item.name),
                axisLabel: {
                    rotate: 45
                }
            },
            yAxis: {
                type: 'value',
                name: '准确率(%)',
                min: 80,
                max: 100
            },
            series: [{
                type: 'bar',
                data: data.map((item, index) => ({
                    value: item.accuracy,
                    itemStyle: {
                        color: item.color
                    }
                })),
                barWidth: '60%',
                label: {
                    show: true,
                    position: 'top',
                    formatter: '{c}%'
                }
            }]
        };

        myChart.setOption(option);
        this.charts.accuracyBarChart = myChart;
    }

    // 更新地图数据
    updateMapData(type) {
        // 过滤设备数据
        const filteredData = Object.keys(equipmentData).filter(name => {
            if (type === 'all') return true;
            return equipmentData[name].status === type;
        }).map(name => ({
            name: name,
            value: equipmentData[name].value,
            status: equipmentData[name].status,
            itemStyle: {
                color: this.getStatusColor(equipmentData[name].status)
            }
        }));

        console.log('筛选类型:', type, '筛选结果数量:', filteredData.length);

        // 更新主页地图
        if (this.charts.chinaMap) {
            this.charts.chinaMap.setOption({
                series: [{
                    data: filteredData
                }]
            });
        }

        // 更新地图监控页面的地图
        if (this.charts.detailMap) {
            this.charts.detailMap.setOption({
                series: [{
                    data: filteredData
                }]
            });
        }

        // 更新统计数据
        this.updateFilteredStats(filteredData, type);
    }

    // 更新筛选后的统计数据
    updateFilteredStats(filteredData, type) {
        // 计算筛选后的统计数据
        const totalDevices = filteredData.length;
        const totalEquipment = filteredData.reduce((sum, item) => sum + item.value, 0);

        // 更新主页统计数据
        const mainTotalDevicesEl = document.querySelector('.dashboard-stats .stat-box:nth-child(1) .stat-number');
        const mainTotalEquipmentEl = document.querySelector('.dashboard-stats .stat-box:nth-child(2) .stat-number');
        
        if (mainTotalDevicesEl) {
            mainTotalDevicesEl.textContent = totalDevices;
        }
        if (mainTotalEquipmentEl) {
            mainTotalEquipmentEl.textContent = totalEquipment;
        }

        // 更新地图监控页面的统计数据
        const mapTotalEl = document.querySelector('#map-monitoring .device-stats .stat-item:nth-child(1) .stat-value');
        const mapNormalEl = document.querySelector('#map-monitoring .device-stats .stat-item:nth-child(2) .stat-value');
        const mapAbnormalEl = document.querySelector('#map-monitoring .device-stats .stat-item:nth-child(3) .stat-value');
        const mapWarningEl = document.querySelector('#map-monitoring .device-stats .stat-item:nth-child(4) .stat-value');

        if (mapTotalEl && mapNormalEl && mapAbnormalEl && mapWarningEl) {
            // 根据筛选类型计算对应的数据
            if (type === 'all') {
                const allData = Object.values(equipmentData);
                const normalCount = allData.filter(d => d.status === 'normal').length;
                const warningCount = allData.filter(d => d.status === 'warning').length;
                const dangerCount = allData.filter(d => d.status === 'danger').length;
                
                mapTotalEl.textContent = allData.length;
                mapNormalEl.textContent = normalCount;
                mapAbnormalEl.textContent = warningCount + dangerCount;
                mapWarningEl.textContent = dangerCount;
            } else {
                mapTotalEl.textContent = totalDevices;
                mapNormalEl.textContent = type === 'normal' ? totalDevices : 0;
                mapAbnormalEl.textContent = (type === 'warning' || type === 'danger') ? totalDevices : 0;
                mapWarningEl.textContent = type === 'danger' ? totalDevices : 0;
            }
        }

        console.log(`筛选类型: ${type}, 省份数量: ${totalDevices}, 设备总数: ${totalEquipment}`);
    }

    // 选择设备
    selectDevice(deviceItem) {
        document.querySelectorAll('.device-item').forEach(item => {
            item.classList.remove('selected');
        });
        deviceItem.classList.add('selected');

        // 这里可以添加地图定位逻辑
        const deviceName = deviceItem.querySelector('.device-name').textContent;
        console.log('选择设备:', deviceName);
    }

    // 加载仪表盘
    loadDashboard() {
        this.updateStats();
        this.updateRecentResults();
    }

    // 更新统计数据
    updateStats() {
        const stats = document.querySelectorAll('.stat-value');
        stats.forEach((stat, index) => {
            const targetValue = stat.textContent;
            const currentValue = 0;
            this.animateNumber(stat, currentValue, parseInt(targetValue.replace(/,/g, '')), 2000);
        });
    }

    // 数字动画
    animateNumber(element, start, end, duration) {
        const range = end - start;
        const startTime = Date.now();
        
        const updateNumber = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = Math.floor(start + range * this.easeOutQuart(progress));
            
            element.textContent = current.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(updateNumber);
            }
        };
        
        updateNumber();
    }

    // 缓动函数
    easeOutQuart(t) {
        return 1 - Math.pow(1 - t, 4);
    }

    // 更新最新结果
    updateRecentResults() {
        const progressBar = document.querySelector('.progress-fill');
        if (progressBar) {
            let width = 0;
            const targetWidth = 78;
            const animate = () => {
                width += 2;
                progressBar.style.width = width + '%';
                if (width < targetWidth) {
                    requestAnimationFrame(animate);
                }
            };
            animate();
        }
    }

    // 开始数据更新
    startDataUpdates() {
        // 每30秒更新一次数据
        setInterval(() => {
            this.updateRealtimeData();
        }, 30000);
    }

    // 更新实时数据
    updateRealtimeData() {
        if (this.currentPage === 'dashboard') {
            // 更新温度图表
            if (this.charts.temperatureChart) {
                const option = this.charts.temperatureChart.getOption();
                const newData = option.series[0].data.map(value => 
                    (parseFloat(value) + (Math.random() - 0.5) * 5).toFixed(1)
                );
                
                this.charts.temperatureChart.setOption({
                    series: [{
                        data: newData
                    }]
                });
            }
        }
    }

    // 诊断模式切换
    switchDiagnosisMode(mode) {
        // 隐藏所有上传区域
        document.querySelectorAll('[id^="upload-"]').forEach(area => {
            area.style.display = 'none';
        });

        // 显示对应模式的上传区域
        const uploadArea = document.getElementById(`upload-${mode}`);
        if (uploadArea) {
            uploadArea.style.display = 'block';
        }

        // 更新诊断按钮文本
        const startButton = document.getElementById('start-diagnosis');
        if (startButton) {
            const buttonTexts = {
                'device-recognition': '开始识别',
                'single-diagnosis': '开始诊断',
                'batch-diagnosis': '开始批量诊断'
            };
            startButton.innerHTML = `<i class="fas fa-play"></i> ${buttonTexts[mode] || '开始处理'}`;
        }

        this.currentDiagnosisMode = mode;
        console.log('切换到诊断模式:', mode);
    }

    // 开始诊断
    async startDiagnosis() {
        const mode = this.currentDiagnosisMode;
        if (!mode) {
            alert('请先选择诊断模式');
            return;
        }

        try {
            switch (mode) {
                case 'device-recognition':
                    await this.startDeviceRecognition();
                    break;
                case 'single-diagnosis':
                    await this.startSingleDiagnosis();
                    break;
                case 'batch-diagnosis':
                    await this.startBatchDiagnosis();
                    break;
                default:
                    alert('未知的诊断模式');
            }
        } catch (error) {
            console.error('诊断过程出错:', error);
            alert('诊断过程出错: ' + error.message);
        }
    }

    // 设备识别
    async startDeviceRecognition() {
        const fileInput = document.getElementById('device-recognition-file');
        if (!fileInput.files.length) {
            alert('请先上传红外图像');
            return;
        }

        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('confidence', document.getElementById('confidence-input')?.value || '0.25');

        this.showProgress('正在识别设备...');

        try {
            console.log('开始设备识别请求...');
            const response = await fetch('/api/device-recognition', {
                method: 'POST',
                body: formData
            });

            console.log('收到服务器响应:', response.status);
            const result = await response.json();
            console.log('解析JSON结果:', result);
            this.hideProgress();

            if (result.success) {
                console.log('识别成功，显示结果');
                this.showDeviceRecognitionResult(result);
            } else {
                console.error('识别失败:', result.error);
                alert('识别失败: ' + result.error);
            }
        } catch (error) {
            this.hideProgress();
            throw error;
        }
    }

    // 单张图片诊断
    async startSingleDiagnosis() {
        const thermalInput = document.getElementById('single-thermal');
        if (!thermalInput.files.length) {
            alert('请先上传红外图像');
            return;
        }

        const formData = new FormData();
        formData.append('thermal_image', thermalInput.files[0]);
        formData.append('confidence', document.getElementById('confidence-input')?.value || '0.25');
        formData.append('temperature', document.getElementById('temp-threshold')?.value || '80');

        // 可选的温度数据
        const tempDataInput = document.getElementById('single-temp-data');
        if (tempDataInput.files.length) {
            formData.append('temperature_data', tempDataInput.files[0]);
        }

        this.showProgress('正在诊断图像...');

        try {
            const response = await fetch('/api/single-image-diagnosis', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.hideProgress();

            if (result.success) {
                this.showSingleDiagnosisResult(result);
            } else {
                alert('诊断失败: ' + result.error);
            }
        } catch (error) {
            this.hideProgress();
            throw error;
        }
    }

    // 批量图像诊断
    async startBatchDiagnosis() {
        const batchInput = document.getElementById('batch-thermal');
        if (!batchInput.files.length) {
            alert('请先上传图像文件');
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < batchInput.files.length; i++) {
            formData.append('images', batchInput.files[i]);
        }
        formData.append('confidence', document.getElementById('confidence-input')?.value || '0.25');
        formData.append('temperature', document.getElementById('temp-threshold')?.value || '80');

        this.showProgress('正在批量诊断图像...');

        try {
            const response = await fetch('/api/batch-image-diagnosis', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.hideProgress();

            if (result.success) {
                this.showBatchDiagnosisResult(result);
            } else {
                alert('批量诊断失败: ' + result.error);
            }
        } catch (error) {
            this.hideProgress();
            throw error;
        }
    }

    // 显示进度
    showProgress(text) {
        const container = document.getElementById('progress-container');
        const progressText = document.getElementById('progress-text');
        
        if (container) {
            container.style.display = 'block';
        }
        if (progressText) {
            progressText.textContent = text;
        }
    }

    // 隐藏进度
    hideProgress() {
        const container = document.getElementById('progress-container');
        if (container) {
            container.style.display = 'none';
        }
    }

    // 显示设备识别结果
    showDeviceRecognitionResult(result) {
        const resultsContainer = document.getElementById('results-container');
        const resultsContent = document.getElementById('results-content');

        if (!resultsContainer || !resultsContent) return;

        let html = `
            <div class="result-summary">
                <h4>识别结果摘要</h4>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="label">文件名：</span>
                        <span class="value">${result.filename}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">识别设备数：</span>
                        <span class="value">${result.devices_detected}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">图像尺寸：</span>
                        <span class="value">${result.image_size[0]} × ${result.image_size[1]}</span>
                    </div>
                </div>
            </div>
        `;

        // 添加原始图像显示
        if (result.image_data) {
            html += `
                <div class="original-image">
                    <h4>原始图像</h4>
                    <div class="image-container">
                        <img src="${result.image_data}" alt="原始红外图像" class="original-img">
                    </div>
                </div>
            `;
        }

        // 添加可视化检测结果图像
        if (result.visualization_data) {
            html += `
                <div class="visualization-image">
                    <h4>检测结果可视化</h4>
                    <div class="interactive-visualization">
                        <div class="visualization-container">
                            <div class="image-wrapper">
                                <img src="${result.image_data}" alt="原始红外图像" class="base-img" id="base-visualization">
                                <svg class="detection-overlay" id="detection-svg"></svg>
                            </div>
                            <div class="info-panel" id="device-info-panel">
                                <div class="info-title">设备信息</div>
                                <div class="info-content" id="device-info-content">
                                    <div class="info-placeholder">将鼠标悬停在检测框上查看设备详情</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        if (result.devices.length > 0) {
            html += `
                <div class="devices-list">
                    <h4>检测到的设备</h4>
                    <div class="devices-grid">
            `;

            result.devices.forEach((device, index) => {
                html += `
                    <div class="device-card">
                        <div class="device-header">
                            <span class="device-type">${device.class_name}</span>
                            <span class="confidence">置信度: ${(device.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="device-details">
                            <p>位置: [${device.bbox.map(x => Math.round(x)).join(', ')}]</p>
                        </div>
                    </div>
                `;
            });

            html += '</div></div>';
        }

        resultsContent.innerHTML = html;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
        
        // 如果有检测结果，初始化交互式可视化
        if (result.visualization_data && result.devices.length > 0) {
            setTimeout(() => {
                this.initInteractiveVisualization(result.devices, result.image_size);
            }, 100);
        }
    }

    // 初始化交互式可视化
    initInteractiveVisualization(devices, imageSize) {
        const baseImg = document.getElementById('base-visualization');
        const svg = document.getElementById('detection-svg');
        const infoPanel = document.getElementById('device-info-content');
        
        if (!baseImg || !svg || !infoPanel) return;

        // 等待图片加载完成
        baseImg.onload = () => {
            const rect = baseImg.getBoundingClientRect();
            const scaleX = baseImg.clientWidth / imageSize[0];
            const scaleY = baseImg.clientHeight / imageSize[1];
            
            // 设置SVG尺寸
            svg.style.width = baseImg.clientWidth + 'px';
            svg.style.height = baseImg.clientHeight + 'px';
            svg.setAttribute('viewBox', `0 0 ${baseImg.clientWidth} ${baseImg.clientHeight}`);
            
            // 清空SVG内容
            svg.innerHTML = '';
            
            // 定义颜色
            const colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', 
                '#FF00FF', '#00FFFF', '#800080', '#FFA500'
            ];
            
            devices.forEach((device, index) => {
                const [x1, y1, x2, y2] = device.bbox;
                const color = colors[index % colors.length];
                
                // 缩放坐标
                const scaledX1 = x1 * scaleX;
                const scaledY1 = y1 * scaleY;
                const scaledX2 = x2 * scaleX;
                const scaledY2 = y2 * scaleY;
                
                // 创建检测框
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', scaledX1);
                rect.setAttribute('y', scaledY1);
                rect.setAttribute('width', scaledX2 - scaledX1);
                rect.setAttribute('height', scaledY2 - scaledY1);
                rect.setAttribute('fill', 'none');
                rect.setAttribute('stroke', color);
                rect.setAttribute('stroke-width', '2');
                rect.setAttribute('class', 'detection-box');
                rect.style.cursor = 'pointer';
                rect.style.transition = 'all 0.3s ease';
                
                // 添加鼠标事件
                rect.addEventListener('mouseenter', (e) => {
                    rect.setAttribute('stroke-width', '4');
                    rect.style.filter = 'drop-shadow(0 0 8px ' + color + ')';
                    
                    // 在右侧信息面板显示设备信息
                    const infoPanel = document.getElementById('device-info-content');
                    const englishName = device.english_name || 'undefined';
                    if (infoPanel) {
                        infoPanel.innerHTML = `
                            <div class="device-info-item">
                                <div class="device-name">${device.class_name}</div>
                                <div class="device-detail">英文名称: ${englishName}</div>
                                <div class="device-detail">置信度: ${(device.confidence * 100).toFixed(1)}%</div>
                                <div class="device-detail">位置: [${device.bbox.map(x => Math.round(x)).join(', ')}]</div>
                                ${device.temperature ? `<div class="device-detail">温度: ${device.temperature}°C</div>` : ''}
                            </div>
                        `;
                    }
                });
                
                rect.addEventListener('mouseleave', (e) => {
                    rect.setAttribute('stroke-width', '2');
                    rect.style.filter = 'none';
                    
                    // 恢复默认信息显示
                    const infoPanel = document.getElementById('device-info-content');
                    if (infoPanel) {
                        infoPanel.innerHTML = '<div class="info-placeholder">将鼠标悬停在检测框上查看设备详情</div>';
                    }
                });
                

                
                svg.appendChild(rect);
                
                // 添加标签（显示英文名称）
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', scaledX1 + 1);
                text.setAttribute('y', scaledY1 - 1);
                text.setAttribute('fill', color);
                text.setAttribute('font-size', '12');
                text.setAttribute('font-weight', 'bold');
                // 使用英文名称显示在标签上
                const englishName = device.english_name || device.class_name;
                text.textContent = `${englishName}: ${(device.confidence * 100).toFixed(0)}%`;
                text.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
                svg.appendChild(text);
            });
        };
        
        // 如果图片已经加载完成，直接执行
        if (baseImg.complete) {
            baseImg.onload();
        }
    }

    // 显示单张图片诊断结果
    showSingleDiagnosisResult(result) {
        const resultsContainer = document.getElementById('results-container');
        const resultsContent = document.getElementById('results-content');

        if (!resultsContainer || !resultsContent) return;

        const summary = result.summary;
        
        let html = `
            <div class="result-summary">
                <h4>诊断结果摘要</h4>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="label">文件名：</span>
                        <span class="value">${result.filename}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">检测设备：</span>
                        <span class="value">${result.devices_detected}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">正常设备：</span>
                        <span class="value status-normal">${summary.normal_count}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">异常设备：</span>
                        <span class="value status-abnormal">${summary.abnormal_count}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">平均温度：</span>
                        <span class="value">${summary.average_temperature}°C</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">高风险设备：</span>
                        <span class="value status-danger">${summary.high_risk_count}</span>
                    </div>
                </div>
            </div>
        `;

        if (result.diagnoses.length > 0) {
            html += `
                <div class="diagnoses-list">
                    <h4>设备诊断详情</h4>
                    <div class="diagnoses-grid">
            `;

            result.diagnoses.forEach((diagnosis, index) => {
                const statusClass = diagnosis.is_abnormal ? 'abnormal' : 'normal';
                const riskClass = diagnosis.risk_level;
                
                html += `
                    <div class="diagnosis-card ${statusClass}">
                        <div class="diagnosis-header">
                            <span class="device-type">${diagnosis.device_type}</span>
                            <span class="status ${statusClass}">${diagnosis.status}</span>
                        </div>
                        <div class="diagnosis-details">
                            <p>温度: <span class="temperature ${riskClass}">${diagnosis.temperature}°C</span></p>
                            <p>置信度: ${(diagnosis.confidence * 100).toFixed(1)}%</p>
                            <p>风险等级: <span class="risk ${riskClass}">${diagnosis.risk_level}</span></p>
                        </div>
                    </div>
                `;
            });

            html += '</div></div>';
        }

        resultsContent.innerHTML = html;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // 显示批量诊断结果
    showBatchDiagnosisResult(result) {
        const resultsContainer = document.getElementById('results-container');
        const resultsContent = document.getElementById('results-content');

        if (!resultsContainer || !resultsContent) return;

        const summary = result.summary;
        
        let html = `
            <div class="result-summary">
                <h4>批量诊断结果摘要</h4>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="label">处理图像：</span>
                        <span class="value">${summary.total_images}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">检测设备：</span>
                        <span class="value">${summary.total_devices}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">正常设备：</span>
                        <span class="value status-normal">${summary.total_normal}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">异常设备：</span>
                        <span class="value status-abnormal">${summary.total_abnormal}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">异常率：</span>
                        <span class="value">${summary.abnormal_rate}%</span>
                    </div>
                </div>
            </div>
        `;

        if (result.results.length > 0) {
            html += `
                <div class="batch-results">
                    <h4>各文件诊断结果</h4>
                    <div class="files-list">
            `;

            result.results.forEach((fileResult, index) => {
                const fileStatusClass = fileResult.summary.abnormal_count > 0 ? 'has-abnormal' : 'all-normal';
                
                html += `
                    <div class="file-result-card ${fileStatusClass}">
                        <div class="file-header">
                            <h5>${fileResult.filename}</h5>
                            <span class="file-status">
                                ${fileResult.summary.abnormal_count > 0 ? '发现异常' : '全部正常'}
                            </span>
                        </div>
                        <div class="file-stats">
                            <span>设备: ${fileResult.devices_detected}</span>
                            <span>正常: ${fileResult.summary.normal_count}</span>
                            <span>异常: ${fileResult.summary.abnormal_count}</span>
                            <span>平均温度: ${fileResult.summary.average_temperature}°C</span>
                        </div>
                    </div>
                `;
            });

            html += '</div></div>';
        }

        resultsContent.innerHTML = html;
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // 重置表单
    resetForm() {
        // 重置所有文件输入
        document.querySelectorAll('input[type="file"]').forEach(input => {
            input.value = '';
        });

        // 清空文件列表
        document.querySelectorAll('.file-list').forEach(list => {
            list.innerHTML = '';
        });

        // 隐藏结果
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }

        // 隐藏进度
        this.hideProgress();

        // 重置模式选择
        document.querySelectorAll('.mode-card').forEach(card => {
            card.classList.remove('selected');
        });

        // 隐藏所有上传区域
        document.querySelectorAll('[id^="upload-"]').forEach(area => {
            area.style.display = 'none';
        });

        this.currentDiagnosisMode = null;
        console.log('表单已重置');
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new UAVDetectionApp();
});

// 导出给其他模块使用
window.UAVApp = UAVDetectionApp;
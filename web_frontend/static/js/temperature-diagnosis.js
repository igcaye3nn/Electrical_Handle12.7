// 温度异常诊断系统
class TemperatureDiagnosisSystem {
    constructor() {
        this.currentMode = 'single';
        this.uploadedFiles = {
            single: null,
            thermal: [],
            reference: [],
            data: null
        };
        this.isProcessing = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.initFileUpload();
        this.updateModeDisplay();
        this.syncParameters();
    }

    // 事件绑定
    bindEvents() {
        // 模式选择
        document.querySelectorAll('.mode-card').forEach(card => {
            card.addEventListener('click', () => {
                const mode = card.dataset.mode;
                this.selectMode(mode);
            });
        });

        // 参数同步
        document.getElementById('confidence-slider')?.addEventListener('input', (e) => {
            document.getElementById('confidence-input').value = e.target.value;
        });

        document.getElementById('confidence-input')?.addEventListener('input', (e) => {
            document.getElementById('confidence-slider').value = e.target.value;
        });

        // 操作按钮
        document.getElementById('start-diagnosis')?.addEventListener('click', () => {
            this.startDiagnosis();
        });

        document.getElementById('reset-form')?.addEventListener('click', () => {
            this.resetForm();
        });
    }

    // 初始化文件上传
    initFileUpload() {
        // 单张图像上传
        this.initSingleFileUpload();
        
        // 批量图像上传
        this.initBatchFileUpload();
        
        // 数据文件上传
        this.initDataFileUpload();
    }

    // 单张图像上传
    initSingleFileUpload() {
        const fileInput = document.getElementById('single-file');
        const fileList = document.getElementById('single-file-list');

        if (!fileInput) return;

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadedFiles.single = file;
                this.displayFileList([file], fileList, 'single');
            }
        });

        // 拖放支持
        const uploadBox = fileInput.parentElement.querySelector('.upload-box');
        this.addDragDropSupport(uploadBox, fileInput);
    }

    // 批量图像上传
    initBatchFileUpload() {
        // 红外图像
        const thermalInput = document.getElementById('batch-thermal');
        const thermalList = document.getElementById('thermal-file-list');

        if (thermalInput) {
            thermalInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.uploadedFiles.thermal = files;
                this.displayFileList(files, thermalList, 'thermal');
            });

            const thermalBox = thermalInput.parentElement.querySelector('.upload-box');
            this.addDragDropSupport(thermalBox, thermalInput);
        }

        // 参考图像
        const referenceInput = document.getElementById('batch-reference');
        const referenceList = document.getElementById('reference-file-list');

        if (referenceInput) {
            referenceInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.uploadedFiles.reference = files;
                this.displayFileList(files, referenceList, 'reference');
            });

            const referenceBox = referenceInput.parentElement.querySelector('.upload-box');
            this.addDragDropSupport(referenceBox, referenceInput);
        }
    }

    // 数据文件上传
    initDataFileUpload() {
        const fileInput = document.getElementById('data-file');
        const fileList = document.getElementById('data-file-list');

        if (!fileInput) return;

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadedFiles.data = file;
                this.displayFileList([file], fileList, 'data');
            }
        });

        const uploadBox = fileInput.parentElement.querySelector('.upload-box');
        this.addDragDropSupport(uploadBox, fileInput);
    }

    // 添加拖拽支持
    addDragDropSupport(element, input) {
        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            element.classList.add('dragover');
        });

        element.addEventListener('dragleave', () => {
            element.classList.remove('dragover');
        });

        element.addEventListener('drop', (e) => {
            e.preventDefault();
            element.classList.remove('dragover');
            
            const files = Array.from(e.dataTransfer.files);
            const event = new Event('change');
            
            if (input.multiple) {
                input.files = e.dataTransfer.files;
            } else if (files.length > 0) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(files[0]);
                input.files = dataTransfer.files;
            }
            
            input.dispatchEvent(event);
        });
    }

    // 显示文件列表
    displayFileList(files, container, type) {
        if (!container) return;

        container.innerHTML = '';
        
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    <i class="file-icon fas fa-file-image"></i>
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${this.formatFileSize(file.size)}</span>
                </div>
                <button class="file-remove" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;

            // 移除文件事件
            fileItem.querySelector('.file-remove').addEventListener('click', () => {
                this.removeFile(type, index);
            });

            container.appendChild(fileItem);
        });
    }

    // 移除文件
    removeFile(type, index) {
        if (type === 'single') {
            this.uploadedFiles.single = null;
            document.getElementById('single-file-list').innerHTML = '';
            document.getElementById('single-file').value = '';
        } else if (type === 'thermal') {
            this.uploadedFiles.thermal.splice(index, 1);
            this.displayFileList(this.uploadedFiles.thermal, document.getElementById('thermal-file-list'), 'thermal');
        } else if (type === 'reference') {
            this.uploadedFiles.reference.splice(index, 1);
            this.displayFileList(this.uploadedFiles.reference, document.getElementById('reference-file-list'), 'reference');
        } else if (type === 'data') {
            this.uploadedFiles.data = null;
            document.getElementById('data-file-list').innerHTML = '';
            document.getElementById('data-file').value = '';
        }
    }

    // 格式化文件大小
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 选择模式
    selectMode(mode) {
        this.currentMode = mode;
        
        // 更新卡片状态
        document.querySelectorAll('.mode-card').forEach(card => {
            card.classList.remove('active');
        });
        document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
        
        // 更新显示
        this.updateModeDisplay();
    }

    // 更新模式显示
    updateModeDisplay() {
        // 隐藏所有上传区域
        document.querySelectorAll('.upload-area-group').forEach(area => {
            area.classList.remove('active');
        });

        // 显示当前模式的上传区域
        const currentArea = document.getElementById(`upload-${this.currentMode}`);
        if (currentArea) {
            currentArea.classList.add('active');
        }
    }

    // 参数同步
    syncParameters() {
        const confidenceSlider = document.getElementById('confidence-slider');
        const confidenceInput = document.getElementById('confidence-input');
        
        if (confidenceSlider && confidenceInput) {
            confidenceSlider.value = confidenceInput.value;
        }
    }

    // 开始诊断
    async startDiagnosis() {
        if (this.isProcessing) return;

        // 验证输入
        if (!this.validateInput()) {
            return;
        }

        this.isProcessing = true;
        this.showProgress();

        try {
            let results;
            
            if (this.currentMode === 'single') {
                results = await this.processSingleImage();
            } else if (this.currentMode === 'batch') {
                results = await this.processBatchImages();
            } else if (this.currentMode === 'data') {
                results = await this.processDataFile();
            }

            this.displayResults(results);
        } catch (error) {
            this.showError('诊断过程中发生错误：' + error.message);
        } finally {
            this.isProcessing = false;
            this.hideProgress();
        }
    }

    // 验证输入
    validateInput() {
        if (this.currentMode === 'single' && !this.uploadedFiles.single) {
            this.showError('请上传图像文件');
            return false;
        }

        if (this.currentMode === 'batch' && this.uploadedFiles.thermal.length === 0) {
            this.showError('请上传红外图像文件');
            return false;
        }

        if (this.currentMode === 'data' && !this.uploadedFiles.data) {
            this.showError('请上传数据文件');
            return false;
        }

        return true;
    }

    // 处理单张图像
    async processSingleImage() {
        const formData = new FormData();
        formData.append('image', this.uploadedFiles.single);
        formData.append('mode', 'single');
        formData.append('confidence', document.getElementById('confidence-input').value);
        formData.append('temp_threshold', document.getElementById('temp-threshold').value);
        formData.append('device_type', document.getElementById('device-type').value);

        this.updateProgress(20, '正在上传图像...');

        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('检测请求失败');
        }

        this.updateProgress(60, '正在进行AI检测...');

        const result = await response.json();

        this.updateProgress(80, '正在分析温度数据...');

        // 温度诊断
        const diagnosisResponse = await fetch('/api/temperature_diagnosis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                detections: result.detections,
                image_name: this.uploadedFiles.single.name
            })
        });

        if (diagnosisResponse.ok) {
            const diagnosisResult = await diagnosisResponse.json();
            result.diagnosis = diagnosisResult;
        }

        this.updateProgress(100, '诊断完成');

        return result;
    }

    // 处理批量图像
    async processBatchImages() {
        const results = [];
        const totalFiles = this.uploadedFiles.thermal.length;

        for (let i = 0; i < totalFiles; i++) {
            const file = this.uploadedFiles.thermal[i];
            this.updateProgress((i / totalFiles) * 80, `正在处理第 ${i + 1}/${totalFiles} 张图像...`);

            try {
                const formData = new FormData();
                formData.append('image', file);
                formData.append('mode', 'batch');
                formData.append('confidence', document.getElementById('confidence-input').value);
                formData.append('temp_threshold', document.getElementById('temp-threshold').value);
                formData.append('device_type', document.getElementById('device-type').value);

                const response = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    result.filename = file.name;

                    // 温度诊断
                    const diagnosisResponse = await fetch('/api/temperature_diagnosis', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            detections: result.detections,
                            image_name: file.name
                        })
                    });

                    if (diagnosisResponse.ok) {
                        const diagnosisResult = await diagnosisResponse.json();
                        result.diagnosis = diagnosisResult;
                    }

                    results.push(result);
                }
            } catch (error) {
                console.error(`处理文件 ${file.name} 时出错:`, error);
            }
        }

        this.updateProgress(100, '批量诊断完成');
        return { batch_results: results };
    }

    // 处理数据文件
    async processDataFile() {
        const formData = new FormData();
        formData.append('data_file', this.uploadedFiles.data);
        formData.append('mode', 'data');
        formData.append('temp_threshold', document.getElementById('temp-threshold').value);
        formData.append('device_type', document.getElementById('device-type').value);

        this.updateProgress(30, '正在上传数据文件...');

        const response = await fetch('/api/analyze_data', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('数据分析请求失败');
        }

        this.updateProgress(70, '正在分析数据...');

        const result = await response.json();

        this.updateProgress(100, '数据分析完成');

        return result;
    }

    // 显示进度
    showProgress() {
        const container = document.getElementById('progress-container');
        if (container) {
            container.style.display = 'block';
        }
        
        // 滚动到进度条
        container?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // 更新进度
    updateProgress(percent, text) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill) {
            progressFill.style.width = percent + '%';
        }
        
        if (progressText) {
            progressText.textContent = text;
        }
    }

    // 隐藏进度
    hideProgress() {
        const container = document.getElementById('progress-container');
        if (container) {
            setTimeout(() => {
                container.style.display = 'none';
            }, 1000);
        }
    }

    // 显示错误
    showError(message) {
        // 可以使用更友好的错误提示
        alert('错误: ' + message);
    }

    // 显示结果
    displayResults(results) {
        const container = document.getElementById('results-container');
        const content = document.getElementById('results-content');
        
        if (!container || !content) return;

        content.innerHTML = '';

        if (this.currentMode === 'single') {
            this.displaySingleResult(results, content);
        } else if (this.currentMode === 'batch') {
            this.displayBatchResults(results, content);
        } else if (this.currentMode === 'data') {
            this.displayDataResults(results, content);
        }

        container.style.display = 'block';
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // 显示单张图像结果
    displaySingleResult(result, container) {
        const detections = result.detections || [];
        const diagnosis = result.diagnosis || {};

        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.innerHTML = `
            <div class="result-header">
                <div class="result-title">检测结果</div>
                <div class="result-confidence">置信度: ${result.confidence || 'N/A'}%</div>
            </div>
            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">检测设备数</div>
                    <div class="detail-value">${detections.length}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">异常设备数</div>
                    <div class="detail-value danger">${diagnosis.abnormal_count || 0}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">最高温度</div>
                    <div class="detail-value temperature">${diagnosis.max_temperature || 'N/A'}°C</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">平均温度</div>
                    <div class="detail-value">${diagnosis.avg_temperature || 'N/A'}°C</div>
                </div>
            </div>
        `;

        container.appendChild(resultCard);

        // 显示具体检测结果
        if (detections.length > 0) {
            const detectionsCard = document.createElement('div');
            detectionsCard.className = 'result-card';
            detectionsCard.innerHTML = `
                <div class="result-header">
                    <div class="result-title">设备详情</div>
                </div>
                <div class="detections-list">
                    ${detections.map(detection => `
                        <div class="detection-item">
                            <div class="detection-info">
                                <div class="detection-class">${detection.class}</div>
                                <div class="detection-confidence">置信度: ${(detection.confidence * 100).toFixed(1)}%</div>
                                <div class="detection-temp">温度: ${detection.temperature || 'N/A'}°C</div>
                                <div class="detection-status ${this.getTemperatureStatus(detection.temperature)}">
                                    ${this.getTemperatureStatusText(detection.temperature)}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;

            container.appendChild(detectionsCard);
        }

        // 显示诊断建议
        if (diagnosis.recommendations) {
            const recommendationsCard = document.createElement('div');
            recommendationsCard.className = 'result-card';
            recommendationsCard.innerHTML = `
                <div class="result-header">
                    <div class="result-title">诊断建议</div>
                </div>
                <div class="recommendations">
                    ${diagnosis.recommendations.map(rec => `
                        <div class="recommendation-item">
                            <i class="fas fa-lightbulb"></i>
                            <span>${rec}</span>
                        </div>
                    `).join('')}
                </div>
            `;

            container.appendChild(recommendationsCard);
        }
    }

    // 显示批量结果
    displayBatchResults(results, container) {
        const batchResults = results.batch_results || [];
        
        // 汇总统计
        const summary = this.calculateBatchSummary(batchResults);
        
        const summaryCard = document.createElement('div');
        summaryCard.className = 'result-card';
        summaryCard.innerHTML = `
            <div class="result-header">
                <div class="result-title">批量处理汇总</div>
            </div>
            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">处理图像数</div>
                    <div class="detail-value">${batchResults.length}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">检测设备总数</div>
                    <div class="detail-value">${summary.totalDetections}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">异常设备总数</div>
                    <div class="detail-value danger">${summary.totalAbnormal}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">最高温度</div>
                    <div class="detail-value temperature">${summary.maxTemperature}°C</div>
                </div>
            </div>
        `;

        container.appendChild(summaryCard);

        // 详细结果列表
        batchResults.forEach(result => {
            const detections = result.detections || [];
            const diagnosis = result.diagnosis || {};
            
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.innerHTML = `
                <div class="result-header">
                    <div class="result-title">${result.filename}</div>
                    <div class="result-confidence">设备数: ${detections.length}</div>
                </div>
                <div class="result-details">
                    <div class="detail-item">
                        <div class="detail-label">异常设备数</div>
                        <div class="detail-value ${diagnosis.abnormal_count > 0 ? 'danger' : 'normal'}">
                            ${diagnosis.abnormal_count || 0}
                        </div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">最高温度</div>
                        <div class="detail-value temperature">${diagnosis.max_temperature || 'N/A'}°C</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">平均温度</div>
                        <div class="detail-value">${diagnosis.avg_temperature || 'N/A'}°C</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">状态</div>
                        <div class="detail-value ${this.getOverallStatus(diagnosis)}">
                            ${this.getOverallStatusText(diagnosis)}
                        </div>
                    </div>
                </div>
            `;

            container.appendChild(resultCard);
        });
    }

    // 显示数据结果
    displayDataResults(results, container) {
        const analysis = results.analysis || {};
        
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.innerHTML = `
            <div class="result-header">
                <div class="result-title">数据分析结果</div>
            </div>
            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">数据点数</div>
                    <div class="detail-value">${analysis.total_points || 0}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">异常点数</div>
                    <div class="detail-value danger">${analysis.abnormal_points || 0}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">最高温度</div>
                    <div class="detail-value temperature">${analysis.max_temperature || 'N/A'}°C</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">平均温度</div>
                    <div class="detail-value">${analysis.avg_temperature || 'N/A'}°C</div>
                </div>
            </div>
        `;

        container.appendChild(resultCard);

        // 如果有趋势图数据，显示图表
        if (analysis.trend_data) {
            const chartCard = document.createElement('div');
            chartCard.className = 'result-card';
            chartCard.innerHTML = `
                <div class="result-header">
                    <div class="result-title">温度趋势图</div>
                </div>
                <div class="temperature-chart" id="temp-trend-result"></div>
            `;

            container.appendChild(chartCard);

            // 绘制趋势图
            setTimeout(() => {
                this.drawTrendChart('temp-trend-result', analysis.trend_data);
            }, 100);
        }
    }

    // 计算批量汇总
    calculateBatchSummary(results) {
        let totalDetections = 0;
        let totalAbnormal = 0;
        let maxTemperature = 0;
        let totalTemperature = 0;
        let temperatureCount = 0;

        results.forEach(result => {
            const detections = result.detections || [];
            const diagnosis = result.diagnosis || {};

            totalDetections += detections.length;
            totalAbnormal += diagnosis.abnormal_count || 0;

            if (diagnosis.max_temperature && diagnosis.max_temperature > maxTemperature) {
                maxTemperature = diagnosis.max_temperature;
            }

            if (diagnosis.avg_temperature) {
                totalTemperature += diagnosis.avg_temperature;
                temperatureCount++;
            }
        });

        return {
            totalDetections,
            totalAbnormal,
            maxTemperature: maxTemperature.toFixed(1),
            avgTemperature: temperatureCount > 0 ? (totalTemperature / temperatureCount).toFixed(1) : 'N/A'
        };
    }

    // 获取温度状态
    getTemperatureStatus(temperature) {
        if (!temperature) return 'normal';
        if (temperature > 80) return 'danger';
        if (temperature > 60) return 'warning';
        return 'normal';
    }

    // 获取温度状态文本
    getTemperatureStatusText(temperature) {
        if (!temperature) return '正常';
        if (temperature > 80) return '高温警告';
        if (temperature > 60) return '温度偏高';
        return '温度正常';
    }

    // 获取整体状态
    getOverallStatus(diagnosis) {
        if (diagnosis.abnormal_count > 0) return 'danger';
        if (diagnosis.max_temperature > 70) return 'warning';
        return 'normal';
    }

    // 获取整体状态文本
    getOverallStatusText(diagnosis) {
        if (diagnosis.abnormal_count > 0) return '需要关注';
        if (diagnosis.max_temperature > 70) return '温度偏高';
        return '状态良好';
    }

    // 绘制趋势图
    drawTrendChart(containerId, trendData) {
        const container = document.getElementById(containerId);
        if (!container || !trendData) return;

        const myChart = echarts.init(container);
        
        const option = {
            tooltip: {
                trigger: 'axis'
            },
            xAxis: {
                type: 'category',
                data: trendData.map(item => item.time || item.index)
            },
            yAxis: {
                type: 'value',
                name: '温度(°C)'
            },
            series: [{
                name: '温度',
                type: 'line',
                data: trendData.map(item => item.temperature),
                smooth: true,
                lineStyle: {
                    color: '#667eea'
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
                }
            }]
        };

        myChart.setOption(option);
    }

    // 重置表单
    resetForm() {
        // 清空文件
        this.uploadedFiles = {
            single: null,
            thermal: [],
            reference: [],
            data: null
        };

        // 清空文件列表
        document.querySelectorAll('.file-list').forEach(list => {
            list.innerHTML = '';
        });

        // 重置文件输入
        document.querySelectorAll('input[type="file"]').forEach(input => {
            input.value = '';
        });

        // 重置参数
        document.getElementById('confidence-slider').value = 0.5;
        document.getElementById('confidence-input').value = 0.5;
        document.getElementById('temp-threshold').value = 80;
        document.getElementById('device-type').value = 'all';

        // 隐藏结果和进度
        document.getElementById('results-container').style.display = 'none';
        document.getElementById('progress-container').style.display = 'none';

        // 重置模式
        this.selectMode('single');
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 检查是否在检测页面
    if (document.getElementById('temperature-diagnosis-container')) {
        new TemperatureDiagnosisSystem();
    }
});

// 导出给其他模块使用
window.TemperatureDiagnosisSystem = TemperatureDiagnosisSystem;
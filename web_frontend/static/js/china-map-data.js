// 中国地图数据加载器
let chinaMapGeoJSON = null;
let isMapDataLoaded = false;

// 设备数据
const equipmentData = {
    '北京': { value: 1234, status: 'normal' },
    '天津': { value: 567, status: 'warning' },
    '河北省': { value: 2345, status: 'normal' },
    '山西省': { value: 1234, status: 'danger' },
    '内蒙古自治区': { value: 890, status: 'normal' },
    '辽宁省': { value: 1567, status: 'normal' },
    '吉林省': { value: 987, status: 'warning' },
    '黑龙江省': { value: 765, status: 'normal' },
    '上海': { value: 2341, status: 'normal' },
    '江苏省': { value: 3456, status: 'normal' },
    '浙江省': { value: 2789, status: 'normal' },
    '安徽省': { value: 1678, status: 'normal' },
    '福建省': { value: 1432, status: 'normal' },
    '江西省': { value: 1123, status: 'normal' },
    '山东省': { value: 4567, status: 'normal' },
    '河南省': { value: 3234, status: 'normal' },
    '湖北省': { value: 2145, status: 'warning' },
    '湖南省': { value: 2567, status: 'normal' },
    '广东省': { value: 5432, status: 'normal' },
    '广西壮族自治区': { value: 1789, status: 'normal' },
    '海南省': { value: 456, status: 'normal' },
    '重庆': { value: 1345, status: 'normal' },
    '四川省': { value: 2678, status: 'danger' },
    '贵州省': { value: 1234, status: 'normal' },
    '云南省': { value: 1567, status: 'normal' },
    '西藏自治区': { value: 234, status: 'normal' },
    '陕西省': { value: 1456, status: 'normal' },
    '甘肃省': { value: 789, status: 'normal' },
    '青海省': { value: 345, status: 'normal' },
    '宁夏回族自治区': { value: 456, status: 'normal' },
    '新疆维吾尔自治区': { value: 678, status: 'warning' },
    '台湾省': { value: 1234, status: 'normal' },
    '香港特别行政区': { value: 789, status: 'normal' },
    '澳门特别行政区': { value: 123, status: 'normal' }
};

// 异步加载中国地图GeoJSON数据
async function loadChinaMapGeoJSON() {
    try {
        console.log('开始加载中国地图GeoJSON数据...');
        const response = await fetch('/api/china-geojson');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        chinaMapGeoJSON = await response.json();
        isMapDataLoaded = true;
        console.log('GeoJSON数据加载成功:', chinaMapGeoJSON.features ? chinaMapGeoJSON.features.length : 0, '个特征');
        
        // 触发地图数据加载完成事件
        const event = new CustomEvent('chinaMapDataReady', {
            detail: { data: chinaMapGeoJSON }
        });
        document.dispatchEvent(event);
        
        return chinaMapGeoJSON;
    } catch (error) {
        console.error('加载GeoJSON数据失败:', error);
        isMapDataLoaded = false;
        return null;
    }
}

// 获取地图数据
function getChinaMapGeoJSON() {
    return chinaMapGeoJSON;
}

// 检查数据是否已加载
function isMapDataReady() {
    return isMapDataLoaded && chinaMapGeoJSON !== null;
}

// 页面加载时自动开始加载地图数据
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadChinaMapGeoJSON);
} else {
    loadChinaMapGeoJSON();
}

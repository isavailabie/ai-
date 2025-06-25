function predict() {
    const dateInput = document.getElementById('date').value;
    if (!dateInput) {
        alert('请选择日期');
        return;
    }
    const resultDiv = document.getElementById('result');
    if (resultDiv) {
        resultDiv.innerHTML = '<div class="loading">正在预测...</div>';
    }

    fetch(`/api/predict?date=${dateInput}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            updateWeatherDisplay(data);
        })
        .catch(error => {
            if (resultDiv) {
                resultDiv.innerHTML = `<div class="error">预测失败: ${error.message}</div>`;
            }
        });
}

function updateWeatherDisplay(data) {
    const resultDiv = document.getElementById('result');
    if (!resultDiv) return;

    // 重置并设置样式
    resultDiv.className = 'result-content';

    const weatherClass = data.weather_class && data.weather_class[0] ? data.weather_class[0] : '晴';

    const weatherClassMap = {
        '晴': 'weather-sunny',
        '多云': 'weather-cloudy',
        '阴': 'weather-overcast',
        '小雨': 'weather-lightrain',
        '中雨': 'weather-moderaterain',
        '大雨': 'weather-heavyrain',
        '暴雨': 'weather-stormrain',
        '雾': 'weather-fog',
        '雨夹雪': 'weather-sleet',
        '小雪': 'weather-snowsmall',
    };

    const cssClass = weatherClassMap[weatherClass] || 'weather-sunny';
    resultDiv.classList.add(cssClass);

    // 给 body 也添加相应的天气样式类，先清除旧的
    document.body.className = document.body.className
        .split(' ')
        .filter(cls => !cls.startsWith('weather-'))
        .join(' ');
    document.body.classList.add(cssClass);

    // 辅助函数：生成两段小时数据时间和温度行
    function generateRows(predictions) {
        const first12 = predictions.slice(0, 12);
        const second12 = predictions.slice(12, 24);

        const timeRow1 = first12.map(pred =>
            `<div class="hourly-item time">${new Date(pred.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>`
        ).join('');
        const tempRow1 = first12.map(pred =>
            `<div class="hourly-item temp">${parseFloat(pred.temp).toFixed(1)}°C</div>`
        ).join('');

        const timeRow2 = second12.map(pred =>
            `<div class="hourly-item time">${new Date(pred.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>`
        ).join('');
        const tempRow2 = second12.map(pred =>
            `<div class="hourly-item temp">${parseFloat(pred.temp).toFixed(1)}°C</div>`
        ).join('');

        return { timeRow1, tempRow1, timeRow2, tempRow2 };
    }

    const patchtstRows = generateRows(data.predictions);
    const transformerRows = generateRows(data.transformer_predictions);
    const timesnetRows = generateRows(data.timesnet_predictions);

    const noonData = data.predictions.find(pred => new Date(pred.date).getHours() === 12);

    // 温度直接取 temp 字段，显示一位小数
    const tempAtNoon = noonData && noonData.temp !== undefined
        ? parseFloat(noonData.temp).toFixed(1)
        : '--';

    // 湿度直接取 humidity 字段，显示整数百分比
    const humidityAtNoon = noonData && noonData.humidity !== undefined
        ? `${parseFloat(noonData.humidity).toFixed(0)}%`
        : '--';

    // 用 u10, v10 计算风速和风向
    const windAtNoon = (noonData && noonData.u10 !== undefined && noonData.v10 !== undefined)
        ? (() => {
            const u = parseFloat(noonData.u10);
            const v = parseFloat(noonData.v10);
            const speed = Math.sqrt(u * u + v * v);
            if (isNaN(speed)) return '--';

            // 风向计算，转成“北、东北、东、东南、南、西南、西、西北”八方向
            // 角度 = 270 - atan2(v, u) * 180/PI，取模360
            let angle = (270 - Math.atan2(v, u) * 180 / Math.PI) % 360;
            if (angle < 0) angle += 360;

            const dirs = ['北', '东北', '东', '东南', '南', '西南', '西', '西北'];
            const idx = Math.round(angle / 45) % 8;

            return `${speed.toFixed(1)}m/s ${dirs[idx]}风`;
        })()
        : '--';

    resultDiv.innerHTML = `
        <div class="result-content">
            <p>📅 日期：${data.date}</p>
            <p>🌤 天气类型：${weatherClass}</p>
            <p>🌡 温度：${tempAtNoon}°C</p>
            <p>💧 湿度：${humidityAtNoon}</p>
            <p>💨 风：${windAtNoon}</p>

            <button onclick="toggleHourly()" id="toggle-button">展开每小时天气</button>
            <div class="hourly-predictions show" id="hourly-container">
                <div class="model-block">
                    <div class="hourly-item model-label patchtst">patchtst</div>
                    ${patchtstRows.timeRow1}
                    ${patchtstRows.tempRow1}
                    ${patchtstRows.timeRow2}
                    ${patchtstRows.tempRow2}
                </div>

                <div class="model-block">
                    <div class="hourly-item model-label transformer">transformer</div>
                    ${transformerRows.timeRow1}
                    ${transformerRows.tempRow1}
                    ${transformerRows.timeRow2}
                    ${transformerRows.tempRow2}
                </div>

                <div class="model-block">
                    <div class="hourly-item model-label timesnet">timesnet</div>
                    ${timesnetRows.timeRow1}
                    ${timesnetRows.tempRow1}
                    ${timesnetRows.timeRow2}
                    ${timesnetRows.tempRow2}
                </div>
            </div>
        </div>
    `;
}



function toggleHourly() {
    const container = document.getElementById('hourly-container');
    const button = document.getElementById('toggle-button');
    if (container.classList.contains('show')) {
        container.classList.remove('show');
        button.innerText = '展开每小时天气';
    } else {
        container.classList.add('show');
        button.innerText = '收起每小时天气';
    }
}

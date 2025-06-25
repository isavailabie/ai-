from flask import Flask, request, jsonify, render_template
import os
from model.patchtst.predict import predict_specific_day
from model.timesnet.predict import predict_with_timesnet
from model.transformer.predict import predict_with_transformer
from model.weather_classifier.classify_prediction import classify_prediction  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # 返回 templates/index.html

@app.route('/api/predict', methods=['GET'])
def api_predict():
    date = request.args.get('date')
    if not date:
        return jsonify({'error': '缺少日期参数'}), 400

    try:
        # 1. PatchTST 预测
        patchtst_model_path = os.path.join('model', 'patchtst', 'model_patchtst_1d.pth')
        patchtst_df = predict_specific_day(date, model_path=patchtst_model_path)
        patchtst_csv = os.path.join('model', 'data', f'prediction_{date}.csv')

        # 2. Transformer 预测
        transformer_df = predict_with_transformer(date)

        # 3. TimesNet 预测
        timesnet_df = predict_with_timesnet(date)

        # 4. 分类只用 PatchTST 的预测结果做分类
        weather_classes, daily_weather, daily_prob = classify_prediction(patchtst_csv)

        # 定义要返回的字段，确保包含风速u10,v10，温度，湿度
        return_fields = ['date', 'u10', 'v10', 'temp', 'humidity']

        # 5. 构造返回数据，包含风速分量
        return jsonify({
            'date': date,
            'predictions': patchtst_df[return_fields].to_dict(orient='records'),
            'transformer_predictions': transformer_df[return_fields].to_dict(orient='records'),
            'timesnet_predictions': timesnet_df[return_fields].to_dict(orient='records'),
            'weather_class': weather_classes,
            'probabilities': [round(daily_prob, 1)] * len(weather_classes),
            'daily_summary': {
                'type': daily_weather,
                'prob': round(daily_prob, 1)
            }
        })

    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

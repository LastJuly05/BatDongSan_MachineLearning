from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Tải mô hình và các metadata cần thiết
try:
    all_models = joblib.load('all_models.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    medians = joblib.load('medians.pkl') # Nạp giá trị trung bình thị trường
except:
    all_models = scaler = features = medians = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if all_models is None or scaler is None or features is None or medians is None:
        return jsonify({'error': 'Hệ thống chưa sẵn sàng. Vui lòng chạy train_models.py trước.'})

    try:
        # Lấy dữ liệu từ giao diện
        data = request.json
        area = float(data.get('area', 0))
        quality = int(data.get('quality', 5))
        year = int(data.get('year', 2000))
        rooms = int(data.get('rooms', 3))
        
        # 1. Tạo DataFrame với giá trị TRUNG VỊ thị trường làm nền
        input_df = pd.DataFrame([medians.values], columns=features)
        
        # 2. Cập nhật các thông số người dùng nhập vào
        if 'GrLivArea' in features: input_df['GrLivArea'] = area
        if 'OverallQual' in features: input_df['OverallQual'] = quality
        if 'YearBuilt' in features: input_df['YearBuilt'] = year
        if 'TotRmsAbvGrd' in features: input_df['TotRmsAbvGrd'] = rooms
        
        # 3. TÍNH TOÁN CÁC SIÊU ĐẶC TRƯNG (Đồng bộ với train_models.py)
        # Giả định năm bán là median của YrSold
        yr_sold = input_df['YrSold'].iloc[0]
        
        input_df['TotalSF'] = input_df['TotalBsmtSF'] + input_df['1stFlrSF'] + input_df['2ndFlrSF']
        # Cập nhật TotalSF dựa trên diện tích mới (GrLivArea xấp xỉ 1st+2nd)
        input_df['TotalSF'] = input_df['TotalBsmtSF'] + area 
        
        input_df['TotalBath'] = input_df['FullBath'] + (0.5 * input_df['HalfBath']) + \
                               input_df['BsmtFullBath'] + (0.5 * input_df['BsmtHalfBath'])
        
        input_df['HouseAge'] = yr_sold - year
        input_df['RemodAge'] = yr_sold - input_df['YearRemodAdd'].iloc[0]
        input_df['TotalQual'] = quality + input_df['OverallCond'].iloc[0]
        
        # Đảm bảo thứ tự cột đúng như lúc train
        input_df = input_df[features]
        
        # 4. Chuẩn hóa
        input_scaled = scaler.transform(input_df)
        
        # 5. Dự đoán bằng cả 5 thuật toán
        predictions = {}
        for name, model in all_models.items():
            pred_log = model.predict(input_scaled)[0]
            real_price = np.expm1(pred_log)
            predictions[name] = "{:,.0f}".format(real_price)
            
        return jsonify({'predictions': predictions})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

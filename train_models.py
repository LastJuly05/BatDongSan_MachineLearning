import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

def load_data():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)
    return df

def feature_engineering(df):
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

    # 2. Tạo đặc trưng mới (Siêu đặc trưng)
    # Tổng diện tích thực tế (Hầm + Tầng 1 + Tầng 2)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Tổng số phòng tắm
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    
    # Tuổi thọ của ngôi nhà khi bán
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # Thời gian kể từ khi sửa chữa lần cuối
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    # Chất lượng tổng thể (Kết hợp chất lượng và điều kiện)
    df['TotalQual'] = df['OverallQual'] + df['OverallCond']

    return df

def preprocess_data(df):
    # Thực hiện Feature Engineering
    df = feature_engineering(df)
    
    # Loại bỏ Id
    df = df.drop(['Id'], axis=1)
    
    # Log-Transformation cho SalePrice
    y = np.log1p(df['SalePrice'])
    X = df.drop('SalePrice', axis=1)    
    
    # Xử lý giá trị thiếu
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    categorical_cols = X.select_dtypes(include=['object', 'str']).columns
    X[categorical_cols] = X[categorical_cols].fillna('None')
    
    # One-Hot Encoding
    X = pd.get_dummies(X)
    
    # Lưu giá trị trung vị
    medians = X.median()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, medians

def train_advanced_models(X_train, y_train):
    param_grids = {
        "Lasso": {
            "model": Lasso(max_iter=10000),
            "params": {'alpha': [0.0005, 0.001, 0.01]}
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {'n_estimators': [200, 300], 'max_depth': [15, 20]}
        },
        "SVR": {
            "model": SVR(),
            "params": {'C': [15, 20], 'epsilon': [0.008, 0.01]}
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42),
            "params": {'n_estimators': [500], 'learning_rate': [0.05], 'max_depth': [3, 4]}
        },
        # FIX: Dùng solver='lbfgs' thay vì 'adam'
        # lbfgs phù hợp với dataset nhỏ/vừa (<10K samples), hội tụ ổn định
        # không cần early_stopping, không bị lỗi multiprocessing trên Windows
        "MLP": {
            "model": MLPRegressor(
                max_iter=5000,
                random_state=42,
                solver='lbfgs',   # Quasi-Newton, hội tụ tốt với dataset nhỏ
                tol=1e-4,
            ),
            "params": {
                'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                'alpha': [0.001, 0.01, 0.1],   # L2 regularization
                'activation': ['relu', 'tanh'],
            }
        }
    }
    
    best_models = {}
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
    
    for name, config in param_grids.items():
        print(f"Đang tối ưu {name}...")
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='r2',
            n_jobs=1          # Windows: dùng 1 process để tránh lỗi multiprocessing
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"  > R2 tối ưu (CV, log-space): {grid_search.best_score_:.4f}")
        if name == "MLP":
            print(f"  > Best params: {grid_search.best_params_}")
        
    return best_models

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names, medians = preprocess_data(df)
    
    trained_models = train_advanced_models(X_train, y_train)
    
    print("\n--- ĐANG LƯU KẾT QUẢ ---")
    joblib.dump(trained_models, 'all_models.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'features.pkl')
    joblib.dump(medians, 'medians.pkl')
    
    # Đánh giá trên log-space (thống nhất với CV) và giá trị gốc (MAE dễ hiểu)
    print("\nĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST:")
    print(f"{'Model':<15} {'R2 (log)':<14} {'R2 (gốc)':<14} {'MAE (gốc, USD)'}")
    print("-" * 60)
    
    y_test_orig = np.expm1(y_test)
    
    for name, model in trained_models.items():
        # Dự đoán ở log-space
        preds_log = model.predict(X_test)
        
        # R2 trên log-space (nhất quán với CV scoring)
        r2_log = r2_score(y_test, preds_log)
        
        # Chuyển về giá trị gốc
        preds_orig = np.expm1(preds_log)
        
        # R2 và MAE trên giá trị gốc
        r2_orig = r2_score(y_test_orig, preds_orig)
        mae_orig = mean_absolute_error(y_test_orig, preds_orig)
        
        print(f"{name:<15} {r2_log:<14.4f} {r2_orig:<14.4f} ${mae_orig:,.0f}")
        
    print("\n--- HOÀN TẤT ---")

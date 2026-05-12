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
from sklearn.metrics import r2_score

def load_data():
    file_path = 'train.csv'
    df = pd.read_csv(file_path)
    return df

def feature_engineering(df):
    # 1. Loại bỏ nhiễu (Outliers) - Kỹ thuật quan trọng trên Kaggle
    # Loại bỏ những căn nhà quá rộng (>4000 sqft) nhưng giá lại quá rẻ
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

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
    categorical_cols = X.select_dtypes(include=['object']).columns
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
        "MLP": {
            "model": MLPRegressor(max_iter=2000, random_state=42),
            "params": {'hidden_layer_sizes': [(100, 50)], 'alpha': [0.005]}
        }
    }
    
    best_models = {}
    print("\n--- BẮT ĐẦU HUẤN LUYỆN SIÊU CẤP (KAGGLE STYLE) ---")
    
    for name, config in param_grids.items():
        print(f"Đang tối ưu {name}...")
        grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"  > R2 tối ưu: {grid_search.best_score_:.4f}")
        
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
    
    print("\nĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST:")
    for name, model in trained_models.items():
        preds = np.expm1(model.predict(X_test))
        r2 = r2_score(np.expm1(y_test), preds)
        print(f"- {name}: {r2:.4f}")
        
    print("\n--- HOÀN TẤT! AI BÂY GIỜ ĐÃ LÀ MỘT CHUYÊN GIA THỰC THỤ ---")

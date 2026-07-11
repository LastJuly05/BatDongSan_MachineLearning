import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
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
    # Tránh thay đổi trực tiếp trên dataframe gốc bằng slice ẩn
    df = df.copy()

    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)

    # Tạo đặc trưng mới (Siêu đặc trưng)
    # Thêm fillna(0) đề phòng trường hợp các cột này bị trống trước khi cộng
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    df['1stFlrSF'] = df['1stFlrSF'].fillna(0)
    df['2ndFlrSF'] = df['2ndFlrSF'].fillna(0)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['TotalBath'] = (df['FullBath'].fillna(0) +
                       (0.5 * df['HalfBath'].fillna(0)) +
                       df['BsmtFullBath'].fillna(0) +
                       (0.5 * df['BsmtHalfBath'].fillna(0)))

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalQual'] = df['OverallQual'] + df['OverallCond']

    return df


def preprocess_data(df):
    # Thực hiện Feature Engineering
    df = feature_engineering(df)

    # Loại bỏ Id nếu có
    if 'Id' in df.columns:
        df = df.drop(['Id'], axis=1)

    # Tách X, y
    y = np.log1p(df['SalePrice'])
    X = df.drop('SalePrice', axis=1)

    # Xử lý giá trị thiếu cho Biến Số
    numeric_cols = X.select_dtypes(include=['number']).columns
    medians = X[numeric_cols].median()
    X[numeric_cols] = X[numeric_cols].fillna(medians)

    # Xử lý giá trị thiếu cho Biến Phân Loại
    categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns
    X[categorical_cols] = X[categorical_cols].fillna('None')

    # One-Hot Encoding toàn bộ X (Để nhất quán cho việc split)
    X = pd.get_dummies(X)
    feature_names = X.columns

    # Chia tập Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, medians


def train_advanced_models(X_train, y_train):
    # --- Chọn đặc trưng riêng cho SVR/MLP (giảm nhiễu do one-hot quá nhiều chiều) ---
    # SVR và MLP rất nhạy với dữ liệu nhiều chiều/thưa (sau one-hot có thể 200-300+ cột),
    # trong khi RandomForest/XGBoost tự động bỏ qua đặc trưng không quan trọng và Lasso tự
    # triệt tiêu hệ số về 0. Nên ta chọn ra top-k đặc trưng tốt nhất riêng cho SVR/MLP.
    k_features = min(60, X_train.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_train_selected = selector.fit_transform(X_train, y_train)

    param_grids = {
        "Lasso": {
            "model": Lasso(max_iter=50000),
            "params": {'alpha': [0.0001, 0.0005, 0.001, 0.005]},
            "use_selected": False
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                'n_estimators': [300, 500],
                'max_depth': [20, None],
                'max_features': [1.0, 'sqrt'],
                'min_samples_split': [2, 5]
            },
            "use_selected": False
        },
        "SVR": {
            "model": SVR(kernel='rbf'),
            "params": {
                'C': [1, 5, 10, 50, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.05, 0.1]
            },
            "use_selected": True
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42),
            "params": {
                'n_estimators': [500, 1000],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 5, 6],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            "use_selected": False
        },
        "MLP": {
            "model": MLPRegressor(
                max_iter=8000,
                random_state=42,
                solver='lbfgs',   # Quasi-Newton, hội tụ tốt với dataset nhỏ
                tol=1e-5,
            ),
            "params": {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.01, 0.1, 1.0, 5.0],   # tăng L2 regularization để chống overfit
                'activation': ['relu', 'tanh'],
            },
            "use_selected": True
        }
    }

    best_models = {}
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")

    for name, config in param_grids.items():
        print(f"Đang tối ưu {name}...")
        X_fit = X_train_selected if config["use_selected"] else X_train

        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='r2',
            n_jobs=1  # Giữ nguyên 1 để an toàn trên Windows/Kaggle notebook lẻ tẻ
        )
        grid_search.fit(X_fit, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"  > R2 tối ưu (CV, log-space): {grid_search.best_score_:.4f}")

    return best_models, selector


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names, medians = preprocess_data(df)

    trained_models, selector = train_advanced_models(X_train, y_train)

    print("\n--- ĐANG LƯU KẾT QUẢ ---")
    joblib.dump(trained_models, 'all_models.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(feature_names), 'features.pkl')  # Lưu dưới dạng list để tiện re-index sau này
    joblib.dump(medians, 'medians.pkl')
    joblib.dump(selector, 'selector.pkl')  # Cần để transform dữ liệu mới cho SVR/MLP lúc predict

    # ĐÁNH GIÁ CUỐI CÙNG
    print("\nĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST:")
    print(f"{'Model':<15} {'R2 (log)':<14} {'R2 (gốc)':<14} {'MAE (gốc, USD)'}")
    print("-" * 60)

    y_test_orig = np.expm1(y_test)
    models_using_selected_features = {"SVR", "MLP"}

    for name, model in trained_models.items():
        X_eval = selector.transform(X_test) if name in models_using_selected_features else X_test

        preds_log = model.predict(X_eval)
        r2_log = r2_score(y_test, preds_log)

        preds_orig = np.expm1(preds_log)
        r2_orig = r2_score(y_test_orig, preds_orig)
        mae_orig = mean_absolute_error(y_test_orig, preds_orig)

        print(f"{name:<15} {r2_log:<14.4f} {r2_orig:<14.4f} ${mae_orig:,.0f}")

    print("\n--- HOÀN TẤT ---")
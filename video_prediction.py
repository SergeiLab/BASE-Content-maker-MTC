import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Установите seed для воспроизводимости
SEED = 42
np.random.seed(SEED)

# 1. Data upload

train = pd.read_csv('interactions_train.csv')
users = pd.read_csv('users.csv')
items = pd.read_csv('items.csv')
test = pd.read_csv('interactions_public_test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Users shape: {users.shape}")
print(f"Items shape: {items.shape}")


# 2. EDA

# Создание целевой переменной
train['target'] = (train['watched_pct'] > 50).astype(int)
print(f"\nРаспределение целевой переменной:")
print(train['target'].value_counts(normalize=True))

# Базовая статистика
print("\nСтатистика обучающей выборки")
print(train.describe())

# Проверка пропусков
print("\nПропущенные значения")
print("Train:")
print(train.isnull().sum())
print("\nUsers:")
print(users.isnull().sum())
print("\nItems:")
print(items.isnull().sum())

# Анализ пользователей
print("\nАнализ пользователей")
print(f"Уникальных пользователей: {users['user_id'].nunique()}")
print("\nРаспределение по полу:")
print(users['sex'].value_counts())
print("\nРаспределение по возрасту:")
print(users['age'].value_counts())

# Анализ видео
print("\nАнализ видео")
print(f"Уникальных видео: {items['item_id'].nunique()}")
print("\nТипы контента:")
print(items['content_type'].value_counts())

# 3. Feature eng.

def create_features(df, users_df, items_df, is_train=True):
    """Создание признаков для модели"""
    
    df = df.copy()
    
    # Объединение с данными пользователей
    df = df.merge(users_df, on='user_id', how='left')
    
    # Объединение с данными видео
    df = df.merge(items_df, on='item_id', how='left')
    
    # Временные признаки
    if 'last_watch_dt' in df.columns:
        df['last_watch_dt'] = pd.to_datetime(df['last_watch_dt'])
        df['watch_hour'] = df['last_watch_dt'].dt.hour
        df['watch_dayofweek'] = df['last_watch_dt'].dt.dayofweek
        df['watch_month'] = df['last_watch_dt'].dt.month
        df['watch_day'] = df['last_watch_dt'].dt.day
        df['is_weekend'] = (df['watch_dayofweek'] >= 5).astype(int)
        df['is_evening'] = ((df['watch_hour'] >= 18) & (df['watch_hour'] <= 23)).astype(int)
        df['is_night'] = ((df['watch_hour'] >= 0) & (df['watch_hour'] <= 6)).astype(int)
    
    # Признаки длительности
    df['total_dur_minutes'] = df['total_dur'] / 60
    df['duration_category'] = pd.cut(df['total_dur_minutes'], 
                                      bins=[0, 30, 60, 90, 120, 1000],
                                      labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    # Признаки возраста
    df['age_numeric'] = df['age'].map({'18-24': 21, '25-34': 29.5, '35-44': 39.5, 
                                        '45-54': 49.5, '55-64': 59.5, '65+': 70})
    
    # Признаки дохода
    df['income_numeric'] = df['income'].map({'low': 1, 'medium': 2, 'high': 3})
    
    # Признаки года выпуска
    df['release_year_filled'] = df['release_year'].fillna(df['release_year'].median())
    df['years_since_release'] = 2024 - df['release_year_filled']
    df['is_new_content'] = (df['years_since_release'] <= 2).astype(int)
    df['is_classic'] = (df['years_since_release'] >= 20).astype(int)
    
    # Обработка жанров
    df['num_genres'] = df['genres'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['has_genre'] = (df['num_genres'] > 0).astype(int)
    
    # Популярные жанры
    popular_genres = ['драма', 'комедия', 'боевик', 'триллер', 'фантастика', 'мелодрама']
    for genre in popular_genres:
        df[f'genre_{genre}'] = df['genres'].fillna('').str.contains(genre, case=False).astype(int)
    
    # Обработка стран
    df['num_countries'] = df['countries'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['is_russian'] = df['countries'].fillna('').str.contains('Россия|СССР', case=False).astype(int)
    df['is_usa'] = df['countries'].fillna('').str.contains('США', case=False).astype(int)
    
    # Обработка актеров/режиссеров
    df['num_actors'] = df['actors'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['num_directors'] = df['directors'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['num_studios'] = df['studios'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['has_actors'] = (df['num_actors'] > 0).astype(int)
    
    # Соответствие контента возрасту пользователя
    df['age_rating_filled'] = df['age_rating'].fillna('18+')
    df['content_age_match'] = 1  
    
    # Детский контент
    df['kids_match'] = (df['kids_flg'] == df['for_kids']).astype(int)
    
    # Агрегированные признаки (если обучающая выборка)
    if is_train:
        # Статистика по пользователям
        user_stats = df.groupby('user_id').agg({
            'watched_pct': ['mean', 'std', 'count'],
            'total_dur': 'mean'
        }).reset_index()
        user_stats.columns = ['user_id', 'user_avg_watched', 'user_std_watched', 
                              'user_watch_count', 'user_avg_duration']
        
        # Статистика по видео
        item_stats = df.groupby('item_id').agg({
            'watched_pct': ['mean', 'std', 'count']
        }).reset_index()
        item_stats.columns = ['item_id', 'item_avg_watched', 'item_std_watched', 'item_watch_count']
        
        # Сохраняем для теста
        global user_stats_global, item_stats_global
        user_stats_global = user_stats
        item_stats_global = item_stats
    
    # Добавляем агрегированные признаки
    if 'user_stats_global' in globals():
        df = df.merge(user_stats_global, on='user_id', how='left')
        df = df.merge(item_stats_global, on='item_id', how='left')
    
    return df

print("Создание признаков для обучающей выборки...")
train_featured = create_features(train, users, items, is_train=True)

print("Создание признаков для тестовой выборки...")
test_featured = create_features(test, users, items, is_train=False)

print(f"\nКоличество признаков после feature engineering: {train_featured.shape[1]}")

# 4. Preparing data for the model

# Выбор признаков
feature_columns = [
    # Базовые
    'total_dur', 'total_dur_minutes',
    
    # Временные
    'watch_hour', 'watch_dayofweek', 'watch_month', 'is_weekend', 'is_evening', 'is_night',
    
    # Пользовательские
    'age_numeric', 'income_numeric', 'kids_flg',
    
    # Контент
    'release_year_filled', 'years_since_release', 'is_new_content', 'is_classic',
    'num_genres', 'has_genre', 'num_countries', 'is_russian', 'is_usa',
    'num_actors', 'num_directors', 'num_studios', 'has_actors',
    'for_kids', 'kids_match',
    
    # Жанры
    'genre_драма', 'genre_комедия', 'genre_боевик', 'genre_триллер', 
    'genre_фантастика', 'genre_мелодрама',
    
    # Агрегированные
    'user_avg_watched', 'user_std_watched', 'user_watch_count', 'user_avg_duration',
    'item_avg_watched', 'item_std_watched', 'item_watch_count'
]

# Категориальные признаки
cat_features = ['sex', 'content_type', 'duration_category']

# Для категориальных признаков
for col in cat_features:
    le = LabelEncoder()
    train_featured[col] = le.fit_transform(train_featured[col].astype(str))
    test_featured[col] = le.transform(test_featured[col].astype(str))
    feature_columns.append(col)

# Заполнение пропусков
for col in feature_columns:
    if col in train_featured.columns:
        train_featured[col] = train_featured[col].fillna(train_featured[col].median())
        test_featured[col] = test_featured[col].fillna(test_featured[col].median())

# Подготовка X и y
X = train_featured[feature_columns]
y = train_featured['target']

X_test = test_featured[feature_columns]

print(f"X shape: {X.shape}")
print(f"y distribution:\n{y.value_counts(normalize=True)}")

# 5. ML

# Разделение на train и validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")

# Модель 1: LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgbm.early_stopping(50), lgbm.log_evaluation(50)]
)

# Предсказания
y_pred_lgbm = lgbm_model.predict(X_val)
y_pred_proba_lgbm = lgbm_model.predict_proba(X_val)[:, 1]

print(f"\nF1 Score (macro): {f1_score(y_val, y_pred_lgbm, average='macro'):.4f}")
print(f"F1 Score (binary): {f1_score(y_val, y_pred_lgbm):.4f}")
print(f"ROC AUC: {roc_auc_score(y_val, y_pred_proba_lgbm):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_lgbm))

# Модель 2: Random Forest (дополнительная модель)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)
y_pred_proba_rf = rf_model.predict_proba(X_val)[:, 1]

print(f"F1 Score (macro): {f1_score(y_val, y_pred_rf, average='macro'):.4f}")
print(f"F1 Score (binary): {f1_score(y_val, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_val, y_pred_proba_rf):.4f}")

# Ансамбль (среднее предсказаний)
print("\n--- Ансамбль моделей (LightGBM + Random Forest) ---")
y_pred_proba_ensemble = (y_pred_proba_lgbm * 0.7 + y_pred_proba_rf * 0.3)
y_pred_ensemble = (y_pred_proba_ensemble > 0.5).astype(int)

print(f"F1 Score (macro): {f1_score(y_val, y_pred_ensemble, average='macro'):.4f}")
print(f"F1 Score (binary): {f1_score(y_val, y_pred_ensemble):.4f}")
print(f"ROC AUC: {roc_auc_score(y_val, y_pred_proba_ensemble):.4f}")

# Оптимизация порога для лучшего F1 macro
print("\n--- Оптимизация порога классификации ---")
thresholds = np.arange(0.35, 0.65, 0.01)
best_f1_macro = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_temp = (y_pred_proba_ensemble > threshold).astype(int)
    f1_macro = f1_score(y_val, y_pred_temp, average='macro')
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_threshold = threshold

print(f"Оптимальный порог: {best_threshold:.2f}")
print(f"F1 Score (macro) с оптимальным порогом: {best_f1_macro:.4f}")

# 6. Importance

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': lgbm_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТоп-20 важных признаков:")
print(feature_importance.head(20))

# 7. Pred

# Используем ансамбль для финального предсказания
test_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
test_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
test_pred_proba = (test_pred_proba_lgbm * 0.7 + test_pred_proba_rf * 0.3)

# Применяем оптимальный порог для бинарной классификации
test_pred_binary = (test_pred_proba > best_threshold).astype(int)

# Конвертируем в проценты (0-100)
# Используем вероятности напрямую для более точного представления
test_featured['watched_pct'] = test_pred_proba * 100

# Подготовка
submission = test_featured[['user_id', 'item_id', 'last_watch_dt', 'total_dur', 'watched_pct']]

# Сохранение результата
output_filename = f'result_seed_{SEED}.csv'
submission.to_csv(output_filename, index=False)
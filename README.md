# -
机器学习的作业
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

# 导入糖尿病数据集
data = pd.read_csv("E:\\excel\\data.csv")

# 对以下字段四舍五入
columns_to_round = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'Fruits',
                    'HeartDiseaseorAttack', 'PhysActivity', 'Veggies',
                    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                    'DiffWalk', 'Sex', 'Education']
data[columns_to_round] = data[columns_to_round].round().astype(int)

# 数据预处理
data_X = data.drop(['target', 'id'], axis=1)  # 去除target,id两列
data_y = data['target']

# 拆分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)

# SMOTE: 合成少数类过采样技术
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 数据标准化
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# 随机森林模型
rf = RandomForestClassifier(random_state=42)

# 定义要调优的超参数范围
param_dist = {
    'n_estimators': randint(50, 200),  # 树的数量
    'max_depth': randint(5, 20),        # 最大深度
    'min_samples_split': randint(2, 10),  # 最小分裂样本数
    'min_samples_leaf': randint(1, 5),    # 最小叶子节点样本数
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': ['balanced', None],
}

# 使用RandomizedSearchCV进行参数调优，选择最佳参数
grid_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=50, cv=5, scoring='f1_macro', n_jobs=-1, random_state=42)
grid_search.fit(X_train_resampled, y_train_resampled)

# 输出最佳参数和对应的F1分数
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.3f}")

# 获取最佳模型
best_rf = grid_search.best_estimator_

# 进行预测
y_pred = best_rf.predict(X_test)

# 计算 F1 分数和准确度
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

# 输出结果
print(f"F1 score: {f1:.3f}")
print(f"Accuracy score: {accuracy:.3f}")

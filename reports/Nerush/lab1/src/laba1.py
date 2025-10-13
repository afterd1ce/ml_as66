import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 1. Загрузка данных
df = pd.read_csv("E:/Projects/ml_as66/reports/Nerush/lab1/src/german_credit.csv")
print("🔹 Информация о данных:")
print(df.info())

# 2. Исследовательский анализ
print("\n🔹 Количество пропущенных значений:")
print(df.isnull().sum())

print("\n🔹 Статистика:")
print(df.describe())

print("\n🔹 Медианы:")
print(df.median(numeric_only=True))

print("\n🔹 Стандартные отклонения:")
print(df.std(numeric_only=True))

# 3. Обработка пропущенных значений
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. One-Hot Encoding для категориальных признаков
df_encoded = pd.get_dummies(df, columns=['personal_status_sex', 'housing'], drop_first=True)

# 5. Нормализация числовых признаков
scaler = MinMaxScaler()
num_cols = ['age', 'credit_amount', 'duration_in_month']
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# 6. Визуализация

# 6.1 Гистограмма целей кредита
plt.figure(figsize=(8, 5))
purpose_counts = df['purpose'].value_counts().head(5)
sns.barplot(x=purpose_counts.index, y=purpose_counts.values, palette="Blues")
plt.title("Топ-5 целей кредита")
plt.ylabel("Количество")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.2 Ящик с усами по credit_amount
plt.figure(figsize=(8, 5))
sns.boxplot(x='default', y='credit_amount', data=df, palette="Set2")
plt.title("Сравнение суммы кредита по кредитоспособности")
plt.xlabel("Кредитоспособность (0 = плохой, 1 = хороший)")
plt.ylabel("Сумма кредита")
plt.tight_layout()
plt.show()

# 6.3 Диаграмма рассеяния age vs duration_in_month
plt.figure(figsize=(8, 5))
sns.scatterplot(x='age', y='duration_in_month', hue='default', data=df, palette="coolwarm")
plt.title("Возраст vs Длительность кредита")
plt.tight_layout()
plt.show()

# 7. Сводная таблица по credit_history
pivot_table = df.pivot_table(
    values=['age', 'duration_in_month'],
    index='credit_history',
    aggfunc='mean'
)
print("\n🔹 Средний возраст и длительность кредита по кредитной истории:")
print(pivot_table)

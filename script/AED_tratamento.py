import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/GusRilima/ProjAplicado02/main/dataset/bank_transactions_data_2.csv"
df = pd.read_csv(url)

df.columns = [c.strip().replace(" ", "_") for c in df.columns]

for col in ["TransactionDate", "PreviousTransactionDate"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

if {"TransactionDate", "PreviousTransactionDate"}.issubset(df.columns):
    df["MinutesSincePrevious"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.total_seconds() / 60
else:
    df["MinutesSincePrevious"] = np.nan

if "TransactionDate" in df.columns:
    df["TransactionHour"] = df["TransactionDate"].dt.hour
    df["TransactionMonth"] = df["TransactionDate"].dt.month
    df["IsNightTransaction"] = df["TransactionHour"].between(0, 5) | df["TransactionHour"].between(22, 23)
    df["IsNightTransaction"] = df["IsNightTransaction"].astype(int)
else:
    df["TransactionHour"] = np.nan
    df["TransactionMonth"] = np.nan
    df["IsNightTransaction"] = 0

for col in ["MinutesSincePrevious", "TransactionHour", "TransactionMonth"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

print("=" * 70)
print("DIMENSÕES DA BASE")
print(df.shape)

print("\nPRIMEIRAS LINHAS")
print(df.head())

print("\nTIPOS DE DADOS")
print(df.dtypes)

print("\nVALORES AUSENTES")
print(df.isnull().sum())

print("\nDUPLICATAS")
print(df.duplicated().sum())

print("\nESTATÍSTICAS DESCRITIVAS - NUMÉRICAS")
print(df.describe(include=[np.number]).T)

print("\nESTATÍSTICAS DESCRITIVAS - CATEGÓRICAS")
print(df.describe(include=["object"]).T)

amount = df["TransactionAmount"].dropna()
media = amount.mean()
desvio = amount.std(ddof=1)
n = amount.shape[0]

if n > 1:
    ic = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=media,
        scale=desvio / np.sqrt(n)
    )
    print("\nINTERVALO DE CONFIANÇA (95%) PARA TransactionAmount")
    print(f"Média: {media:.2f}")
    print(f"IC 95%: ({ic[0]:.2f}, {ic[1]:.2f})")

if {"TransactionAmount", "AccountBalance"}.issubset(df.columns):
    corr_df = df[["TransactionAmount", "AccountBalance"]].dropna()
    if len(corr_df) > 1:
        r, p = stats.pearsonr(corr_df["TransactionAmount"], corr_df["AccountBalance"])
        print("\nCORRELAÇÃO ENTRE TransactionAmount E AccountBalance")
        print(f"Correlação de Pearson: {r:.4f}")
        print(f"p-valor: {p:.4f}")

if {"Channel", "TransactionAmount"}.issubset(df.columns):
    grupos = [
        grupo["TransactionAmount"].dropna().values
        for _, grupo in df.groupby("Channel")
        if grupo["TransactionAmount"].dropna().shape[0] > 1
    ]
    if len(grupos) >= 2:
        f_stat, p_val = stats.f_oneway(*grupos)
        print("\nANOVA - TransactionAmount POR Channel")
        print(f"Estatística F: {f_stat:.4f}")
        print(f"p-valor: {p_val:.4f}")

sns.set(style="whitegrid")

for col in ["TransactionAmount", "AccountBalance", "LoginAttempts"]:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribuição de {col}")
        plt.tight_layout()
        plt.show()

for col in ["TransactionAmount", "MinutesSincePrevious"]:
    if col in df.columns:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.tight_layout()
        plt.show()

for col in ["TransactionType", "Channel"]:
    if col in df.columns:
        plt.figure(figsize=(9, 4))
        ordem = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=ordem)
        plt.title(f"Frequência de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if "TransactionMonth" in df.columns:
    plt.figure(figsize=(8, 4))
    df["TransactionMonth"].value_counts().sort_index().plot(kind="bar")
    plt.title("Quantidade de transações por mês")
    plt.xlabel("Mês")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.show()

if {"TransactionAmount", "AccountBalance"}.issubset(df.columns):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="AccountBalance", y="TransactionAmount")
    plt.title("TransactionAmount x AccountBalance")
    plt.tight_layout()
    plt.show()

corr_cols = [
    "TransactionAmount",
    "AccountBalance",
    "TransactionDuration",
    "LoginAttempts",
    "CustomerAge",
    "MinutesSincePrevious",
    "TransactionHour"
]
corr_cols = [c for c in corr_cols if c in df.columns]

if len(corr_cols) > 1:
    plt.figure(figsize=(9, 5))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matriz de correlação")
    plt.tight_layout()
    plt.show()

df["AmountZScore"] = np.abs(stats.zscore(df["TransactionAmount"], nan_policy="omit"))
df["AmountZScore"] = pd.Series(df["AmountZScore"], index=df.index).fillna(0)

online_flag = pd.Series(False, index=df.index)
if "Channel" in df.columns:
    online_flag = df["Channel"].astype(str).str.lower().str.contains("online", na=False)

df["ReferenceFlag"] = 0
df.loc[df["AmountZScore"] >= 2, "ReferenceFlag"] = 1

if "LoginAttempts" in df.columns:
    df.loc[df["LoginAttempts"] >= 3, "ReferenceFlag"] = 1

df["RiskScore"] = 0
df["RiskScore"] += (df["AmountZScore"] >= 2).astype(int)

if "LoginAttempts" in df.columns:
    df["RiskScore"] += (df["LoginAttempts"] >= 3).astype(int)

if "MinutesSincePrevious" in df.columns:
    df["RiskScore"] += (df["MinutesSincePrevious"] <= 10).astype(int)

df["RiskScore"] += ((df["IsNightTransaction"] == 1) & online_flag).astype(int)

df["RiskFlag"] = (df["RiskScore"] >= 2).astype(int)

proxy_accuracy = accuracy_score(df["ReferenceFlag"], df["RiskFlag"])

print("\nAVALIAÇÃO EXPLORATÓRIA")
print(f"Acurácia proxy: {proxy_accuracy:.4f}")
print("Essa medida é apenas exploratória, pois a base não possui rótulo real de fraude.")

top_risco = df.sort_values("RiskScore", ascending=False).head(20)

print("\nTOP 20 TRANSAÇÕES COM MAIOR SCORE DE RISCO")
cols_show = [
    "TransactionAmount",
    "AccountBalance",
    "TransactionType",
    "Channel",
    "Location",
    "LoginAttempts",
    "CustomerAge",
    "MinutesSincePrevious",
    "RiskScore",
    "RiskFlag"
]
cols_show = [c for c in cols_show if c in top_risco.columns]
print(top_risco[cols_show])

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="RiskFlag")
plt.title("Distribuição do indicador de risco")
plt.tight_layout()
plt.show()

print("\nCONCLUSÃO RESUMIDA")
print("1. A base foi carregada e inspecionada.")
print("2. Foram criadas variáveis temporais derivadas.")
print("3. A AED analisou distribuições, frequências e correlações.")
print("4. O script inclui intervalo de confiança, correlação e ANOVA.")
print("5. O resultado final é um score de risco exploratório para cada transação.")
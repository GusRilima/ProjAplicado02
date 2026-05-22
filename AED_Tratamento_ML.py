import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import accuracy_score
# Adicionando bibliotecas para a modelagem não supervisionada
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# ETAPA 1 - LEITURA DA BASE
# ============================================================
# Essa etapa inicia a inspeção geral da base.
url = "https://raw.githubusercontent.com/GusRilima/ProjAplicado02/main/dataset/bank_transactions_data_2.csv"
df = pd.read_csv(url)

# ============================================================
# ETAPA 2 - PADRONIZAÇÃO E TRATAMENTO INICIAL
# ============================================================
# Aqui os nomes das colunas são ajustados para facilitar o uso

df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Conversão das colunas de data para formato datetime.
for col in ["TransactionDate", "PreviousTransactionDate"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Criação da variável com o intervalo, em minutos,
# entre a transação atual e a transação anterior.
if {"TransactionDate", "PreviousTransactionDate"}.issubset(df.columns):
    df["MinutesSincePrevious"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.total_seconds() / 60
else:
    df["MinutesSincePrevious"] = np.nan

# Extração de componentes temporais da data principal.
# Essas variáveis ajudam a analisar o comportamento ao longo do tempo.
if "TransactionDate" in df.columns:
    df["TransactionHour"] = df["TransactionDate"].dt.hour
    df["TransactionMonth"] = df["TransactionDate"].dt.month

    # Identificação de transações noturnas.
    # Foi considerado horário noturno entre 22h e 5h.
    df["IsNightTransaction"] = (
        df["TransactionHour"].between(0, 5) | df["TransactionHour"].between(22, 23)
    ).astype(int)
else:
    df["TransactionHour"] = np.nan
    df["TransactionMonth"] = np.nan
    df["IsNightTransaction"] = 0

# Preenchimento simples de ausências criadas nas variáveis derivadas.
# A mediana foi usada por ser menos sensível a valores extremos.
for col in ["MinutesSincePrevious", "TransactionHour", "TransactionMonth"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ============================================================
# ETAPA 3 - INSPEÇÃO INICIAL DA BASE
# ============================================================
# Exibição de informações básicas para entender a estrutura,
# qualidade e composição inicial do conjunto de dados.
print("=" * 70)
print("ETAPA 1 - LEITURA E INSPEÇÃO INICIAL")
print("Objetivo: entender a estrutura da base e verificar sua qualidade.")
print("=" * 70)

print("\nDimensões da base:")
print(df.shape)

print("\nPrimeiras linhas:")
print(df.head())

print("\nTipos de dados:")
print(df.dtypes)

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

print("\nDuplicatas:")
print(df.duplicated().sum())

print("\nEstatísticas descritivas - numéricas")
print(df.describe(include=[np.number]).T)

print("\nEstatísticas descritivas - categóricas")
print(df.describe(include=["object"]).T)

# ============================================================
# ETAPA 4 - ANÁLISE ESTATÍSTICA
# ============================================================
# Esta etapa aplica medidas inferenciais simples para complementar
# a AED e tornar a análise mais consistente.
print("\n" + "=" * 70)
print("ETAPA 2 - ANÁLISE ESTATÍSTICA")
print("Objetivo: resumir o comportamento da base e verificar relações iniciais.")
print("=" * 70)

# Intervalo de confiança para a média do valor das transações.
# Isso fornece uma faixa provável para a média populacional.
if "TransactionAmount" in df.columns:
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
        print("\nIntervalo de confiança (95%) para TransactionAmount")
        print("Explicação: estima uma faixa provável para a média do valor das transações.")
        print(f"Média: {media:.2f}")
        print(f"IC 95%: ({ic[0]:.2f}, {ic[1]:.2f})")

# Correlação de Pearson entre valor da transação e saldo.
# Essa medida indica o grau de associação linear entre as duas variáveis.
if {"TransactionAmount", "AccountBalance"}.issubset(df.columns):
    corr_df = df[["TransactionAmount", "AccountBalance"]].dropna()
    if len(corr_df) > 1:
        r, p = stats.pearsonr(corr_df["TransactionAmount"], corr_df["AccountBalance"])
        print("\nCorrelação entre TransactionAmount e AccountBalance")
        print("Explicação: mede a intensidade da relação linear entre valor da transação e saldo.")
        print(f"Correlação de Pearson: {r:.4f}")
        print(f"p-valor: {p:.4f}")

# ANOVA para comparar o valor médio das transações entre canais.
# A ideia é verificar se o canal influencia o comportamento do valor.
if {"Channel", "TransactionAmount"}.issubset(df.columns):
    grupos = [
        grupo["TransactionAmount"].dropna().values
        for _, grupo in df.groupby("Channel")
        if grupo["TransactionAmount"].dropna().shape[0] > 1
    ]
    if len(grupos) >= 2:
        f_stat, p_val = stats.f_oneway(*grupos)
        print("\nANOVA - TransactionAmount por Channel")
        print("Explicação: verifica se a média do valor das transações muda entre os canais.")
        print(f"Estatística F: {f_stat:.4f}")
        print(f"p-valor: {p_val:.4f}")

# ============================================================
# ETAPA 5 - VISUALIZAÇÕES DA AED
# ============================================================
# Os gráficos apoiam a interpretação visual de distribuição,
# frequência, dispersão e correlação.
print("\n" + "=" * 70)
print("ETAPA 3 - VISUALIZAÇÕES")
print("Objetivo: identificar padrões, dispersão e possíveis comportamentos atípicos.")
print("=" * 70)

sns.set(style="whitegrid")

# Histogramas para variáveis numéricas principais.
for col in ["TransactionAmount", "AccountBalance", "LoginAttempts"]:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribuição de {col}")
        plt.tight_layout()
        plt.show()

# Boxplots para observar dispersão e valores extremos.
for col in ["TransactionAmount", "MinutesSincePrevious"]:
    if col in df.columns:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.tight_layout()
        plt.show()

# Gráficos de frequência para variáveis categóricas.
for col in ["TransactionType", "Channel"]:
    if col in df.columns:
        plt.figure(figsize=(9, 4))
        ordem = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=ordem)
        plt.title(f"Frequência de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Volume de transações por mês.
if "TransactionMonth" in df.columns:
    plt.figure(figsize=(8, 4))
    df["TransactionMonth"].value_counts().sort_index().plot(kind="bar")
    plt.title("Quantidade de transações por mês")
    plt.xlabel("Mês")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.show()

# Dispersão entre saldo e valor da transação.
if {"TransactionAmount", "AccountBalance"}.issubset(df.columns):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="AccountBalance", y="TransactionAmount")
    plt.title("TransactionAmount x AccountBalance")
    plt.tight_layout()
    plt.show()

# Matriz de correlação para variáveis numéricas relevantes.
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

# ============================================================
# ETAPA 6 - SCORE DE RISCO EXPLORATÓRIO
# ============================================================
# Como não existe rótulo real de fraude, o script cria um
# score simples para destacar transações potencialmente atípicas.
print("\n" + "=" * 70)
print("ETAPA 4 - SCORE DE RISCO EXPLORATÓRIO")
print("Objetivo: sinalizar transações potencialmente atípicas sem usar rótulo real de fraude.")
print("=" * 70)

# Z-score do valor da transação.
# Quanto maior o desvio em relação ao comportamento médio,
# maior a chance de a operação ser considerada atípica.
if "TransactionAmount" in df.columns:
    df["AmountZScore"] = np.abs(stats.zscore(df["TransactionAmount"], nan_policy="omit"))
    df["AmountZScore"] = pd.Series(df["AmountZScore"], index=df.index).fillna(0)
else:
    df["AmountZScore"] = 0

# Identificação de operações online.
online_flag = pd.Series(False, index=df.index)
if "Channel" in df.columns:
    online_flag = df["Channel"].astype(str).str.lower().str.contains("online", na=False)

# Indicador de referência exploratório.
# Ele serve apenas como comparação simples para a avaliação final.
df["ReferenceFlag"] = 0
df.loc[df["AmountZScore"] >= 2, "ReferenceFlag"] = 1

if "LoginAttempts" in df.columns:
    df.loc[df["LoginAttempts"] >= 3, "ReferenceFlag"] = 1

# Construção do score de risco.
# Cada critério soma 1 ponto ao risco da transação.
df["RiskScore"] = 0
df["RiskScore"] += (df["AmountZScore"] >= 2).astype(int)

if "LoginAttempts" in df.columns:
    df["RiskScore"] += (df["LoginAttempts"] >= 3).astype(int)

if "MinutesSincePrevious" in df.columns:
    df["RiskScore"] += (df["MinutesSincePrevious"] <= 10).astype(int)

df["RiskScore"] += ((df["IsNightTransaction"] == 1) & online_flag).astype(int)

# Transformação do score em indicador binário.
# Transações com 2 ou mais sinais passam a ser marcadas como risco.
df["RiskFlag"] = (df["RiskScore"] >= 2).astype(int)

# Avaliação exploratória do indicador.
# Como não há fraude confirmada, trata-se apenas de uma comparação
# com a referência simples criada anteriormente.
proxy_accuracy = accuracy_score(df["ReferenceFlag"], df["RiskFlag"])

print("\nAvaliação exploratória")
print("Explicação: como não há rótulo real de fraude, a comparação é feita com uma referência simples.")
print(f"Acurácia proxy: {proxy_accuracy:.4f}")
print("Essa medida é apenas exploratória.")

# Exibição das transações com maior score de risco.
top_risco = df.sort_values("RiskScore", ascending=False).head(20)

print("\nTop 20 transações com maior score de risco")
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

# Visualização final da distribuição do indicador de risco.
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="RiskFlag")
plt.title("Distribuição do indicador de risco")
plt.tight_layout()
plt.show()

# ============================================================
# ETAPA 7 - APLICAÇÃO DE MACHINE LEARNING (ISOLATION FOREST)
# ============================================================
# Como a acurácia calculada acima usa uma referência criada na mão,
# decidimos aplicar um modelo não supervisionado para validar os achados
# e encontrar anomalias que as regras manuais podem ter deixado passar.
print("\n" + "=" * 70)
print("ETAPA 5 - MODELAGEM COM ISOLATION FOREST")
print("Objetivo: Detectar anomalias usando Machine Learning não supervisionado.")
print("=" * 70)

# Selecionando variáveis numéricas relevantes para o modelo
features_ml = ["TransactionAmount", "LoginAttempts", "MinutesSincePrevious", "TransactionHour"]
features_ml = [f for f in features_ml if f in df.columns]

if len(features_ml) > 0:
    # Preenchendo eventuais nulos restantes com a mediana
    df_ml = df[features_ml].fillna(df[features_ml].median())
    
    # Padronizando os dados (importante para algoritmos baseados em distâncias ou densidade)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_ml)

    # Treinando o modelo assumindo 5% de contaminação (taxa estimada de anomalias/fraudes)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df["IsoForest_Label"] = iso_forest.fit_predict(df_scaled)
    
    # O modelo retorna -1 para anomalia e 1 para normal.
    # Convertendo para 1 (Risco) e 0 (Normal) para manter o padrão anterior
    df["RiskFlag_ML"] = df["IsoForest_Label"].apply(lambda x: 1 if x == -1 else 0)
    
    # Criando um alerta final cruzando a nossa heurística mais forte (Score >= 3) com o ML
    df["Final_Risk_Alert"] = df["RiskFlag_ML"] | (df["RiskScore"] >= 3).astype(int)

    print(f"\nTotal de transações na base: {len(df)}")
    print(f"Transações marcadas como risco pela Heurística: {df['RiskFlag'].sum()}")
    print(f"Transações marcadas como risco pelo Modelo (ML): {df['RiskFlag_ML'].sum()}")
    print(f"Total de Alertas Finais gerados (Híbrido): {df['Final_Risk_Alert'].sum()}")

    # Avaliando o perfil dos grupos em vez de usar a acurácia proxy
    print("\nPerfil médio das transações (Normal vs Risco Final):")
    perfil = df.groupby("Final_Risk_Alert")[features_ml].mean().round(2)
    print(perfil)
    
    print("\nTop 5 transações mais suspeitas detectadas pelo modelo:")
    top_suspeitas = df[df["Final_Risk_Alert"] == 1].sort_values("TransactionAmount", ascending=False).head(5)
    cols_show = ["TransactionAmount", "LoginAttempts", "MinutesSincePrevious", "IsNightTransaction", "RiskScore", "Final_Risk_Alert"]
    print(top_suspeitas[[c for c in cols_show if c in top_suspeitas.columns]])


# ============================================================
# ETAPA 8 - CONCLUSÃO
# ============================================================
# Resumo textual do que foi feito ao longo do script.
print("\nCONCLUSÃO RESUMIDA")
print("1. A base foi carregada, inspecionada e tratada nas variáveis temporais.")
print("2. A AED analisou distribuições, frequências e correlações.")
print("3. O script inclui medidas de estatística inferencial compatíveis com a proposta da entrega.")
print("4. O resultado final combina um score heurístico com um modelo de Isolation Forest para melhor detecção de anomalias.")

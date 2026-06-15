"""
Trabalho Final - Previsão de Churn de Clientes de Telecomunicações
Disciplina de Inteligência Artificial - Unicesumar 2026 - Professor Munif Gebara

Integrantes:
- Micaela
- Miguel
- Nathan
- Igor

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import joblib

# Configurações gerais 
SEED = 42 # variável de aleatoriedade (bagging), é definida para ser a mesma para os dois treinos
os.makedirs("graficos", exist_ok=True)
os.makedirs("modelos", exist_ok=True)

#  1. ---- CARREGAMENTO E PREPARAÇÃO DOS DADOS 
print("=" * 60)
print("1. Carregando e preparando os dados...")
print("=" * 60)

# ---- Leitura do csv ----
df = pd.read_csv("telco_churn.csv")
print(f"   Registros: {len(df)}") # conta linhas
print(f"   Atributos: {df.shape[1]}") # conta colunas
print(f"   Churn (%):\n{df['Churn'].value_counts(normalize=True).mul(100).round(1)}") # mostra proporção das classes - 73,6% não cancelaram, 26,4% cancelaram, ou seja, é um dataset desbalanceado 

# ---- Preparação da base ----
df = df.drop(columns=["customerID"]) # eliminar inuteis
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # converte pra número, totalCharges é o total de cobranças do cliente acumulado
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median()) # tranforma nulos na média

le = LabelEncoder() # tranforma valores em números, sim vira 1, não vira 0
cat_cols = df.select_dtypes(include="object").columns.tolist() # encontra todas as colunas de texto
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# separação da base entre treino e teste
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n   Treino: {len(X_train)} | Teste: {len(X_test)}")


#  2. ---- TREINAMENTO - ÁRVORE DE DECISÃO (Parte 1) 
print("\n" + "=" * 60)
print("2. Treinando Árvore de Decisão (Parte 1)...")
print("=" * 60)

# -----------------------------
dt = DecisionTreeClassifier(
    max_depth=8, min_samples_leaf=20,   # profundidade da árvore = 8, min_samples = cada decisão precisa ter ao menos 20 clientes
    class_weight="balanced", random_state=SEED    # class_weight = balanceia os casos com variância extrema
)                                                 # random_state = quando tiver decisões aleatórias elas são limitadas a serem as mesmas (para gerar os mesmos resultados)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

acc_dt  = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt  = recall_score(y_test, y_pred_dt)
f1_dt   = f1_score(y_test, y_pred_dt)

# -----------------------------

print(f"   Acurácia : {acc_dt:.4f}")
print(f"   Precisão : {prec_dt:.4f}")
print(f"   Revocação: {rec_dt:.4f}")
print(f"   F1-Score : {f1_dt:.4f}")

joblib.dump(dt, "modelos/arvore_decisao.pkl")


# ---- 3. TREINAMENTO - RANDOM FOREST - BAGGING (Parte 2) 
print("\n" + "=" * 60)
print("3. Treinando Random Forest (Parte 2 - Bagging/Ensemble)...")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=5, # conjunto de 200 árvores diferentes com profundidade de 12, e pelo menos 5 clientes
    class_weight="balanced", random_state=SEED
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf  = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf  = recall_score(y_test, y_pred_rf)
f1_rf   = f1_score(y_test, y_pred_rf)

print(f"   Acurácia : {acc_rf:.4f}")
print(f"   Precisão : {prec_rf:.4f}")
print(f"   Revocação: {rec_rf:.4f}")
print(f"   F1-Score : {f1_rf:.4f}")

joblib.dump(rf, "modelos/random_forest.pkl")


# ---- 4. GRÁFICOS DE AVALIAÇÃO 
print("\n" + "=" * 60)
print("4. Gerando gráficos de avaliação...")
print("=" * 60)

metricas = ["Acurácia", "Precisão", "Revocação", "F1-Score"]
vals_dt  = [acc_dt, prec_dt, rec_dt, f1_dt]
vals_rf  = [acc_rf, prec_rf, rec_rf, f1_rf]

# --- Gráfico 1: Comparação de Métricas ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metricas))
w = 0.35
bars1 = ax.bar(x - w/2, vals_dt, w, label="Árvore de Decisão (Parte 1)",
               color="#7F77DD", alpha=0.90)
bars2 = ax.bar(x + w/2, vals_rf, w, label="Random Forest (Parte 2)",
               color="#1D9E75", alpha=0.90)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=11)
ax.set_ylabel("Valor", fontsize=11)
ax.set_title("Comparação de Métricas entre os Modelos", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("graficos/01_comparacao_metricas.png", dpi=150)
plt.close()
print("   ✓ graficos/01_comparacao_metricas.png")

# --- Gráfico 2: Matrizes de Confusão ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = ["Não Churn", "Churn"]

for ax, y_pred, title in zip(
    axes,
    [y_pred_dt, y_pred_rf],
    ["Árvore de Decisão (Parte 1)", "Random Forest (Parte 2)"]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, cbar=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predição", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)

plt.suptitle("Matrizes de Confusão", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("graficos/02_matrizes_confusao.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ graficos/02_matrizes_confusao.png")

# --- Gráfico 3: Importância de Features ---
feat_imp  = pd.Series(rf.feature_importances_, index=X.columns)
top_feats = feat_imp.nlargest(12)

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = ["#1D9E75" if i >= 9 else "#9FE1CB" for i in range(len(top_feats))]
top_feats.sort_values().plot(kind="barh", ax=ax, color=colors_bar)
ax.set_xlabel("Importância", fontsize=11)
ax.set_title("Top 12 Atributos Mais Importantes — Random Forest", fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("graficos/03_importancia_features.png", dpi=150)
plt.close()
print("   ✓ graficos/03_importancia_features.png")

# --- Gráfico 4: Distribuição + F1 ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

orig_churn = pd.read_csv("telco_churn.csv")["Churn"].value_counts()
axes[0].pie(orig_churn, labels=["Não Churn", "Churn"],
            autopct="%1.1f%%", colors=["#9FE1CB", "#7F77DD"],
            startangle=90, wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[0].set_title("Distribuição de Churn no Dataset", fontsize=11, fontweight="bold")

models   = ["Árvore de Decisão\n(Parte 1)", "Random Forest\n(Parte 2)"]
f1_vals  = [f1_dt, f1_rf]
bar_clrs = ["#7F77DD", "#1D9E75"]
bars = axes[1].bar(models, f1_vals, color=bar_clrs, alpha=0.90, width=0.45)
for bar, val in zip(bars, f1_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 0.85)
axes[1].set_ylabel("F1-Score", fontsize=11)
axes[1].set_title("F1-Score por Modelo", fontsize=11, fontweight="bold")
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].yaxis.grid(True, alpha=0.3)
axes[1].set_axisbelow(True)

plt.tight_layout()
plt.savefig("graficos/04_distribuicao_e_f1.png", dpi=150)
plt.close()
print("   ✓ graficos/04_distribuicao_e_f1.png")

# ---- 5. RELATÓRIO FINAL 
print("\n" + "=" * 60)
print("5. RELATÓRIO FINAL DE COMPARAÇÃO")
print("=" * 60)
print(f"{'Métrica':<18} {'Árvore Decisão':>16} {'Random Forest':>14}")
print("-" * 50)
for m, v1, v2 in zip(metricas, vals_dt, vals_rf):
    print(f"{m:<18} {v1:>16.4f} {v2:>14.4f}")
print("-" * 50)
vencedor = "Random Forest" if f1_rf > f1_dt else "Árvore de Decisão"
print(f"\n  Melhor modelo pelo F1-Score: {vencedor}")
print(f"  Todos os gráficos salvos em: graficos/")
print(f"  Modelos salvos em: modelos/")
print("=" * 60)

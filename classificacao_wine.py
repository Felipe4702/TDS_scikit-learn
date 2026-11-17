"""Funcoes de classificacao do dataset Wine com pipelines scikit-learn.

Este arquivo concentra toda a implementacao. Pode ser executado sozinho
ou importado pelo notebook classificacao_wine.ipynb.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def carregar_dados():
    """Retorna os atributos, o alvo e os nomes das classes do Wine."""
    wine = load_wine(as_frame=True)
    return wine.data, wine.target, wine.target_names


def dividir_dados(X, y):
    """Separa 75% para treino e 25% para teste, preservando as classes."""
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def criar_pipelines():
    """Cria os dois pipelines solicitados no enunciado."""
    return {
        "Regressao Logistica": Pipeline([
            ("padronizacao", StandardScaler()),
            ("classificador", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "KNN (k=5)": Pipeline([
            ("padronizacao", StandardScaler()),
            ("classificador", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def realizar_validacao_cruzada(modelos, X_treino, y_treino):
    """Avalia cada pipeline em 5 folds estratificados, so no treino."""
    resultados = {
        nome: cross_val_score(modelo, X_treino, y_treino, cv=5, scoring="accuracy")
        for nome, modelo in modelos.items()
    }
    resumo = pd.DataFrame({
        nome: {
            "Media": notas.mean(),
            "Desvio padrao": notas.std(),
            "Minimo": notas.min(),
            "Maximo": notas.max(),
        }
        for nome, notas in resultados.items()
    }).T
    return resultados, resumo


def avaliar_modelos(modelos, X_treino, X_teste, y_treino, y_teste, nomes_classes):
    """Treina no conjunto completo de treino e avalia somente no teste."""
    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        resultados[nome] = {
            "modelo": modelo,
            "previsoes": previsoes,
            "acuracia": accuracy_score(y_teste, previsoes),
            "relatorio": classification_report(
                y_teste,
                previsoes,
                labels=np.arange(len(nomes_classes)),
                target_names=nomes_classes,
                zero_division=0,
            ),
        }
    return resultados


def selecionar_melhor_modelo(resultados_cv):
    """Seleciona pela media da validacao cruzada, sem consultar o teste."""
    return max(resultados_cv, key=lambda nome: resultados_cv[nome].mean())


def plotar_matriz_confusao(modelo, X_teste, y_teste, nomes_classes, nome_modelo):
    """Desenha a matriz de confusao do modelo selecionado."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_estimator(
        modelo, X_teste, y_teste,
        display_labels=nomes_classes,
        cmap="Blues", values_format="d", ax=ax,
    )
    ax.set_title(f"Matriz de confusao - {nome_modelo}")
    fig.tight_layout()
    return fig, ax


def main():
    """Executa todas as etapas tambem fora do notebook."""
    X, y, nomes_classes = carregar_dados()
    print(f"Amostras: {len(X)} | Atributos: {X.shape[1]} | Classes: {y.nunique()}")
    print("Distribuicao das classes:\n", y.value_counts().sort_index().to_string())
    print("\nPrimeiras linhas:\n", X.head().to_string())

    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)
    print(f"\nTreino: {len(X_treino)} | Teste: {len(X_teste)}")

    modelos = criar_pipelines()
    resultados_cv, resumo_cv = realizar_validacao_cruzada(modelos, X_treino, y_treino)
    print("\nAcuracia por fold:")
    for nome, notas in resultados_cv.items():
        print(f"{nome}: {np.round(notas, 4)}")
    print("\nResumo da validacao cruzada:\n", resumo_cv.round(4).to_string())

    melhor_nome = selecionar_melhor_modelo(resultados_cv)
    resultados_teste = avaliar_modelos(
        modelos, X_treino, X_teste, y_treino, y_teste, nomes_classes
    )
    for nome, resultado in resultados_teste.items():
        print(f"\n===== {nome} =====")
        print(f"Acuracia no teste: {resultado['acuracia']:.4f}")
        print(resultado["relatorio"])

    print(f"Melhor pipeline pela validacao cruzada: {melhor_nome}")
    plotar_matriz_confusao(
        resultados_teste[melhor_nome]["modelo"],
        X_teste, y_teste, nomes_classes, melhor_nome,
    )
    plt.savefig("matriz_confusao.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Matriz de confusao salva em matriz_confusao.png")


if __name__ == "__main__":
    main()

# TDS_scikit-learn

Atividade prática de **Classificação e Pipelines com scikit-learn** — Ferramentas para Data Science (UFSJ).

## Arquivos

- `classificacao_wine.py`: **código-fonte principal**, com funções reutilizáveis e `main()` para execução pelo terminal.
- `classificacao_wine.ipynb`: importa as funções do `.py`, executa cada etapa e apresenta resultados, matriz de confusão e respostas teóricas em Markdown.
- `environment.yml`: dependências Conda compatíveis com Linux e Windows.
- `.gitignore`: arquivos locais que não devem ser versionados.

Mantenha o `.py` e o `.ipynb` **na mesma pasta**. O notebook depende do `.py` para funcionar.

## Criar o ambiente Conda (Linux ou Windows)

Na pasta do projeto:

```bash
conda env create -f environment.yml
conda activate tds-scikit-learn
```

Se o ambiente já existir e precisar atualizar dependências:

```bash
conda env update -f environment.yml --prune
```

### Linux

Se `conda activate` não funcionar no Bash, execute `conda init bash` e reabra o terminal (ou rode `source ~/.bashrc`).

### Windows

Use Anaconda Prompt, Miniconda Prompt ou o terminal do VS Code. No PowerShell, se necessário, execute `conda init powershell` e reabra o terminal.

**Não copie o ambiente instalado de um sistema para outro:** recrie-o com `environment.yml`.

## Executar o código Python

```bash
python classificacao_wine.py
```

Exibe as métricas no terminal e abre o gráfico da matriz de confusão.

## Executar o notebook

```bash
jupyter lab
```

Abra `classificacao_wine.ipynb` e execute as células em ordem. No VS Code, abra o notebook e selecione o interpretador do ambiente `tds-scikit-learn`.

Se o kernel não aparecer, execute:

```bash
python -m ipykernel install --user --name tds-scikit-learn --display-name "Python (tds-scikit-learn)"
```

## Método

1. Dataset `load_wine(as_frame=True)`; 75% treino e 25% teste (`random_state=42`, estratificado).
2. Pipelines com `StandardScaler` e Regressão Logística (`max_iter=1000`, `random_state=42`) ou KNN (`n_neighbors=5`).
3. Validação cruzada estratificada (`cross_val_score`, `cv=5`) **apenas no treino**.
4. Seleção do melhor pipeline pela média da validação cruzada, seguida de treinamento no treino completo, relatório de classificação e matriz de confusão no teste.

O `Pipeline` aprende a padronização dentro de cada partição de treino, evitando vazamento de informações da validação.

## GitHub e entrega

Repositório: https://github.com/Felipe4702/TDS_scikit-learn

```bash
git add .
git commit -m "Organiza classificacao Wine em modulo Python e notebook"
git push
```

**Entrega pelo SIGAA até 25/11/2026:** o arquivo `.ipynb` com os resultados salvos, conforme o enunciado. Para que outra pessoa consiga executar o notebook, forneça também o `classificacao_wine.py` e o `environment.yml` pelo repositório. Antes de enviar, reinicie o kernel, execute todas as células e salve o notebook.

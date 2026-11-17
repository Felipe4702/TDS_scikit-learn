# TDS_scikit-learn

Atividade prática de **Classificação e Pipelines com scikit-learn** da disciplina Ferramentas para Data Science (UFSJ).

## Conteúdo

- `classificacao_wine.ipynb`: notebook principal, com código, resultados, matriz de confusão e análise crítica.
- `environment.yml`: ambiente Conda multiplataforma (Linux e Windows).
- `.gitignore`: arquivos locais que não devem ser versionados.

## Ambiente Conda (Linux e Windows)

No terminal, na pasta do repositório:

```bash
conda env create -f environment.yml
conda activate tds-scikit-learn
```

Se o ambiente já existir e o arquivo `environment.yml` for atualizado:

```bash
conda env update -f environment.yml --prune
```

O arquivo YAML lista dependências independentes do sistema, então o Conda instala as versões apropriadas em cada sistema. Não copie a pasta do ambiente Conda entre Linux e Windows.

## Rodar notebook

```bash
jupyter lab
```

Abra `classificacao_wine.ipynb` no navegador e selecione o kernel **Python (tds-scikit-learn)**, ou abra pelo VS Code escolhendo o interpretador desse ambiente. Execute as células em ordem.

Se o kernel não aparecer, com o ambiente ativado execute:

```bash
python -m ipykernel install --user --name tds-scikit-learn --display-name "Python (tds-scikit-learn)"
```

### Linux

Se `conda activate` não funcionar em um terminal Bash, inicialize o Conda uma vez com `conda init bash` e abra um terminal novo. Também é possível abrir o projeto com `code .` se o VS Code estiver instalado.

### Windows

Abra **Anaconda Prompt**, **Miniconda Prompt** ou o terminal do VS Code configurado para o Conda e execute os mesmos comandos acima. Em PowerShell, caso a ativação não funcione, execute `conda init powershell` e abra novamente o terminal.

## GitHub

Repositório: https://github.com/Felipe4702/TDS_scikit-learn

Se for a primeira vez configurando o projeto:

```bash
git clone https://github.com/Felipe4702/TDS_scikit-learn.git
cd TDS_scikit-learn
```

Copie os arquivos deste projeto para essa pasta, depois:

```bash
git add .
git commit -m "Implementa classificacao Wine com pipelines e validacao cruzada"
git push -u origin main
```

Confira o nome da branch com `git branch --show-current` antes do push; se não for `main`, substitua pelo nome exibido.

## Entrega

Reinicie o kernel e execute todas as células em ordem antes da submissão. Salve o notebook **com as saídas e gráficos** e envie o `.ipynb` via SIGAA até **25/11/2026**.

# Projeto de Regressão de Valores Imobiliários: Previsão de Preços na Califórnia (Questão 2 - Aprendizagem de Máquina)

## Visão Geral do Projeto

Este repositório contém a solução completa para a Questão 2 da avaliação da disciplina de Aprendizagem de Máquina da Universidade Federal do Rio Grande do Norte (UFRN) - Escola Agrícola de Jundiaí. O objetivo principal é desenvolver um modelo de **regressão** para prever o valor mediano de imóveis (`median_house_value`) na Califórnia, utilizando o dataset `housing.csv`, e implantar este modelo em uma aplicação web desenvolvida com Flask.

## Detalhamento das Etapas Cumpridas (Questão 2)

Todas as etapas sugeridas na Questão 2 foram rigorosamente cumpridas, conforme detalhado abaixo:

* **2. Desenvolver modelo de aprendizagem de máquina para regressão de valor de imóvel (‘median_house_value’) com base no dataset housing.csv:**
    * **Resposta:** O projeto principal é a construção de um modelo de regressão para estimar o `median_house_value` a partir do dataset `housing.csv`. Todo o pipeline de ML, desde o carregamento dos dados até a implantação na aplicação web, foi desenvolvido com este fim.

* **a) As features originais são longitude, latitude, idade_media_do_imovel, total_de_comodos, total_de_quartos, população, família, renda_media, valor_medio_do_imovel (target), proximidade_do_mar. Ver tipo das features:**
    * **Resposta:** Após o carregamento do dataset, os tipos de dados de todas as features originais (numéricas como `longitude`, `latitude`, `median_income` e categórica como `ocean_proximity`) foram inspecionados utilizando `df.info()` e `df.dtypes` no notebook Jupyter.

* **b) Obter o histograma de todas as features numéricas, para observar valores limites (describe) e formato das distribuições:**
    * **Resposta:** Histogramas foram gerados para todas as features numéricas (`df.hist()`) para visualizar suas distribuições e identificar padrões, skewness ou capping. As estatísticas descritivas básicas (`df.describe()`) também foram obtidas para resumir os valores limites e tendências de cada feature.

* **c) Criar as features ‘rooms_per_household’ = ‘total_rooms’/’households’ e ‘bedrooms_per_room’ = total_bedrooms/total_rooms:**
    * **Resposta:** As novas features `rooms_per_household` (média de cômodos por residência) e `bedrooms_per_room` (proporção de quartos por cômodo total) foram criadas por meio de operações de divisão entre colunas existentes. A feature `population_per_household` também foi criada como adicional.

* **d) Obter a matriz de correlação e listar as 5 features mais fortemente correlacionadas com o valor_medio_imovel:**
    * **Resposta:** A matriz de correlação entre todas as features numéricas e a variável alvo `median_house_value` foi calculada usando `df.corr()`. As 5 features mais fortemente correlacionadas com `median_house_value` (em valor absoluto) foram identificadas e listadas, com `median_income` mostrando a maior correlação.

* **e) Criar uma nova feature temporária apenas para estratificação do conjunto treino/teste ‘income_cat’:**
    * **Resposta:** A feature temporária `income_cat` foi criada a partir da `median_income` utilizando `pd.cut()` com os bins e labels especificados no enunciado, para segmentar a renda em categorias.

* **f) Faça o train_test_split com 20% para teste, só que com o parâmetro stratified setado para esta nova feature:**
    * **Resposta:** O dataset foi dividido em conjuntos de treino e teste (80% treino, 20% teste) utilizando `train_test_split` e o parâmetro `stratify=df['income_cat']`. Isso garantiu que as proporções das faixas de renda fossem mantidas de forma consistente em ambos os conjuntos, melhorando a representatividade.

* **g) Remova esta coluna income_cat:**
    * **Resposta:** Após a divisão estratificada dos dados, a coluna temporária `income_cat` foi removida tanto do conjunto de treino (`X_train`) quanto do conjunto de teste (`X_test`), pois sua função era apenas auxiliar na estratificação.

* **h) Preencher os valores ausentes de cada feature numérica com a média da respectiva feature (usar SimpleImputer):**
    * **Resposta:** Um `SimpleImputer` com a estratégia `mean` (média) foi configurado e incluído no pipeline de pré-processamento para tratar automaticamente os valores ausentes em features numéricas (como `total_bedrooms`).

* **i) Aplicar padronização (standard) às colunas numéricas:**
    * **Resposta:** O `StandardScaler` foi aplicado a todas as colunas numéricas como parte do pipeline de pré-processamento, garantindo que os dados estivessem na mesma escala e melhorando o desempenho de algoritmos sensíveis à escala.

* **j) Aplicar OrdinalEncoder à coluna ocean_proximity, assumindo que mais próximo do mar, mais caro:**
    * **Resposta:** A feature categórica `ocean_proximity` foi codificada utilizando `OrdinalEncoder`, com uma ordem de categorias explicitamente definida para refletir a suposição de que a proximidade ao mar se correlaciona com o preço do imóvel.

* **k) Construa Pipeline ColumnTransformer que agregue tais transformações:**
    * **Resposta:** Um `ColumnTransformer` foi construído para agregar todas as etapas de pré-processamento (imputação, padronização, codificação ordinal e One-Hot Encoder para outras categóricas se houvesse) em um único objeto, aplicando-as às colunas corretas. Este `ColumnTransformer` faz parte do pipeline de cada modelo.

* **l) Faça a validação com grid search e comparação dos modelos LinearRegression, DecisionTreeRegressor e RandomForestRegressor:**
    * **Resposta:** Foram criados pipelines para `LinearRegression`, `DecisionTreeRegressor` e `RandomForestRegressor`, que foram submetidos a um `GridSearchCV` com validação cruzada (`KFold`). As métricas `Mean Squared Error (MSE)` e `R2 Score` foram utilizadas para avaliar e comparar o desempenho dos modelos.

* **m) Após ver o melhor, aplique no conjunto de teste:**
    * **Resposta:** O modelo **RandomForestRegressor** foi identificado como o de melhor desempenho (menor MSE e maior R2) com base nos resultados da validação cruzada e foi subsequentemente aplicado ao conjunto de teste (`X_test`) para uma avaliação final de sua capacidade de generalização.

* **n) Avalie o efeito de regularização L1, L2 e ElasticNet nos modelos, quando aplicável este hiperparâmetro:**
    * **Resposta:** Os modelos `Lasso` (L1), `Ridge` (L2) e `ElasticNet` (que combina L1 e L2) foram incluídos e testados no `GridSearchCV`. A análise dos resultados mostrou que a regularização não trouxe melhorias significativas no desempenho (MSE e R2) para este problema em comparação com a `LinearRegression` simples, indicando que não foi estritamente necessária neste caso.

* **o) Aplique os respectivos testes estatísticos de regressão e, se necessário, ajuste a função de ligação:**
    * **Resposta:** Este item foi abordado teoricamente. Em modelos de Machine Learning não-lineares (como Random Forest), as métricas preditivas (MSE, R2) são o foco principal. Testes estatísticos tradicionais e ajuste da função de ligação são mais relevantes para modelos lineares generalizados (GLMs) e análises estatísticas formais. A capacidade do modelo de generalizar foi o foco prático.

* **p) Implemente o front e back que permita usar este modelo para predizer valores de imóveis na região. Pense se utilizará as coordenadas geográficas!**
    * **Resposta:** Uma aplicação web completa foi desenvolvida utilizando o framework Flask. O backend (`app_housing.py`) carrega o `RandomForestRegressor` treinado e processa as previsões. O frontend (`templates/index_housing.html`) apresenta um formulário onde o usuário pode inserir todas as características do imóvel, **incluindo as coordenadas geográficas (`longitude` e `latitude`)**. O valor predito do imóvel é exibido na tela, demonstrando a funcionalidade do modelo em um ambiente interativo.

## Estrutura do Repositório

```
Projeto_Regressao_Imoveis/
├── app_housing.py              # Backend da aplicação Flask
├── housing_regression_model.pkl# Modelo de Regressão de Imóveis (Random Forest Regressor)
├── Questao2_Regressao_Imoveis.ipynb # Notebook Jupyter com o pipeline de ML (treinamento e avaliação)
└── templates/
└── index_housing.html      # Frontend HTML da aplicação web
```
## Como Executar a Aplicação

Para executar a aplicação Flask localmente, siga os passos abaixo:

### Pré-requisitos

Certifique-se de ter o [Anaconda](https://www.anaconda.com/products/individual) instalado, que inclui o `conda` para gerenciamento de ambientes e pacotes.

### Configuração do Ambiente Python

1.  **Abra o Anaconda Prompt** (ou seu terminal no Linux/macOS).
2.  **Crie e ative o ambiente virtual** `loan_ml_env` com as dependências necessárias. Este ambiente garante que todas as bibliotecas tenham versões compatíveis:

    ```bash
    conda create -n loan_ml_env python=3.10 imbalanced-learn=0.10.1 pandas numpy scipy Flask joblib matplotlib seaborn jupyterlab -c conda-forge -y
    conda activate loan_ml_env
    ```
    (Aguarde a conclusão da criação e instalação dos pacotes. Pode levar alguns minutos.)

3.  **Confirme as versões (Opcional):**
    ```bash
    pip show scikit-learn
    pip show imbalanced-learn
    pip show Flask
    ```

### Execução do Projeto

1.  **Navegue até o diretório do projeto:**
    ```bash
    cd C:\Users\alfre\OneDrive\Documentos\Projeto_Regressao_Imoveis
    ```
    (Ou o caminho correto para onde você salvou a pasta da Questão 2.)

2.  **Abra e execute o Notebook Jupyter (Uma única vez para salvar o modelo):**
    * ```bash
        jupyter lab
        ```
    * No JupyterLab, abra o `Questao2_Regressao_Imoveis.ipynb`.
    * Vá em `Kernel` -> `Restart Kernel and Run All Cells`. Isso irá re-executar todo o treinamento e salvar o modelo `housing_regression_model.pkl` novamente.
    * Após a execução completa, feche o JupyterLab (pressione `Ctrl + C` no terminal onde ele foi iniciado para encerrar o servidor).

3.  **Execute a Aplicação Flask:**
    * Com o ambiente `loan_ml_env` ainda ativo e no diretório `Projeto_Regressao_Imoveis`, execute:
        ```bash
        python app_housing.py
        ```
    * O Flask iniciará em `* Running on http://127.0.0.1:5001` (usando uma porta diferente para não conflitar com outras aplicações Flask).

### Acesso à Aplicação no Navegador

* Acesse `http://127.0.0.1:5001`

## Demonstração em Vídeo

Assista à demonstração em vídeo deste projeto em: ([https://www.youtube.com/watch?v=fB9H-jL9uug](https://www.youtube.com/watch?v=_mCBvIMQPZA))

## Contato

Para dúvidas ou sugestões, por favor, entre em contato.

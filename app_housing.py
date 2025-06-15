import joblib
from flask import Flask, request, render_template
import pandas as pd
import numpy as np # Importado para np.nan e cálculos

app = Flask(__name__)

# Carregar o modelo treinado
try:
    model = joblib.load('housing_regression_model.pkl')
    print("Modelo 'housing_regression_model.pkl' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None # Define model como None para evitar erros se o carregamento falhar

@app.route('/')
def home_housing():
    # Renderiza o template HTML do formulário para a Questão 2
    return render_template('index_housing.html')

@app.route('/predict_housing', methods=['POST'])
def predict_housing():
    if model is None:
        return render_template('index_housing.html', prediction_text="Erro: Modelo não carregado. Contate o administrador.")

    # Coletar os dados do formulário
    longitude = float(request.form['longitude'])
    latitude = float(request.form['latitude'])
    housing_median_age = float(request.form['housing_median_age'])
    total_rooms = float(request.form['total_rooms'])
    total_bedrooms = float(request.form['total_bedrooms'])
    population = float(request.form['population'])
    households = float(request.form['households'])
    median_income = float(request.form['median_income'])
    ocean_proximity = request.form['ocean_proximity']

    # Criar um DataFrame com os dados de entrada, na ordem correta das colunas originais
    # As colunas originais no DataFrame carregado eram:
    # 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    # 'population', 'households', 'median_income', 'ocean_proximity'

    # As features criadas no notebook também precisam ser calculadas aqui:
    # 'rooms_per_household' = 'total_rooms' / 'households'
    # 'bedrooms_per_room' = 'total_bedrooms' / 'total_rooms'
    # 'population_per_household' = 'population' / 'households'

    # Preparar o DataFrame de entrada com todas as features esperadas pelo modelo,
    # incluindo as que foram criadas. A ordem importa para o ColumnTransformer!
    # A ordem das colunas no X_train original (antes do drop de income_cat) era:
    # 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    # 'population', 'households', 'median_income', 'ocean_proximity',
    # 'rooms_per_household', 'bedrooms_per_room', 'population_per_household'

    # Vamos construir um DataFrame com todas as colunas que o modelo espera no X_train
    # antes do ColumnTransformer, na ordem correta.
    # É crucial que essa ordem e nomes correspondam à ordem das colunas no X_train original
    # que alimentou o modelo treinado.

    # Lista das colunas na ordem original de X_train após a remoção de 'income_cat'
    # (Verificar a saída de X_train.columns.tolist() no seu notebook após remover 'income_cat')
    # Foi: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    #        'population', 'households', 'median_income', 'ocean_proximity',
    #        'rooms_per_household', 'bedrooms_per_room', 'population_per_household']

    input_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity, # Vai como object para o OrdinalEncoder
        'rooms_per_household': total_rooms / households,
        'bedrooms_per_room': total_bedrooms / total_rooms,
        'population_per_household': population / households
    }

    # Criar o DataFrame de entrada. É importante manter a ordem das colunas
    # que o modelo foi treinado para esperar.
    # A ordem obtida de X_train.columns.tolist() é a mais segura.
    # (Copie e cole a lista de colunas exata da saída do seu notebook para garantir)
    columns_order_for_prediction = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                                    'total_bedrooms', 'population', 'households', 'median_income',
                                    'ocean_proximity', 'rooms_per_household', 'bedrooms_per_room',
                                    'population_per_household']

    input_df = pd.DataFrame([input_data], columns=columns_order_for_prediction)

    # Realizar a predição
    predicted_value = model.predict(input_df)[0]

    # Formatar o resultado para exibição
    result_text = f"O valor médio predito do imóvel é: ${predicted_value:,.2f}"

    return render_template('index_housing.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True, port=5001) # Usar uma porta diferente (ex: 5001) para não conflitar com Questão 1
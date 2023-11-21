import pandas as pd
base = pd.read_excel('DESPESAS_MEDICAS_TREINO.xls')

estatistica = base.describe()

# Separação das variáveis previsoras e f(x)
previsores = base.iloc[:, 0:6].values
f = base.iloc[:, 6].values

# Pré processamento categórico
from sklearn.preprocessing import LabelEncoder

# Inicialize o LabelEncoder
label_encoder = LabelEncoder()

# Mapeie os dados categóricos para números
for col in range(previsores.shape[1]):
    previsores[:, col] = label_encoder.fit_transform(previsores[:, col])

# NORMALIZAÇÃO DA BASE DE DADOS 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# DIVISAO DA BASE EM CONJUNTOS DE TREINAMENTO E TESTE
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, f_treinamento, f_teste = train_test_split(previsores, f, test_size=0.25, random_state=0)

from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs', max_iter=10000,tol=0.001)

treino = nn.fit(previsores_treinamento, f_treinamento)

teste = nn.predict(previsores_teste)

from sklearn import metrics
mae = metrics.mean_absolute_error(f_teste,teste)
mse = metrics.mean_squared_error(f_teste,teste)
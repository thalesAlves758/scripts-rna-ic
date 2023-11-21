import pandas as pd
base = pd.read_csv('Airlines - TREINO.csv')

# Separação das variáveis previsoras e f(x)
previsores = base.iloc[:, 1:8].values
f = base.iloc[:, 8].values

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
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, f, test_size=0.20, random_state=0)

from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(8,),  activation='tanh', solver='lbfgs', max_iter=1000,tol=0.001)

treino = nn.fit(previsores_treinamento, classe_treinamento)

teste = nn.predict(previsores_teste)

#MATRIZ DE CONFUSÃO
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(classe_teste, teste)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(classe_teste,teste)

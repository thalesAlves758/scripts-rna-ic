import pandas as pd

def executeRegressorAlgorithm(trainTestInputs, neuronsList, activations, solvers):
    from sklearn.neural_network import MLPRegressor
    from sklearn import metrics

    previsores_treinamento, previsores_teste, f_treinamento, f_teste = trainTestInputs

    data = []

    for neurons in neuronsList:
        neuronExecutionData = []

        for solver in solvers:
            row = []

            for activation in activations:
                try:
                    nn = MLPRegressor(solver=solver, activation=activation, hidden_layer_sizes=(neurons), max_iter=10000,tol=0.001)

                    nn.fit(previsores_treinamento, f_treinamento)
                    teste = nn.predict(previsores_teste)

                    row.append(metrics.mean_absolute_error(f_teste,teste))
                except:
                    row.append('error')

            neuronExecutionData.append(row)

        data.append(neuronExecutionData)

    return data

def writeExcelFile(filepath, data, columns, index, sheetsNames):
    import os
    mode = 'a' if os.path.exists(filepath) else 'w'

    for i in range(len(data)):
        df = pd.DataFrame(data[i], columns=columns, index=index)

        if mode == 'a':
          with pd.ExcelWriter(filepath, mode=mode, engine="openpyxl", if_sheet_exists='replace') as writer:
              df.to_excel(writer, sheet_name=sheetsNames[i])
        else:
          with pd.ExcelWriter(filepath, mode=mode) as writer:
              df.to_excel(writer, sheet_name=sheetsNames[i])

# SCRIPT PRINCIPAL
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

# previsores = [
#     [row[0], label_encoder.fit_transform([row[1]])[0], row[2], row[3],
#      label_encoder.fit_transform([row[4]])[0], label_encoder.fit_transform([row[5]])[0]]
#     for row in previsores
# ]

# NORMALIZAÇÃO DA BASE DE DADOS
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# DIVISAO DA BASE EM CONJUNTOS DE TREINAMENTO E TESTE
from sklearn.model_selection import train_test_split
trainTestInputs = train_test_split(previsores, f, test_size=0.25, random_state=0)

solvers = ['lbfgs', 'sgd', 'adam']
activations = ['identity', 'logistic', 'tanh', 'relu']
#neurons = [5, 8, 10, 15, 20]
neurons = [6, 7]
excelName = 'resultado regressao.xlsx'

data = executeRegressorAlgorithm(trainTestInputs, neurons, activations, solvers)

# CRIAR EXCEL COM O RESULTADO
sheetsNames = [str(neuronSize) + ' neurons' for neuronSize in neurons]
writeExcelFile(excelName, data, activations, solvers, sheetsNames)

# # MOSTRAR RESULTADO NO CONSOLE
# from tabulate import tabulate
# for i in range(len(data)):
#    print(str(neurons[i]) + ' neuronios')
#    print(tabulate(data[i], headers=activations, showindex=solvers, tablefmt="fancy_grid"))
#    print('\n')

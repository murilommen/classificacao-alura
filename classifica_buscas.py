#from collections import Counter
import pandas as pd

df = pd.read_csv('busca.csv')

X_df = df[['home', 'busca', 'logado']]
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15,random_state=None)

def fit_and_predict(nome, modelo, X_train, Y_train, X_test, Y_test):
	modelo.fit(X_train, Y_train)

	resultado = modelo.predict(X_test)
	acertos = (resultado == Y_test)
	total_de_acertos = sum(acertos)
	total_de_elementos = len(X_test)

	taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

	msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)

	print(msg)

	return taxa_de_acerto

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
# resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, X_train, X_test, Y_train, Y_test)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, X_train, X_test, Y_train, Y_test)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

print(vencedor)

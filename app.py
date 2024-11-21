from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Treinamento do modelo de machine learning
def treinar_modelo():
    dados = {
        'forca_adversario': [2, 0, 1, 2, 1, 2, 1, 0, 2, 1, 2, 0, 1, 1, 0, 2, 1, 2, 1, 0, 2, 2, 1, 2, 1, 0],
        'treino_intensidade': [0, 1, 2, 1, 2, 2, 0, 2, 2, 2, 1, 0, 2, 2, 1, 2, 0, 1, 2, 2, 1, 2, 2, 1, 0, 1],
        'condicao_fisica': [1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 0, 2, 2],
        'jogadores_chave': [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        'motivacao': [2, 1, 0, 2, 2, 2, 1, 0, 1, 2, 2, 1, 0, 2, 1, 1, 2, 1, 0, 2, 2, 1, 2, 0, 1, 2],
        'vitoria': [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    }
    df = pd.DataFrame(dados)

    X = df[['forca_adversario', 'treino_intensidade', 'condicao_fisica', 'jogadores_chave', 'motivacao']]
    y = df['vitoria']

    modelo = DecisionTreeClassifier()
    modelo.fit(X, y)

    return modelo

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prever', methods=['POST'])
def prever():
    modalidade = request.form['modalidade']
    forca_adversario = int(request.form['forca_adversario'])
    treino_intensidade = int(request.form['treino_intensidade'])
    condicao_fisica = int(request.form['condicao_fisica'])
    jogadores_chave = int(request.form['jogadores_chave'])
    motivacao = int(request.form['motivacao'])

    dados_entrada = [[forca_adversario, treino_intensidade, condicao_fisica, jogadores_chave, motivacao]]
    modelo = treinar_modelo()

    previsao = modelo.predict(dados_entrada)[0]

    if previsao == 1:
        resultado = "vitoria"
        mensagem = f"Parabéns! Vocês têm grandes chances de vencer no {modalidade}. Continuem assim!"
    else:
        resultado = "derrota"
        mensagem = f"As chances de vencer no {modalidade} são baixas. Mas não desanimem, continuem se esforçando!"

    return jsonify({"resultado": resultado, "mensagem": mensagem})

if __name__ == '__main__':
    app.run(debug=True)

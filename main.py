import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Título do aplicativo
st.title("Sistema de Recomendação para Saúde Cardiovascular com Explicações")

# Carregar o arquivo CSV
dados = pd.read_csv("randomforest.csv")  # Certifique-se de ajustar o caminho correto do arquivo

# Preparar os dados
colunas_para_remover = ['cardio', 'prediction margin', 'predicted cardio']
X = dados.drop(columns=colunas_para_remover, errors='ignore')  # Remova essas colunas
y = dados['cardio']  # Variável alvo

# Treinar o modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=0)
modelo.fit(X, y)

# Inicializar o estado do ano de nascimento, se ainda não existir
if 'ano_nascimento' not in st.session_state:
    st.session_state['ano_nascimento'] = 1990

# Interface para o usuário (Profissional de Saúde)
st.header("Insira os dados do paciente:")

# Usar st.number_input com o valor inicial do session state
ano_nascimento = st.number_input(
    "Ano de Nascimento:",
    min_value=1900,
    max_value=2024,
    value=st.session_state['ano_nascimento']
)

# Calcular a idade em dias
ano_atual = 2024  # Use o ano atual
idade_em_dias = (ano_atual - ano_nascimento) * 365

# Outras entradas do usuário
genero = st.selectbox("Gênero:", [0, 1])
altura = st.number_input("Altura (em cm):", min_value=50, max_value=250, value=170)
peso = st.number_input("Peso (em kg):", min_value=10, max_value=300, value=70)
p_sistolica = st.number_input("Pressão Sistólica:", min_value=50, max_value=250, value=120)
p_diastolica = st.number_input("Pressão Diastólica:", min_value=30, max_value=150, value=80)
colesterol = st.selectbox("Colesterol (0: Normal, 1: Acima do Normal):", [0, 1])
glicemia = st.selectbox("Glicemia (0: Normal, 1: Alta):", [0, 1])
fumante = st.selectbox("Fumante (0: Não, 1: Sim):", [0, 1])
alcool = st.selectbox("Consome Álcool (0: Não, 1: Sim):", [0, 1])
atv_fisica = st.selectbox("Atividade Física (0: Não, 1: Sim):", [0, 1])

# Coletar os dados do usuário
dados_usuario = pd.DataFrame({
    'ano_binarized=1': [idade_em_dias],  # Usar idade em dias calculada
    'genero_binarized=1': [genero],
    'altura_binarized=1': [altura],
    'peso_binarized=1': [peso],
    'p_sistolica_binarized=1': [p_sistolica],
    'p_diastolica_binarized=1': [p_diastolica],
    'colesterol_binarized=1': [colesterol],
    'glicemia_binarized=1': [glicemia],
    'fumante_binarized=1': [fumante],
    'alcool_binarized=1': [alcool],
    'atv_fisica_binarized=1': [atv_fisica]
})

# Garantir que as colunas de dados_usuario correspondam às de X
dados_usuario = dados_usuario.reindex(columns=X.columns, fill_value=0)

# Fazer a previsão
if st.button("Recomendar"):
    previsao = modelo.predict(dados_usuario)
    probabilidade = modelo.predict_proba(dados_usuario)[0][1]  # Probabilidade da classe 1

    # Lógica de recomendação
    st.subheader("Recomendação:")
    if previsao[0] == 1:
        st.write("Consulta médica urgente recomendada.")
    else:
        st.write("Mudanças no estilo de vida e check-ups regulares recomendados.")

    # Explicar a previsão usando SHAP
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)

    # Mostrar os motivos da previsão
    st.subheader("Motivos da Recomendação:")
    # Use Matplotlib para plotar os valores SHAP
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(plt)

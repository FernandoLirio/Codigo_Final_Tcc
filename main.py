import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Título do aplicativo
st.title("Sistema de Recomendação para Saúde Cardiovascular")

# Carregar o arquivo CSV
dados = pd.read_csv("base-de-dados-sem-id.csv")  # Ajuste o caminho do arquivo conforme necessário

# Preparar os dados
colunas_para_remover = ['cardio']  # Ajuste as colunas conforme a base de dados
X = dados.drop(columns=colunas_para_remover, errors='ignore')  # Remova a coluna alvo
y = dados['cardio']  # Variável alvo

# Configurando o modelo Random Forest com os parâmetros usados no Weka
modelo = RandomForestClassifier(
    n_estimators=100,  # Número de árvores
    random_state=0,  # Para reprodutibilidade
    max_features=None,  # Você pode ajustar conforme a configuração do Weka
    min_samples_split=2,  # Divisão mínima por nó
    min_samples_leaf=1  # Folhas mínimas
)
modelo.fit(X, y)

# Interface para o usuário (Profissional de Saúde)
st.header("Preencha as informações do paciente:")

# Entrada de dados do usuário
ano_nascimento = st.number_input("Ano de Nascimento:", min_value=1900, max_value=2024, value=1990)
ano_atual = 2024  # Use o ano atual
idade_em_dias = (ano_atual - ano_nascimento) * 365

genero = st.selectbox("Gênero:", ["Feminino", "Masculino"])
altura = st.number_input("Altura (em cm):", min_value=50, max_value=250, value=170)
peso = st.number_input("Peso (em kg):", min_value=10, max_value=300, value=70)
p_sistolica = st.number_input("Pressão Sistólica:", min_value=50, max_value=250, value=120)
p_diastolica = st.number_input("Pressão Diastólica:", min_value=30, max_value=150, value=80)
colesterol = st.selectbox("Colesterol:", ["Normal", "Acima do Normal"])
glicemia = st.selectbox("Glicemia:", ["Normal", "Alta"])
fumante = st.selectbox("Fumante:", ["Não", "Sim"])
alcool = st.selectbox("Consome Álcool:", ["Não", "Sim"])
atv_fisica = st.selectbox("Pratica Atividade Física:", ["Não", "Sim"])

# Coletar os dados do usuário
dados_usuario = pd.DataFrame({
    'ano_binarized=1': [idade_em_dias],
    'genero_binarized=1': [1 if genero == "Masculino" else 0],
    'altura_binarized=1': [altura],
    'peso_binarized=1': [peso],
    'p_sistolica_binarized=1': [p_sistolica],
    'p_diastolica_binarized=1': [p_diastolica],
    'colesterol_binarized=1': [1 if colesterol == "Acima do Normal" else 0],
    'glicemia_binarized=1': [1 if glicemia == "Alta" else 0],
    'fumante_binarized=1': [1 if fumante == "Sim" else 0],
    'alcool_binarized=1': [1 if alcool == "Sim" else 0],
    'atv_fisica_binarized=1': [1 if atv_fisica == "Sim" else 0]
})

# Garantir que as colunas de dados_usuario correspondam às de X
dados_usuario = dados_usuario.reindex(columns=X.columns, fill_value=0)

# Fazer a previsão
if st.button("Ver Recomendação"):
    previsao = modelo.predict(dados_usuario)
    probabilidade = modelo.predict_proba(dados_usuario)[0][1]

    # Lógica de recomendação
    st.subheader("Recomendação:")
    if previsao[0] == 1:
        st.write("**Consulta médica urgente recomendada.**")
    else:
        st.write("**Mudanças no estilo de vida e check-ups regulares recomendados.**")

    # Explicar a previsão usando SHAP
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(dados_usuario)

    # Mostrar os motivos da previsão de forma simples
    st.subheader("Fatores que Influenciaram a Decisão:")
    fatores_importantes = sorted(
        zip(X.columns, shap_values[1][0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]  # Mostrar os 5 fatores mais importantes

    for feature, value in fatores_importantes:
        if value > 0:
            st.write(f"- {feature.replace('_binarized=1', '')}: Aumentou o risco.")
        else:
            st.write(f"- {feature.replace('_binarized=1', '')}: Reduziu o risco.")

    # Gráfico de importância dos fatores
    st.subheader("Gráfico de Importância dos Fatores:")
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(plt)

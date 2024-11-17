import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Título do aplicativo
st.title("Sistema de Recomendação para Saúde Cardiovascular")

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
st.header("Preencha as informações do paciente:")

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
genero = st.selectbox("Gênero:", ["Feminino", "Masculino"])
altura = st.number_input("Altura (em cm):", min_value=50, max_value=250, value=170)
peso = st.number_input("Peso (em kg):", min_value=10, max_value=300, value=70)
p_sistolica = st.number_input("Pressão Sistólica (número mais alto da pressão arterial):", min_value=50, max_value=250, value=120)
p_diastolica = st.number_input("Pressão Diastólica (número mais baixo da pressão arterial):", min_value=30, max_value=150, value=80)
colesterol = st.selectbox("Colesterol:", ["Normal", "Acima do Normal"])
glicemia = st.selectbox("Glicemia (Açúcar no sangue):", ["Normal", "Alta"])
fumante = st.selectbox("Fumante:", ["Não", "Sim"])
alcool = st.selectbox("Consome Álcool:", ["Não", "Sim"])
atv_fisica = st.selectbox("Pratica Atividade Física:", ["Não", "Sim"])

# Coletar os dados do usuário
dados_usuario = pd.DataFrame({
    'Idade em Dias': [idade_em_dias],  # Usar idade em dias calculada
    'Gênero': [1 if genero == "Masculino" else 0],
    'Altura': [altura],
    'Peso': [peso],
    'Pressão Sistólica': [p_sistolica],
    'Pressão Diastólica': [p_diastolica],
    'Colesterol Elevado': [1 if colesterol == "Acima do Normal" else 0],
    'Glicemia Alta': [1 if glicemia == "Alta" else 0],
    'Fumante': [1 if fumante == "Sim" else 0],
    'Consome Álcool': [1 if alcool == "Sim" else 0],
    'Atividade Física': [1 if atv_fisica == "Sim" else 0]
})

# Garantir que as colunas de dados_usuario correspondam às de X
dados_usuario = dados_usuario.reindex(columns=X.columns, fill_value=0)

# Fazer a previsão
if st.button("Ver Recomendação"):
    previsao = modelo.predict(dados_usuario)
    probabilidade = modelo.predict_proba(dados_usuario)[0][1]  # Probabilidade da classe 1

    # Lógica de recomendação
    st.subheader("Recomendação:")
    if previsao[0] == 1:
        st.write("**Consulta médica urgente recomendada.**")
        st.write("Consulte um médico o mais rápido possível para uma avaliação detalhada.")
    else:
        st.write("**Mudanças no estilo de vida e check-ups regulares recomendados.**")
        st.write("Mantenha hábitos saudáveis e consulte o médico regularmente.")

    # Explicar a previsão usando SHAP
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(dados_usuario)

    # Mostrar os motivos da previsão de forma simples
    st.subheader("Fatores que Influenciaram a Decisão:")
    st.write("Aqui estão os fatores que mais contribuíram para a recomendação:")

    # Explicação simplificada de cada fator
    fatores_importantes = sorted(
        zip(X.columns, shap_values[1][0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]  # Mostrar os 5 fatores mais importantes

    for feature, value in fatores_importantes:
        if value > 0:
            st.write(f"- **{feature}**: Este fator aumentou o risco de problemas cardíacos.")
        else:
            st.write(f"- **{feature}**: Este fator ajudou a reduzir o risco de problemas cardíacos.")

    # Gráfico de importância dos fatores
    st.subheader("Gráfico de Importância dos Fatores:")
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot(plt)

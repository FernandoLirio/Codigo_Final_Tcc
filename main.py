import pandas as pd

# Carregar o dataset com o delimitador correto
file_path = 'cardio_train.csv'
df = pd.read_csv(file_path, delimiter=';')

# Converter idade de dias para anos
df['age_years'] = (df['age'] / 365).astype(int)

# Filtrar outliers na altura e peso com base em intervalos razoáveis
df_filtered = df[(df['height'] >= 120) & (df['height'] <= 220) &
                 (df['weight'] >= 40) & (df['weight'] <= 150)]

# Filtrar outliers na pressão arterial com base em valores razoáveis
df_filtered = df_filtered[(df_filtered['ap_hi'] >= 80) & (df_filtered['ap_hi'] <= 200) &
                          (df_filtered['ap_lo'] >= 60) & (df_filtered['ap_lo'] <= 140)]

# Excluir a coluna original de idade (em dias)
df_filtered = df_filtered.drop(columns=['age'])

# Exibir as primeiras linhas do dataframe tratado
print(df_filtered.head())

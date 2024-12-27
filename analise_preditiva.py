import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Usar o DataFrame df que já foi carregado no arquivo principal
df = st.session_state.df  # Acessa o DataFrame do estado da sessão

# Verificar se df é None
if df is None:
    st.error("Não foi possível carregar os dados. Verifique o arquivo e tente novamente.")
else:
    # Renomear colunas para minúsculas
    df.columns = df.columns.str.lower()

    # Converter a coluna 'datamovimento' para datetime
    df['datamovimento'] = pd.to_datetime(df['datamovimento'], errors='coerce')

    # Remover linhas com datas inválidas
    df = df[df['datamovimento'].notnull()]

# Título da página
st.title("Análise Preditiva")

# Exibir as primeiras linhas do DataFrame
st.subheader("Visualização dos Dados")
st.dataframe(df.head())

# Análise Exploratória
st.subheader("Análise Exploratória")

# Exibir informações do DataFrame
st.write("Informações do DataFrame:")
st.write(df.info())

# Exibir estatísticas descritivas
st.write("Estatísticas Descritivas:")
st.write(df.describe())

# Gráfico de distribuição de uma coluna (exemplo: 'faturamento')
if 'faturamento' in df.columns:
    st.subheader("Distribuição do Faturamento")
    plt.figure(figsize=(12, 6))
    sns.histplot(df['faturamento'], bins=75, kde=True)
    st.pyplot(plt) 

# Faturamento total
faturamento_total = df['faturamento'].sum()
st.write(f"**Faturamento Total:** R$ {faturamento_total:,.2f}")

# Faturamento médio mensal
faturamento_mensal = df.groupby(df['datamovimento'].dt.to_period('M'))['faturamento'].sum()
faturamento_medio_mensal = faturamento_mensal.mean()
st.write(f"**Faturamento Médio Mensal:** R$ {faturamento_medio_mensal:,.2f}")

# Faturamento médio por cliente
faturamento_medio_por_cliente = faturamento_total / df['codigocliente'].nunique()
st.write(f"**Faturamento Médio por Cliente:** R$ {faturamento_medio_por_cliente:,.2f}")

# Faturamento médio por nota fiscal
faturamento_medio_por_nota = faturamento_total / df['notasaida'].nunique()
st.write(f"**Faturamento Médio por Nota Fiscal:** R$ {faturamento_medio_por_nota:,.2f}")

# Faturamento médio por nome_cluster
faturamento_por_nome_cluster = df.groupby('nome_cluster')['faturamento'].sum()
faturamento_medio_por_nome_cluster = faturamento_por_nome_cluster.mean()
st.write(f"**Faturamento Médio por Cluster:** R$ {faturamento_medio_por_nome_cluster:,.2f}")

# Margem por nome_cluster
df.loc[:, 'margem'] = df['faturamento'] - df['custo']  # Calcular a margem usando .loc
margem_por_nome_cluster = df.groupby('nome_cluster')['margem'].mean()
st.write("**Margem por Cluster:**")
st.dataframe(margem_por_nome_cluster)


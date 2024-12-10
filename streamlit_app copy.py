import streamlit as st
import cx_Oracle
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Função para carregar dados (mover para o início)
@st.cache_data(ttl=timedelta(hours=12))
def carregar_dados():
    try:
        dsn = cx_Oracle.makedsn(os.getenv("DB_HOST"), os.getenv("DB_PORT"), service_name=os.getenv("DB_SERVICE"))
        connection = cx_Oracle.connect(user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"), dsn=dsn)
        
        # Executar a consulta
        cursor = connection.cursor()
        cursor.execute(os.getenv("SQL_QUERY"))
        
        # Obter nomes das colunas
        colunas = [desc[0].upper() for desc in cursor.description]
        
        # Converter resultados para DataFrame
        resultados = cursor.fetchall()
        df = pd.DataFrame(resultados, columns=colunas)
        
        cursor.close()
        connection.close()
        
        return df
    
    except cx_Oracle.DatabaseError as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None
    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")
        return None

# Função para criar análise de coorte (mantém igual)
def criar_cohort_analysis(df, modo='Normal'):
    """
    Função para criar a análise de coorte com dois modos diferentes
    """
    try:
        # Criar cópia do DataFrame
        df_cohort = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_cohort['DATAMOVIMENTO'] = pd.to_datetime(df_cohort['DATAMOVIMENTO'])
        
        if modo == 'Normal':
            # Modo normal - primeira compra define a safra
            df_cohort['COHORT_MES'] = df_cohort.groupby('CODIGOCLIENTE')['DATAMOVIMENTO'].transform('min').dt.strftime('%Y-%m')
            
        else:
            # Modo ajustado - identifica quebras na sequência de compras
            # Ordenar dados por cliente e data
            df_cohort = df_cohort.sort_values(['CODIGOCLIENTE', 'DATAMOVIMENTO'])
            
            # Criar coluna de mês-ano para cada transação
            df_cohort['MES_ANO'] = df_cohort['DATAMOVIMENTO'].dt.strftime('%Y-%m')
            
            # Identificar meses consecutivos
            df_cohort['MES_ANTERIOR'] = df_cohort.groupby('CODIGOCLIENTE')['DATAMOVIMENTO'].shift()
            df_cohort['MESES_DIFF'] = (df_cohort['DATAMOVIMENTO'].dt.year * 12 + df_cohort['DATAMOVIMENTO'].dt.month) - \
                                    (df_cohort['MES_ANTERIOR'].dt.year * 12 + df_cohort['MES_ANTERIOR'].dt.month)
            
            # Identificar nova safra quando há quebra na sequência (diferença > 1 mês)
            df_cohort['NOVA_SAFRA'] = (df_cohort['MESES_DIFF'] > 1) | \
                                     (df_cohort['MES_ANTERIOR'].isna())
            
            # Criar identificador de grupo de safra
            df_cohort['GRUPO_SAFRA'] = df_cohort.groupby('CODIGOCLIENTE')['NOVA_SAFRA'].cumsum()
            
            # Definir mês de safra como primeiro mês de cada grupo
            df_cohort['COHORT_MES'] = df_cohort.groupby(['CODIGOCLIENTE', 'GRUPO_SAFRA'])['DATAMOVIMENTO'].transform('min').dt.strftime('%Y-%m')
        
        # Criar mês da transação
        df_cohort['MES_TRANSACAO'] = df_cohort['DATAMOVIMENTO'].dt.strftime('%Y-%m')
        
        # Converter para datetime para cálculo correto do período
        df_cohort['COHORT_MES'] = pd.to_datetime(df_cohort['COHORT_MES'])
        df_cohort['MES_TRANSACAO'] = pd.to_datetime(df_cohort['MES_TRANSACAO'])
        
        # Calcular o índice do período
        df_cohort['PERIODO_INDEX'] = ((df_cohort['MES_TRANSACAO'].dt.year - df_cohort['COHORT_MES'].dt.year) * 12 +
                                    (df_cohort['MES_TRANSACAO'].dt.month - df_cohort['COHORT_MES'].dt.month))
        
        # Criar matriz de coorte
        cohort_data = df_cohort.groupby(['COHORT_MES', 'PERIODO_INDEX'])['CODIGOCLIENTE'].nunique().reset_index()
        cohort_data['COHORT_MES'] = cohort_data['COHORT_MES'].dt.strftime('%Y-%m')
        
        # Criar matriz pivotada
        cohort_matrix = cohort_data.pivot(index='COHORT_MES',
                                        columns='PERIODO_INDEX',
                                        values='CODIGOCLIENTE')
        
        # Calcular taxas de retenção
        retention_matrix = cohort_matrix.divide(cohort_matrix[0], axis=0) * 100
        
        return retention_matrix
        
    except Exception as e:
        st.error(f"Erro ao criar análise de coorte: {e}")
        return None

# Função para plotar heatmap (mantém igual)
def plotar_heatmap_cohort(retention_matrix):
    """
    Função para criar o heatmap da análise de coorte em alta resolução
    """
    try:
        # Configurar o estilo do matplotlib
        plt.style.use('default')
        
        # Criar figura em alta resolução (FHD - 1920x1080)
        plt.figure(figsize=(16, 9), dpi=120)
        
        # Criar heatmap com cores invertidas
        ax = sns.heatmap(retention_matrix,
                        annot=True,
                        fmt='.1f',
                        cmap='RdYlGn',  # Invertido de RdYlGn_r para RdYlGn
                        vmin=0,
                        vmax=100,
                        annot_kws={'size': 8},
                        cbar_kws={'label': 'Taxa de Retenção (%)'})
        
        # Melhorar a formatação e estilo
        plt.title('Análise de Coorte - Taxa de Retenção (%)', 
                 pad=20, 
                 size=14, 
                 fontweight='bold')
        
        plt.xlabel('Número de Meses', size=12, labelpad=10)
        plt.ylabel('Mês de Aquisição', size=12, labelpad=10)
        
        # Rotacionar labels do eixo x para melhor legibilidade
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Ajustar layout para evitar cortes
        plt.tight_layout()
        
        # Adicionar grade para melhor visualização
        ax.grid(False)
        
        # Melhorar a aparência geral
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        return plt
        
    except Exception as e:
        st.error(f"Erro ao criar heatmap: {e}")
        st.write("Estilos disponíveis:", plt.style.available)  # Debug
        return None

# Função para criar tabela de cobertura (mantém igual)
def criar_tabela_cobertura(df):
    """
    Função para criar a tabela de cobertura de vendas
    """
    try:
        # Criar cópia do DataFrame
        df_cobertura = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_cobertura['DATAMOVIMENTO'] = pd.to_datetime(df_cobertura['DATAMOVIMENTO'])
        
        # Criar colunas de ano-mês para safra e movimento
        df_cobertura['SAFRA'] = df_cobertura.groupby('CODIGOCLIENTE')['DATAMOVIMENTO'].transform('min').dt.strftime('%Y-%m')
        df_cobertura['MOVIMENTO'] = df_cobertura['DATAMOVIMENTO'].dt.strftime('%Y-%m')
        
        # Agrupar e contar clientes únicos
        tabela_cobertura = df_cobertura.groupby(['SAFRA', 'MOVIMENTO'])['CODIGOCLIENTE'].nunique().reset_index()
        
        # Criar tabela pivotada
        tabela_final = tabela_cobertura.pivot(index='SAFRA', 
                                            columns='MOVIMENTO', 
                                            values='CODIGOCLIENTE')
        
        # Adicionar total por linha
        tabela_final['Total Geral'] = tabela_final.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_final.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Novos Clientes'])
        for coluna in tabela_final.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                # Pegar apenas os valores onde o mês da coluna coincide com o mês da safra
                valores_iniciais[coluna] = tabela_final[coluna][tabela_final.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_final, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).astype(int)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de cobertura: {e}")
        return None

def criar_tabela_faturamento(df):
    """
    Função para criar a tabela de faturamento de vendas
    """
    try:
        # Criar cópia do DataFrame
        df_faturamento = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_faturamento['DATAMOVIMENTO'] = pd.to_datetime(df_faturamento['DATAMOVIMENTO'])
        
        # Criar colunas de ano-mês para safra e movimento
        df_faturamento['SAFRA'] = df_faturamento.groupby('CODIGOCLIENTE')['DATAMOVIMENTO'].transform('min').dt.strftime('%Y-%m')
        df_faturamento['MOVIMENTO'] = df_faturamento['DATAMOVIMENTO'].dt.strftime('%Y-%m')
        
        # Agrupar e somar faturamento
        tabela_faturamento = df_faturamento.groupby(['SAFRA', 'MOVIMENTO'])['FATURAMENTO'].sum().reset_index()
        
        # Criar tabela pivotada
        tabela_final = tabela_faturamento.pivot(index='SAFRA', 
                                              columns='MOVIMENTO', 
                                              values='FATURAMENTO')
        
        # Adicionar total por linha
        tabela_final['Total Geral'] = tabela_final.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_final.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Faturamento Inicial'])
        for coluna in tabela_final.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                # Pegar apenas os valores onde o mês da coluna coincide com o mês da safra
                valores_iniciais[coluna] = tabela_final[coluna][tabela_final.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_final, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).round(2)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de faturamento: {e}")
        return None

def criar_tabela_margem(df):
    """
    Função para criar a tabela de margem de vendas (Faturamento - CustoTotal)
    """
    try:
        # Criar cópia do DataFrame
        df_margem = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_margem['DATAMOVIMENTO'] = pd.to_datetime(df_margem['DATAMOVIMENTO'])
        
        # Calcular o custo total e a margem
        df_margem['CUSTOTOTAL'] = df_margem['CUSTO'] * df_margem['QUANTIDADE']
        df_margem['MARGEM'] = df_margem['FATURAMENTO'] - df_margem['CUSTOTOTAL']
        
        # Criar colunas de ano-mês para safra e movimento
        df_margem['SAFRA'] = df_margem.groupby('CODIGOCLIENTE')['DATAMOVIMENTO'].transform('min').dt.strftime('%Y-%m')
        df_margem['MOVIMENTO'] = df_margem['DATAMOVIMENTO'].dt.strftime('%Y-%m')
        
        # Agrupar e somar margem
        tabela_margem = df_margem.groupby(['SAFRA', 'MOVIMENTO'])['MARGEM'].sum().reset_index()
        
        # Criar tabela pivotada
        tabela_final = tabela_margem.pivot(index='SAFRA', 
                                         columns='MOVIMENTO', 
                                         values='MARGEM')
        
        # Adicionar total por linha
        tabela_final['Total Geral'] = tabela_final.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_final.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Margem Inicial'])
        for coluna in tabela_final.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                # Pegar apenas os valores onde o mês da coluna coincide com o mês da safra
                valores_iniciais[coluna] = tabela_final[coluna][tabela_final.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_final, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).round(2)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de margem: {e}")
        return None

# Carregar dados uma única vez
df = carregar_dados()

# Remover filiais 1 e 5 do DataFrame
if df is not None:
    df = df[~df['CODIGOFILIAL'].astype(str).isin(['1', '5', '2'])]
    
    # Resetar o índice após a remoção
    df = df.reset_index(drop=True)

# Título da aplicação
st.title("Análise Coorte e Retenção")
st.subheader("Villa Camarão!")

# Sidebar com filtros
st.sidebar.header("Filtros")

# Inicializar filtros apenas se df não for None
if df is not None:
    # Converter DATAMOVIMENTO para datetime se ainda não estiver
    df['DATAMOVIMENTO'] = pd.to_datetime(df['DATAMOVIMENTO'])
    
    # Obter data mínima e máxima do DataFrame
    data_min = df['DATAMOVIMENTO'].min()
    data_max = df['DATAMOVIMENTO'].max()
    
    # Filtro de Data
    st.sidebar.subheader("Filtro de Período")
    data_selecionada = st.sidebar.date_input(
        "Selecione o mês/ano inicial:",
        value=data_min,
        min_value=data_min,
        max_value=data_max
    )
    
    # Converter data selecionada para datetime
    data_filtro = pd.to_datetime(data_selecionada)
    
    # Filtro de Filial
    filiais_disponiveis = ['Todas'] + sorted([str(x) for x in df['CODIGOFILIAL'].unique() if pd.notna(x)])
    filial_selecionada = st.sidebar.selectbox(
        'Selecione a Filial:',
        filiais_disponiveis
    )
    
    # Filtro de Cluster
    clusters_disponiveis = ['Todos'] + sorted([str(x) for x in df['NOME_CLUSTER'].unique() if pd.notna(x)])
    cluster_selecionado = st.sidebar.selectbox(
        'Selecione o Cluster:',
        clusters_disponiveis
    )
    
    # Filtro de Rede
    redes_disponiveis = ['Todas'] + sorted([str(x) for x in df['REDE'].unique() if pd.notna(x)])
    rede_selecionada = st.sidebar.selectbox(
        'Selecione a Rede:',
        redes_disponiveis
    )
    
    # Filtro de Cliente
    clientes_disponiveis = ['Todos'] + sorted([str(x) for x in df['CLIENTE'].unique() if pd.notna(x)])
    cliente_selecionado = st.sidebar.selectbox(
        'Selecione o Cliente:',
        clientes_disponiveis
    )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    # Aplicar filtro de data (maior ou igual à data selecionada)
    df_filtrado = df_filtrado[df_filtrado['DATAMOVIMENTO'] >= data_filtro]
    
    if filial_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['CODIGOFILIAL'].astype(str) == filial_selecionada]
    
    if cluster_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['NOME_CLUSTER'].astype(str) == cluster_selecionado]
    
    if rede_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['REDE'].astype(str) == rede_selecionada]
    
    if cliente_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['CLIENTE'].astype(str) == cliente_selecionado]
    
    # Mostrar contagem de registros após filtros
    st.sidebar.markdown("---")
    st.sidebar.write("### Resumo dos Dados")
    st.sidebar.write(f"Total de registros: {len(df_filtrado):,}")
    
    # Verificar se existem dados após os filtros
    if len(df_filtrado) > 0:
        st.sidebar.write(f"Período dos dados filtrados:")
        st.sidebar.write(f"De: {df_filtrado['DATAMOVIMENTO'].min().strftime('%d/%m/%Y')}")
        st.sidebar.write(f"Até: {df_filtrado['DATAMOVIMENTO'].max().strftime('%d/%m/%Y')}")
        
        # Mostrar quantidade de meses no período
        meses_periodo = (df_filtrado['DATAMOVIMENTO'].max().year - df_filtrado['DATAMOVIMENTO'].min().year) * 12 + \
                       df_filtrado['DATAMOVIMENTO'].max().month - df_filtrado['DATAMOVIMENTO'].min().month + 1
        st.sidebar.write(f"Total de meses: {meses_periodo}")
    else:
        st.sidebar.warning("Nenhum dado encontrado para os filtros selecionados.")

# Adicionar seletor de modo de coorte na sidebar
st.sidebar.markdown("---")
modo_coorte = st.sidebar.radio(
    "Modo de Análise de Coorte:",
    ["Normal", "Ajustado"],
    help="Normal: Considera a primeira compra como safra\nAjustado: Reinicia a safra quando há interrupção nas compras"
)

# Navegação entre páginas
pagina = st.sidebar.selectbox("Escolha uma página:", ["Página Inicial", "Página 1", "Página 2"])

if pagina == "Página Inicial":
    st.write("Esta é a página inicial.")
    
elif pagina == "Página 1":
    st.write("Bem-vindo à Página 1!")
    
    if df is not None:
        st.success("Dados carregados com sucesso!")
        
        # Criar análise de coorte com o modo selecionado
        retention_matrix = criar_cohort_analysis(df_filtrado, modo=modo_coorte)
        
        if retention_matrix is not None:
            st.write(f"## Análise de Coorte - Modo {modo_coorte}")
            
            # Adicionar informações dos filtros aplicados
            if filial_selecionada != 'Todas' or cluster_selecionado != 'Todos':
                st.write("### Filtros Aplicados:")
                if filial_selecionada != 'Todas':
                    st.write(f"- Filial: {filial_selecionada}")
                if cluster_selecionado != 'Todos':
                    st.write(f"- Cluster: {cluster_selecionado}")
            
            # Criar e exibir o heatmap
            fig = plotar_heatmap_cohort(retention_matrix)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
            
elif pagina == "Página 2":
    st.write("## Análise de Vendas")
    
    if df is not None:
        # Adicionar informações dos filtros aplicados
        if filial_selecionada != 'Todas' or cluster_selecionado != 'Todos':
            st.write("### Filtros Aplicados:")
            if filial_selecionada != 'Todas':
                st.write(f"- Filial: {filial_selecionada}")
            if cluster_selecionado != 'Todos':
                st.write(f"- Cluster: {cluster_selecionado}")
        
        # Criar tabs para as diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["Cobertura de Clientes", "Faturamento", "Margem"])
        
        with tab1:
            st.write("### Tabela de Cobertura de Clientes")
            tabela_cobertura = criar_tabela_cobertura(df_filtrado)
            
            if tabela_cobertura is not None:
                st.dataframe(
                    tabela_cobertura,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("""
                **Como interpretar a tabela:**
                - As linhas mostram o mês de primeira compra (safra) dos clientes
                - As colunas mostram os meses subsequentes de compra
                - Os números representam a quantidade de clientes únicos
                - A coluna 'Total Geral' mostra o total de clientes por safra
                """)
        
        with tab2:
            st.write("### Tabela de Faturamento")
            tabela_faturamento = criar_tabela_faturamento(df_filtrado)
            
            if tabela_faturamento is not None:
                st.dataframe(
                    tabela_faturamento.style.format("R$ {:,.2f}"),
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("""
                **Como interpretar a tabela:**
                - As linhas mostram o mês de primeira compra (safra) dos clientes
                - As colunas mostram os meses subsequentes de compra
                - Os valores representam o faturamento total
                - A coluna 'Total Geral' mostra o faturamento total por safra
                """)
        
        with tab3:
            st.write("### Tabela de Margem")
            tabela_margem = criar_tabela_margem(df_filtrado)
            
            if tabela_margem is not None:
                st.dataframe(
                    tabela_margem.style.format("R$ {:,.2f}"),
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("""
                **Como interpretar a tabela:**
                - As linhas mostram o mês de primeira compra (safra) dos clientes
                - As colunas mostram os meses subsequentes de compra
                - Os valores representam a margem (Faturamento - Custo)
                - A coluna 'Total Geral' mostra a margem total por safra
                """)
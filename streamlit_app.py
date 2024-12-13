import streamlit as st
from st_files_connection import FilesConnection
#import cx_Oracle
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import boto3
import io

# Configurar a página
st.set_page_config(
    page_title="Análise Coorte e Retenção",  # Título da página
    page_icon="📊",  # Ícone da página (opcional)
    layout="wide"  # Ativar o Wide Mode
)

# Definir imagem de fundo
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://villacamarao.com.br/wp-content/uploads/2021/05/Prancheta1_3.svg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Função para carregar dados
@st.cache_data(ttl=timedelta(hours=12))
def carregar_dados():
    try:
        # Criar cliente S3 usando as credenciais do arquivo .streamlit/secrets.toml
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
            aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
            region_name=st.secrets["aws"]["aws_default_region"]
        )
        
        # Definir bucket e arquivo
        bucket = 'datalake-out-etl'
        file_key = 'base-cohort-analysis/base-cohort-analysis.csv'
        
        # Baixar arquivo do S3
        obj = s3_client.get_object(Bucket=bucket, Key=file_key)
        
        # Ler CSV para DataFrame
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        # Validar se o DataFrame foi carregado corretamente
        if df is None or df.empty:
            raise Exception("Nenhum dado foi carregado do arquivo CSV")
            
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Função para criar análise de coorte (mantém igual)
def criar_cohort_analysis(df, modo='Normal'):
    """
    Função para criar a análise de coorte com dois modos diferentes
    """
    try:
        # Criar cópia do DataFrame
        df_cohort = df.copy()
        
        # Converter datamovimento para datetime
        df_cohort['datamovimento'] = pd.to_datetime(df_cohort['datamovimento'])
        
        # Obter data mínima e máxima do DataFrame
        data_inicial = df_cohort['datamovimento'].min().replace(day=1)
        data_final = df_cohort['datamovimento'].max().replace(day=1)
        
        if modo == 'Normal':
            df_cohort['COHORT_MES'] = df_cohort.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        else:
            # Modo ajustado - identifica quebras na sequência de compras
            df_cohort = df_cohort.sort_values(['codigocliente', 'datamovimento'])
            df_cohort['MES_ANO'] = df_cohort['datamovimento'].dt.strftime('%Y-%m')
            df_cohort['MES_ANTERIOR'] = df_cohort.groupby('codigocliente')['datamovimento'].shift()
            df_cohort['MESES_DIFF'] = (df_cohort['datamovimento'].dt.year * 12 + df_cohort['datamovimento'].dt.month) - \
                                    (df_cohort['MES_ANTERIOR'].dt.year * 12 + df_cohort['MES_ANTERIOR'].dt.month)
            df_cohort['NOVA_SAFRA'] = (df_cohort['MESES_DIFF'] > 1) | (df_cohort['MES_ANTERIOR'].isna())
            df_cohort['GRUPO_SAFRA'] = df_cohort.groupby('codigocliente')['NOVA_SAFRA'].cumsum()
            df_cohort['COHORT_MES'] = df_cohort.groupby(['codigocliente', 'GRUPO_SAFRA'])['datamovimento'].transform('min').dt.strftime('%Y-%m')
        
        # Criar mês da transação
        df_cohort['MES_TRANSACAO'] = df_cohort['datamovimento'].dt.strftime('%Y-%m')
        
        # Converter para datetime para cálculo correto do período
        df_cohort['COHORT_MES'] = pd.to_datetime(df_cohort['COHORT_MES'])
        df_cohort['MES_TRANSACAO'] = pd.to_datetime(df_cohort['MES_TRANSACAO'])
        
        # Calcular o índice do período
        df_cohort['PERIODO_INDEX'] = ((df_cohort['MES_TRANSACAO'].dt.year - df_cohort['COHORT_MES'].dt.year) * 12 +
                                    (df_cohort['MES_TRANSACAO'].dt.month - df_cohort['COHORT_MES'].dt.month))
        
        # Criar matriz de coorte
        cohort_data = df_cohort.groupby(['COHORT_MES', 'PERIODO_INDEX'])['codigocliente'].nunique().reset_index()
        
        # Criar todas as combinações possíveis de safras e períodos
        cohort_meses = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        max_periodo = ((data_final.year - data_inicial.year) * 12 + 
                      data_final.month - data_inicial.month)
        
        todas_combinacoes = pd.DataFrame([(cohort, periodo) 
                                        for cohort in cohort_meses 
                                        for periodo in range(int(max_periodo) + 1)],
                                       columns=['COHORT_MES', 'PERIODO_INDEX'])
        
        # Fazer merge com os dados reais
        cohort_data = todas_combinacoes.merge(cohort_data, 
                                            on=['COHORT_MES', 'PERIODO_INDEX'], 
                                            how='left')
        
        # Preencher valores nulos com 0
        cohort_data['codigocliente'] = cohort_data['codigocliente'].fillna(0)
        
        # Converter COHORT_MES para string no formato adequado
        cohort_data['COHORT_MES'] = cohort_data['COHORT_MES'].dt.strftime('%Y-%m')
        
        # Criar matriz pivotada
        cohort_matrix = cohort_data.pivot(index='COHORT_MES',
                                        columns='PERIODO_INDEX',
                                        values='codigocliente')
        
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
        df_cobertura['datamovimento'] = pd.to_datetime(df_cobertura['datamovimento'])
        
        # Criar colunas de ano-mês para safra e movimento
        df_cobertura['SAFRA'] = df_cobertura.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        df_cobertura['MOVIMENTO'] = df_cobertura['datamovimento'].dt.strftime('%Y-%m')
        
        # Agrupar e contar clientes únicos
        tabela_cobertura = df_cobertura.groupby(['SAFRA', 'MOVIMENTO'])['codigocliente'].nunique().reset_index()
        
        # Obter data mínima e máxima do DataFrame
        data_inicial = df_cobertura['datamovimento'].min().replace(day=1)
        data_final = df_cobertura['datamovimento'].max().replace(day=1)
        
        # Criar range de datas para safras e movimentos
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas_str = todas_datas.strftime('%Y-%m')
        
        # Criar todas as combinações possíveis
        todas_combinacoes = pd.DataFrame([(safra, movimento) 
                                        for safra in todas_datas_str 
                                        for movimento in todas_datas_str],
                                       columns=['SAFRA', 'MOVIMENTO'])
        
        # Fazer merge com os dados reais
        tabela_final = todas_combinacoes.merge(tabela_cobertura, 
                                             on=['SAFRA', 'MOVIMENTO'], 
                                             how='left')
        
        # Criar tabela pivotada
        tabela_final = tabela_final.pivot(index='SAFRA', 
                                        columns='MOVIMENTO', 
                                        values='codigocliente')
        
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
        
        # Converter datamovimento para datetime
        df_faturamento['datamovimento'] = pd.to_datetime(df_faturamento['datamovimento'])
        
        # Obter data mínima e máxima do DataFrame
        data_inicial = df_faturamento['datamovimento'].min().replace(day=1)
        data_final = df_faturamento['datamovimento'].max().replace(day=1)
        
        # Criar range de datas baseado nos dados
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas = pd.DataFrame({'MOVIMENTO': todas_datas.strftime('%Y-%m')})
        
        # Criar colunas de ano-mês para safra e movimento
        df_faturamento['SAFRA'] = df_faturamento.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        df_faturamento['MOVIMENTO'] = df_faturamento['datamovimento'].dt.strftime('%Y-%m')
        
        # Agrupar e somar faturamento
        tabela_faturamento = df_faturamento.groupby(['SAFRA', 'MOVIMENTO'])['faturamento'].sum().reset_index()
        
        # Criar todas as combinações possíveis
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas_str = todas_datas.strftime('%Y-%m')
        
        todas_combinacoes = pd.DataFrame([(safra, movimento) 
                                        for safra in todas_datas_str 
                                        for movimento in todas_datas_str],
                                       columns=['SAFRA', 'MOVIMENTO'])
        
        # Fazer merge com os dados reais
        tabela_final = todas_combinacoes.merge(tabela_faturamento, 
                                             on=['SAFRA', 'MOVIMENTO'], 
                                             how='left')
        
        # Preencher valores nulos com 0
        tabela_final['faturamento'] = tabela_final['faturamento'].fillna(0)
        
        # Criar tabela pivotada
        tabela_pivot = tabela_final.pivot(index='SAFRA', 
                                        columns='MOVIMENTO', 
                                        values='faturamento')
        
        # Ordenar as colunas cronologicamente
        tabela_pivot = tabela_pivot.reindex(sorted(tabela_pivot.columns), axis=1)
        
        # Adicionar total por linha
        tabela_pivot['Total Geral'] = tabela_pivot.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_pivot.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Faturamento Inicial'])
        for coluna in tabela_pivot.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                valores_iniciais[coluna] = tabela_pivot[coluna][tabela_pivot.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_pivot, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).round(2)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de faturamento: {e}")
        return None

def criar_tabela_margem(df):
    """
    Função para criar a tabela de margem de vendas
    """
    try:
        # Criar cópia do DataFrame
        df_margem = df.copy()
        
        # Converter datamovimento para datetime
        df_margem['datamovimento'] = pd.to_datetime(df_margem['datamovimento'])
        
        # Obter data mínima e máxima do DataFrame
        data_inicial = df_margem['datamovimento'].min().replace(day=1)
        data_final = df_margem['datamovimento'].max().replace(day=1)
        
        # Criar range de datas baseado nos dados
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas = pd.DataFrame({'MOVIMENTO': todas_datas.strftime('%Y-%m')})
        
        # Calcular o custo total e a margem
        df_margem['custototal'] = df_margem['custo'] #* df_margem['quantidade']
        df_margem['margem'] = df_margem['faturamento'] - df_margem['custototal']
        
        # Criar colunas de ano-mês para safra e movimento
        df_margem['SAFRA'] = df_margem.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        df_margem['MOVIMENTO'] = df_margem['datamovimento'].dt.strftime('%Y-%m')
        
        # Agrupar e somar margem
        tabela_margem = df_margem.groupby(['SAFRA', 'MOVIMENTO'])['margem'].sum().reset_index()
        
        # Criar todas as combinações possíveis
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas_str = todas_datas.strftime('%Y-%m')
        
        todas_combinacoes = pd.DataFrame([(safra, movimento) 
                                        for safra in todas_datas_str 
                                        for movimento in todas_datas_str],
                                       columns=['SAFRA', 'MOVIMENTO'])
        
        # Fazer merge com os dados reais
        tabela_final = todas_combinacoes.merge(tabela_margem, 
                                             on=['SAFRA', 'MOVIMENTO'], 
                                             how='left')
        
        # Preencher valores nulos com 0
        tabela_final['margem'] = tabela_final['margem'].fillna(0)
        
        # Criar tabela pivotada
        tabela_pivot = tabela_final.pivot(index='SAFRA', 
                                        columns='MOVIMENTO', 
                                        values='margem')
        
        # Ordenar as colunas cronologicamente
        tabela_pivot = tabela_pivot.reindex(sorted(tabela_pivot.columns), axis=1)
        
        # Adicionar total por linha
        tabela_pivot['Total Geral'] = tabela_pivot.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_pivot.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Margem Inicial'])
        for coluna in tabela_pivot.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                valores_iniciais[coluna] = tabela_pivot[coluna][tabela_pivot.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_pivot, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).round(2)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de margem: {e}")
        return None

# Função para criar tabela de notas emitidas
def criar_tabela_notas(df):
    """
    Função para criar a tabela de notas emitidas
    """
    try:
        # Criar cópia do DataFrame
        df_notas = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_notas['datamovimento'] = pd.to_datetime(df_notas['datamovimento'])
        
        # Criar colunas de ano-mês para safra e movimento
        df_notas['SAFRA'] = df_notas.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        df_notas['MOVIMENTO'] = df_notas['datamovimento'].dt.strftime('%Y-%m')
        
        # Agrupar e contar notas emitidas de forma distinta
        tabela_notas = df_notas.groupby(['SAFRA', 'MOVIMENTO'])['notasaida'].nunique().reset_index()
        
        # Obter data mínima e máxima do DataFrame
        data_inicial = df_notas['datamovimento'].min().replace(day=1)
        data_final = df_notas['datamovimento'].max().replace(day=1)
        
        # Criar range de datas para safras e movimentos
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas_str = todas_datas.strftime('%Y-%m')
        
        # Criar todas as combinações possíveis
        todas_combinacoes = pd.DataFrame([(safra, movimento) 
                                        for safra in todas_datas_str 
                                        for movimento in todas_datas_str],
                                       columns=['SAFRA', 'MOVIMENTO'])
        
        # Fazer merge com os dados reais
        tabela_final = todas_combinacoes.merge(tabela_notas, 
                                             on=['SAFRA', 'MOVIMENTO'], 
                                             how='left')
        
        # Preencher valores nulos com 0
        tabela_final['notasaida'] = tabela_final['notasaida'].fillna(0)
        
        # Criar tabela pivotada
        tabela_pivot = tabela_final.pivot(index='SAFRA', 
                                        columns='MOVIMENTO', 
                                        values='notasaida')
        
        # Ordenar as colunas cronologicamente
        tabela_pivot = tabela_pivot.reindex(sorted(tabela_pivot.columns), axis=1)
        
        # Adicionar total por linha
        tabela_pivot['Total Geral'] = tabela_pivot.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_pivot.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Notas Iniciais'])
        for coluna in tabela_pivot.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                valores_iniciais[coluna] = tabela_pivot[coluna][tabela_pivot.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_pivot, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).astype(int)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de notas: {e}")
        return None

# Função para criar tabela de quantidade de vendas
def criar_tabela_quantidade(df):
    """
    Função para criar a tabela de quantidade de vendas
    """
    try:
        # Criar cópia do DataFrame
        df_quantidade = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_quantidade['datamovimento'] = pd.to_datetime(df_quantidade['datamovimento'])
        
        # Criar colunas de ano-mês para safra e movimento
        df_quantidade['SAFRA'] = df_quantidade.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        df_quantidade['MOVIMENTO'] = df_quantidade['datamovimento'].dt.strftime('%Y-%m')
        
        # Agrupar e somar quantidade
        tabela_quantidade = df_quantidade.groupby(['SAFRA', 'MOVIMENTO'])['quantidade'].sum().reset_index()
        
        # Obter data mínima e máxima do DataFrame
        data_inicial = df_quantidade['datamovimento'].min().replace(day=1)
        data_final = df_quantidade['datamovimento'].max().replace(day=1)
        
        # Criar range de datas para safras e movimentos
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas_str = todas_datas.strftime('%Y-%m')
        
        # Criar todas as combinações possíveis
        todas_combinacoes = pd.DataFrame([(safra, movimento) 
                                        for safra in todas_datas_str 
                                        for movimento in todas_datas_str],
                                       columns=['SAFRA', 'MOVIMENTO'])
        
        # Fazer merge com os dados reais
        tabela_final = todas_combinacoes.merge(tabela_quantidade, 
                                             on=['SAFRA', 'MOVIMENTO'], 
                                             how='left')
        
        # Preencher valores nulos com 0
        tabela_final['quantidade'] = tabela_final['quantidade'].fillna(0)
        
        # Criar tabela pivotada
        tabela_pivot = tabela_final.pivot(index='SAFRA', 
                                        columns='MOVIMENTO', 
                                        values='quantidade')
        
        # Ordenar as colunas cronologicamente
        tabela_pivot = tabela_pivot.reindex(sorted(tabela_pivot.columns), axis=1)
        
        # Adicionar total por linha
        tabela_pivot['Total Geral'] = tabela_pivot.sum(axis=1)
        
        # Adicionar total por coluna
        total_colunas = tabela_pivot.sum().to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Quantidade Inicial'])
        for coluna in tabela_pivot.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                valores_iniciais[coluna] = tabela_pivot[coluna][tabela_pivot.index == coluna].fillna(0).sum()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_pivot, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).astype(int)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de quantidade: {e}")
        return None

# Função para criar tabela de margem percentual
def criar_tabela_margem_percentual(df):
    """
    Função para criar a tabela de margem percentual
    """
    try:
        # Criar cópia do DataFrame
        df_margem = df.copy()
        
        # Converter DATAMOVIMENTO para datetime
        df_margem['datamovimento'] = pd.to_datetime(df_margem['datamovimento'])
        
        # Criar colunas de ano-mês para safra e movimento
        df_margem['SAFRA'] = df_margem.groupby('codigocliente')['datamovimento'].transform('min').dt.strftime('%Y-%m')
        df_margem['MOVIMENTO'] = df_margem['datamovimento'].dt.strftime('%Y-%m')
        
        # Calcular a margem
        df_margem['margem'] = df_margem['faturamento'] - df_margem['custo']
        
        # Calcular a margem percentual
        df_margem['margem_percentual'] = (df_margem['margem'] / df_margem['faturamento']) * 100
        
        # Agrupar e calcular a soma da margem e do faturamento
        tabela_margem_percentual = df_margem.groupby(['SAFRA', 'MOVIMENTO']).agg(
            margem_total=('margem', 'sum'),
            faturamento_total=('faturamento', 'sum')
        ).reset_index()
        
        # Calcular a margem percentual
        tabela_margem_percentual['margem_percentual'] = (tabela_margem_percentual['margem_total'] / tabela_margem_percentual['faturamento_total']) * 100
        
        # Formatar os números para usar vírgula como separador decimal
        # tabela_margem_percentual['margem_percentual'] = tabela_margem_percentual['margem_percentual'].apply(lambda x: f"{x:,.2f}".replace('.', ','))

        # Obter data mínima e máxima do DataFrame
        data_inicial = df_margem['datamovimento'].min().replace(day=1)
        data_final = df_margem['datamovimento'].max().replace(day=1)
        
        # Criar range de datas para safras e movimentos
        todas_datas = pd.date_range(start=data_inicial, end=data_final, freq='MS')
        todas_datas_str = todas_datas.strftime('%Y-%m')
        
        # Criar todas as combinações possíveis
        todas_combinacoes = pd.DataFrame([(safra, movimento) 
                                        for safra in todas_datas_str 
                                        for movimento in todas_datas_str],
                                       columns=['SAFRA', 'MOVIMENTO'])
        
        # Fazer merge com os dados reais
        tabela_final = todas_combinacoes.merge(tabela_margem_percentual, 
                                             on=['SAFRA', 'MOVIMENTO'], 
                                             how='left')
        
        # Preencher valores nulos com 0
        tabela_final['margem_percentual'] = tabela_final['margem_percentual'].fillna(0)
        
        # Criar tabela pivotada
        tabela_pivot = tabela_final.pivot(index='SAFRA', 
                                        columns='MOVIMENTO', 
                                        values='margem_percentual')
        
        # Ordenar as colunas cronologicamente
        tabela_pivot = tabela_pivot.reindex(sorted(tabela_pivot.columns), axis=1)
        
        # Adicionar total por linha
        tabela_pivot['Total Geral'] = (tabela_pivot.sum(axis=1) / tabela_final.groupby('SAFRA')['faturamento_total'].sum()).fillna(0) * 100
        
        # Adicionar total por coluna
        total_colunas = (tabela_pivot.sum() / tabela_final.groupby('MOVIMENTO')['faturamento_total'].sum()).to_frame().T
        total_colunas.index = ['Total Geral']
        
        # Criar linha com valores iniciais de cada safra
        valores_iniciais = pd.DataFrame(index=['Margem Inicial'])
        for coluna in tabela_pivot.columns:
            if coluna == 'Total Geral':
                valores_iniciais[coluna] = 0
            else:
                valores_iniciais[coluna] = tabela_pivot[coluna][tabela_pivot.index == coluna].fillna(0).mean()
        
        # Concatenar com a tabela original e totais
        tabela_final = pd.concat([tabela_pivot, total_colunas, valores_iniciais])
        
        # Formatar tabela
        tabela_final = tabela_final.fillna(0).round(2)
        
        return tabela_final
        
    except Exception as e:
        st.error(f"Erro ao criar tabela de margem percentual: {e}")
        return None

# Carregar dados uma única vez
df = carregar_dados()

# Renomear colunas para minúsculas
# df.columns = df.columns.str.lower()

# Verificar as colunas do DataFrame
#st.write("Colunas disponíveis no DataFrame:", df.columns.tolist())

# Remover filiais 1 e 5 do DataFrame
if df is not None:
    df = df[~df['codigofilial'].astype(str).isin(['1', '5', '2'])]
    
    # Resetar o índice após a remoção
    df = df.reset_index(drop=True)

# Título da aplicação
st.title("Análise Coorte e Retenção")
st.subheader("Villa Camarão!")

# Adicionar logo
st.image("https://villacamarao.com.br/wp-content/uploads/2021/05/Prancheta1_3.svg", width=150)

# Sidebar com filtros
st.sidebar.header("Filtros")

# Inicializar filtros apenas se df não for None
if df is not None:
    # Converter DATAMOVIMENTO para datetime se ainda não estiver
    df['datamovimento'] = pd.to_datetime(df['datamovimento'])
    
    # Obter data mínima e máxima do DataFrame
    data_min = df['datamovimento'].min()
    data_max = df['datamovimento'].max()
    
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
    filiais_disponiveis = ['Todas'] + sorted([str(x) for x in df['codigofilial'].unique() if pd.notna(x)])
    filial_selecionada = st.sidebar.selectbox(
        'Selecione a Filial:',
        filiais_disponiveis
    )
    
        # Filtro de Gerente de Carteira
    gerentes_carteira_disponiveis = ['Todos'] + sorted([str(x) for x in df['gerentecarteira'].unique() if pd.notna(x)])
    gerente_carteira_selecionado = st.sidebar.selectbox(
        'Selecione o Gerente de Carteira:',
        gerentes_carteira_disponiveis
    )
    
    # Filtro de Gerente de Venda
    gerentes_venda_disponiveis = ['Todos'] + sorted([str(x) for x in df['gerentevenda'].unique() if pd.notna(x)])
    gerente_venda_selecionado = st.sidebar.selectbox(
        'Selecione o Gerente de Venda:',
        gerentes_venda_disponiveis
    )

    # Filtro de Cluster
    clusters_disponiveis = ['Todos'] + sorted([str(x) for x in df['nome_cluster'].unique() if pd.notna(x)])
    cluster_selecionado = st.sidebar.selectbox(
        'Selecione o Cluster:',
        clusters_disponiveis
    )
    
    # Filtro de Rede
    redes_disponiveis = ['Todas'] + sorted([str(x) for x in df['rede'].unique() if pd.notna(x)])
    rede_selecionada = st.sidebar.selectbox(
        'Selecione a Rede:',
        redes_disponiveis
    )
    
    # Filtro de Cliente
    clientes_disponiveis = ['Todos'] + sorted([str(x) for x in df['cliente'].unique() if pd.notna(x)])
    cliente_selecionado = st.sidebar.selectbox(
        'Selecione o Cliente:',
        clientes_disponiveis
    )
    
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    # Aplicar filtro de data (maior ou igual à data selecionada)
    df_filtrado = df_filtrado[df_filtrado['datamovimento'] >= data_filtro]
    
    if filial_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['codigofilial'].astype(str) == filial_selecionada]
    
    if cluster_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['nome_cluster'].astype(str) == cluster_selecionado]
    
    if rede_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['rede'].astype(str) == rede_selecionada]
    
    if cliente_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['cliente'].astype(str) == cliente_selecionado]
    
    # Aplicar filtro de Gerente de Carteira
    if gerente_carteira_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['gerentecarteira'].astype(str) == gerente_carteira_selecionado]
    
    # Aplicar filtro de Gerente de Venda
    if gerente_venda_selecionado != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['gerentevenda'].astype(str) == gerente_venda_selecionado]
    
    # Mostrar contagem de registros após filtros
    st.sidebar.markdown("---")
    st.sidebar.write("### Resumo dos Dados")
    st.sidebar.write(f"Total de registros: {len(df_filtrado):,}")
    
    # Verificar se existem dados após os filtros
    if len(df_filtrado) > 0:
        st.sidebar.write(f"Período dos dados filtrados:")
        st.sidebar.write(f"De: {df_filtrado['datamovimento'].min().strftime('%d/%m/%Y')}")
        st.sidebar.write(f"Até: {df_filtrado['datamovimento'].max().strftime('%d/%m/%Y')}")
        
        # Mostrar quantidade de meses no período
        meses_periodo = (df_filtrado['datamovimento'].max().year - df_filtrado['datamovimento'].min().year) * 12 + \
                       df_filtrado['datamovimento'].max().month - df_filtrado['datamovimento'].min().month + 1
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
pagina = st.sidebar.selectbox("Escolha uma página:", ["Página Inicial", "Gráfico de retenção", "Análise Coorte"])

if pagina == "Página Inicial":
    st.write("## Visualização dos Dados")
    
    if df is not None:
        # Mostrar as primeiras linhas do DataFrame
        # st.write("### Primeiras linhas do DataFrame:")
        # st.dataframe(df.head(10))
        st.markdown("""
        **Proposta do projeto:**
        - Entender o comportamento de compra (safra) dos clientes
        - Os dados são atualizados diariamente
        - Os dados são divididos em quantidade de clientes, faturamento, margem e número de notas/pedidos
        - A coluna 'Total Geral' mostra o total por safra
        - A linha 'Total Geral' mostra o total por mês da venda
        - A linha 'Total Inicial' mostra o total da primeira venda do mês
        - Utilize a barra lateral para filtrar os registros e navegar entre as páginas
        """)
        
        # Mostrar informações básicas sobre o DataFrame
        st.write("### Informações do DataFrame:")
        st.write(f"Total de registros: {len(df):,}")
        st.write(f"Total de colunas: {len(df.columns)}")
        st.write("Todos os direitos reservados.")
    
elif pagina == "Gráfico de retenção":
    st.write("Bem-vindo à Página de Gráfico de retenção!")
    
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
            
elif pagina == "Análise Coorte":
    #st.write("## Análise Coorte")
    
    if df is not None:
        # Adicionar informações dos filtros aplicados
        if filial_selecionada != 'Todas' or cluster_selecionado != 'Todos':
            st.write("#### Filtros Aplicados:")
            if filial_selecionada != 'Todas':
                st.write(f"- Filial: {filial_selecionada}")
            if cluster_selecionado != 'Todos':
                st.write(f"- Cluster: {cluster_selecionado}")
        
        # Criar tabs para as diferentes visualizações
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Cobertura de Clientes", "Faturamento", "Margem", "Margem Percentual", "Notas Emitidas", "Volume"])
        
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

        with tab4:  # Nova aba para a tabela de margem percentual
            st.write("### Tabela de Margem Percentual")
            tabela_margem_percentual = criar_tabela_margem_percentual(df_filtrado)
            
            if tabela_margem_percentual is not None:
                st.dataframe(
                    tabela_margem_percentual,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown(""" 
                **Como interpretar a tabela:**
                - As linhas mostram o mês de primeira compra (safra) dos clientes
                - As colunas mostram os meses subsequentes de compra
                - Os valores representam a margem percentual
                - A coluna 'Total Geral' mostra a média da margem percentual por safra
                """)

        with tab5:  # Nova aba para a tabela de notas
            st.write("### Tabela de Notas Emitidas")
            tabela_notas = criar_tabela_notas(df_filtrado)
            
            if tabela_notas is not None:
                st.dataframe(
                    tabela_notas,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown(""" 
                **Como interpretar a tabela:**
                - As linhas mostram o mês de primeira compra (safra) dos clientes
                - As colunas mostram os meses subsequentes de compra
                - Os valores representam a quantidade de notas emitidas
                - A coluna 'Total Geral' mostra o total de notas emitidas por safra
                """)

        with tab6:  # Nova aba para a tabela de quantidade de vendas
            st.write("### Tabela de Volume Vendido")
            tabela_quantidade = criar_tabela_quantidade(df_filtrado)
            
            if tabela_quantidade is not None:
                st.dataframe(
                    tabela_quantidade,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown(""" 
                **Como interpretar a tabela:**
                - As linhas mostram o mês de primeira compra (safra) dos clientes
                - As colunas mostram os meses subsequentes de compra
                - Os valores representam a quantidade total de vendas
                - A coluna 'Total Geral' mostra a quantidade total por safra
                """)


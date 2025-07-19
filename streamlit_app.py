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
import time  # Importar time para simular a atualização
import re
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Configurar a página
st.set_page_config(
    page_title="Análise Coorte e Retenção, Modelagem e Machine Learning",  # Título da página
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
def get_aws_credentials():
    """
    Obtém as credenciais da AWS de forma segura, priorizando variáveis de ambiente.
    Retorna um dicionário com as credenciais.
    """
    # Prioridade 1: Variáveis de ambiente (Railway, Docker, etc.)
    if "AWS_ACCESS_KEY_ID" in os.environ:
        st.info("Carregando credenciais AWS a partir das variáveis de ambiente.")
        return {
            "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "aws_default_region": os.environ["AWS_DEFAULT_REGION"]
        }
    
    # Prioridade 2: st.secrets (Streamlit Cloud)
    # Verifica se st.secrets existe e se a seção 'aws' está presente
    elif hasattr(st, 'secrets') and "aws" in st.secrets:
        st.info("Carregando credenciais AWS a partir do Streamlit Secrets.")
        return {
            "aws_access_key_id": st.secrets["aws"]["aws_access_key_id"],
            "aws_secret_access_key": st.secrets["aws"]["aws_secret_access_key"],
            "aws_default_region": st.secrets["aws"]["aws_default_region"]
        }
    
    else:
        # Se nenhuma das opções estiver disponível, levanta um erro claro.
        raise ValueError("Credenciais da AWS não encontradas. Configure as variáveis de ambiente ou o arquivo secrets.toml.")

# Agora, refatore sua função carregar_dados para usar essa nova função.
def carregar_dados():
    try:
        # 1. Obter as credenciais de forma segura
        aws_creds = get_aws_credentials()

        # 2. Criar o cliente S3 usando as credenciais desempacotadas
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_creds["aws_access_key_id"],
            aws_secret_access_key=aws_creds["aws_secret_access_key"],
            region_name=aws_creds["aws_default_region"]
            # Alternativa mais limpa: boto3.client('s3', **aws_creds)
        )
        
        # 3. O resto da sua lógica permanece igual
        bucket = 'datalake-out-etl'
        file_key = 'base-cohort-analysis/base-cohort-analysis.csv'
        
        obj = s3_client.get_object(Bucket=bucket, Key=file_key)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        if df is None or df.empty:
            raise Exception("Nenhum dado foi carregado do arquivo CSV")
            
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        # A mensagem de erro agora será muito mais específica (ex: do ValueError ou do boto3)
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

# Função para carregar dados com barra de progresso
def carregar_dados_com_progresso():
    with st.spinner("Atualizando a base de dados..."):
        # Simular um tempo de carregamento (substitua isso pela sua lógica de carregamento)
        time.sleep(2)  # Simula um atraso de 2 segundos
        
        # Aqui você deve chamar a função real para carregar os dados
        df = carregar_dados()  # Chame sua função de carregamento de dados
        st.session_state.df = df  # Armazena o DataFrame no session_state
        return df

# Carregar dados uma única vez
df = carregar_dados_com_progresso()

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

# Verifica se as credenciais da Microsoft estão nas variáveis de ambiente
if "MS_CLIENT_ID" not in os.environ or "MS_CLIENT_SECRET" not in os.environ:
    st.error("As credenciais de autenticação da Microsoft não foram configuradas. Contate o administrador.")
else:
    # Configura o provedor de autenticação
    auth_providers = {
        "microsoft": {
            "client_id": os.environ.get("MS_CLIENT_ID"),
            "client_secret": os.environ.get("MS_CLIENT_SECRET"),
            "redirect_uri": "https://cohort-analysis-production.up.railway.app/_st_login", # IMPORTANTE: Usar a URL de produção aqui
            "tenant_id": "common", # Ou o ID do seu tenant específico para maior segurança
            "label": "Login com Microsoft (Villa Camarão)"
        }
    }

    # Exibe o botão de login
    login_button = st.login(providers="microsoft")

    if login_button.is_logged_in:
        # Pega o email do usuário logado
        user_email = login_button.user.email
        #allowed_domain = "villacamarao.com.br"
        
        st.sidebar.write(f"Logado como: {user_email}")

        # Verifica se o domínio do email é o permitido
        if user_email.endswith(allowed_domain):
            # --- SE O ACESSO FOR PERMITIDO, MOSTRA O APP ---
            st.header("Dashboard Principal")
            
            # Carrega e exibe os dados
            df_cohort = carregar_dados()

            if df_cohort is not None:
                st.success("Dados carregados com sucesso!")
                st.dataframe(df_cohort)
                # ... coloque aqui o resto do seu aplicativo, gráficos, etc.
            else:
                st.warning("Não foi possível carregar os dados do dashboard.")

        else:
            # --- SE O DOMÍNIO NÃO FOR PERMITIDO ---
            st.error(f"Acesso Negado. Apenas usuários com o domínio '{allowed_domain}' são permitidos.")
            st.warning(f"Seu email ({user_email}) não tem permissão para acessar esta aplicação.")
            st.info("Por favor, faça logout e tente novamente com uma conta autorizada.")
            # O botão de logout já é exibido automaticamente por st.login

    else:
        st.info("Por favor, faça login para acessar o dashboard.")
        st.image("https://www.villacamarao.com.br/wp-content/uploads/2023/12/logo-villa-camarao.png", width=300)

# Fim do treche de autenticação

# Adicionar logo
st.image("https://villacamarao.com.br/wp-content/uploads/2021/05/Prancheta1_3.svg", width=150)

st.divider()

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
    
    # Filtro de UF do Cliente
    ufs_disponiveis = ['Todos'] + sorted([str(x) for x in df['ufcliente'].unique() if pd.notna(x)])
    uf_selecionada = st.sidebar.selectbox(
        'Selecione a UF do Cliente:',
        ufs_disponiveis,
        index=0  # Opção padrão (primeira UF selecionada)
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
    
    # Aplicar filtro de UF do Cliente
    if uf_selecionada  != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['ufcliente'].astype(str) == uf_selecionada]
    
    # Adicionar selectbox para escolher entre amostra e dados completos
    opcao_dados = st.sidebar.selectbox(
        "Escolha o tipo de dados:",
        ["Amostra", "Dados Completos"],
        index=1  # Amostra como padrão
    )

    # Se a opção for "Amostra", usar uma amostra dos dados
    if opcao_dados == "Amostra":
        if len(df_filtrado) > 10000:  # Exemplo de limite
            df_filtrado = df_filtrado.sample(n=10000, random_state=1)

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
pagina = st.sidebar.selectbox("Escolha uma página:", ["Página Inicial", "Gráfico de retenção", "Análise Coorte", "Análise Exploratória", "Modelagem e Treinamento"])

if pagina == "Página Inicial":
    st.subheader("Visualização dos Dados")
    
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

elif pagina == "Análise Exploratória":
    # Exibir as primeiras linhas do DataFrame
    st.subheader("Visualização dos Dados")
    st.dataframe(df_filtrado.head())

    # Análise Exploratória
    st.subheader("Análise Exploratória")

    # Exibir informações do DataFrame
    st.write("Informações do DataFrame:")
    st.write(df_filtrado.info())

    # Criar abas para organizar a análise
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Informações", "Estatísticas Descritivas", "Distribuição", "Correlação", "Pré-processamento"])

    with tab1:
        # st.write("## Bem-vindo à Página de Análise Exploratória!")
        # import analise_preditiva  # Importa a nova página
        # Faturamento total
        faturamento_total = df_filtrado['faturamento'].sum()
        st.write(f"**Faturamento Total:** R$ {faturamento_total:,.2f}")

        # Faturamento médio mensal
        faturamento_mensal = df_filtrado.groupby(df_filtrado['datamovimento'].dt.to_period('M'))['faturamento'].sum()
        faturamento_medio_mensal = faturamento_mensal.mean()
        st.write(f"**Faturamento Médio Mensal:** R$ {faturamento_medio_mensal:,.2f}")

        # Faturamento médio por cliente
        faturamento_medio_por_cliente = faturamento_total / df_filtrado['codigocliente'].nunique()
        st.write(f"**Faturamento Médio por Cliente:** R$ {faturamento_medio_por_cliente:,.2f}")

        # Faturamento médio por nota fiscal
        faturamento_medio_por_nota = faturamento_total / df_filtrado['notasaida'].nunique()
        st.write(f"**Faturamento Médio por Nota Fiscal:** R$ {faturamento_medio_por_nota:,.2f}")

        # Faturamento médio por nome_cluster
        faturamento_por_nome_cluster = df_filtrado.groupby('nome_cluster')['faturamento'].sum()
        faturamento_medio_por_nome_cluster = faturamento_por_nome_cluster.mean()
        st.write(f"**Faturamento Médio por Cluster:** R$ {faturamento_medio_por_nome_cluster:,.2f}")

        # Margem por nome_cluster
        df_filtrado.loc[:, 'margem'] = df_filtrado['faturamento'] - df_filtrado['custo']  # Calcular a margem usando .loc
        margem_por_nome_cluster = df_filtrado.groupby('nome_cluster')['margem'].mean()
        st.write("**Margem por Cluster:**")
        st.dataframe(margem_por_nome_cluster)

    with tab2:
        # Exibir estatísticas descritivas
        st.subheader("Estatísticas Descritivas")
        tabela_estatisticas = df_filtrado.describe()
        st.dataframe(tabela_estatisticas.style.background_gradient(cmap='viridis'))

        # EDA
        # Tipos de dados
        # st.write("## Análise Exploratória de Dados")
        st.subheader("Tipos de Dados das Colunas")
        st.write(df_filtrado.dtypes)

        # Existem valores ausentes ou nulos? Quais colunas são mais afetadas?
        st.subheader("Existem valores ausentes ou nulos?")
        valores_ausentes = df_filtrado.isna().sum()
        st.dataframe(valores_ausentes)

        # Identificar as colunas mais afetadas por valores ausentes ou nulos
        colunas_afetadas = valores_ausentes[valores_ausentes > 0].sort_values(ascending=False)
        st.subheader("Colunas Mais Afetadas por Valores Ausentes ou Nulos")
        st.dataframe(colunas_afetadas)

    with tab3:
        # Compreensão das variáveis
        # Qual é a variável alvo?
        st.subheader("Definição da Variável Alvo")
        variavel_alvo = st.selectbox("Selecione a Variável Alvo:", ["faturamento", "margem", "quantidade"])
        st.write(f"Variável Alvo Selecionada: {variavel_alvo}")

        # Quais variáveis são categóricas e quais são numéricas?

        # definir manualmente as variáveis que são numéricas mas que são, na verdade, categóricas.
        df_filtrado['codigocliente'] = df_filtrado['codigocliente'].astype(str)
        df_filtrado['codigofilial'] = df_filtrado['codigofilial'].astype(str)
        df_filtrado['notasaida'] = df_filtrado['notasaida'].astype(str)

        # Identificar as variáveis categóricas e numéricas, excluindo a variável alvo
        variaveis_categoricas = df_filtrado.select_dtypes(include=['object']).columns.difference(['variavel_alvo'])
        variaveis_numericas = df_filtrado.select_dtypes(include=['int64', 'float64']).columns.difference(['variavel_alvo'])

        st.subheader("Variáveis Categóricas")
        st.write("As seguintes variáveis são categóricas:")
        st.write(variaveis_categoricas)

        st.subheader("Variáveis Numéricas")
        st.write("As seguintes variáveis são numéricas:")
        st.write(variaveis_numericas)

        # Distribuição e Estatisticas básicas
        st.subheader("Distribuição da Variável Alvo")

        # Garantir que a variável alvo seja numérica
        df_filtrado[variavel_alvo] = pd.to_numeric(df_filtrado[variavel_alvo], errors='coerce')

        # Adicionar opção para gerar o gráfico de distribuição
        mostrar_distribuicao = st.checkbox("Mostrar Distribuição da Variável Alvo", value=False)

        if mostrar_distribuicao:
            plt.figure(figsize=(10, 6))
            sns.histplot(df_filtrado[variavel_alvo].dropna(), kde=True, color='blue')  # Remover NaN para o histograma
            plt.title(f"Distribuição da Variável Alvo: {variavel_alvo}")
            plt.xlabel(variavel_alvo)
            plt.ylabel("Frequência")
            st.pyplot(plt)

        # Existem valores extremos ou outliwes em variáveis numéricas?
        st.subheader("Existem valores extremos ou outliers em variáveis numéricas?")

        # Identificar valores extremos ou outliers em variáveis numéricas
        valores_extremos = df_filtrado[variaveis_numericas].apply(lambda x: x[(np.abs(x-x.mean())>=(3*x.std()))], axis=1)

        # Mostrar as variáveis com valores extremos ou outliers
        st.dataframe(valores_extremos)

        # Identificar quais variáveis têm mais valores extremos ou outliers
        contagem_valores_extremos = valores_extremos.sum().sort_values(ascending=False)
        st.subheader("Variáveis com Mais Valores Extremos ou Outliers")
        st.dataframe(contagem_valores_extremos)

        # Como as variáveis categóricas estão distribuídas?
        st.subheader("Distribuição das Variáveis Categóricas")

        mostrar_dist_categorias = st.checkbox("Mostrar Distribuição de Variáveis Categóricas", value=False)

        if mostrar_dist_categorias:
            # Criar um gráfico de barras para cada variável categórica
            # Vamos filtrar apenas algumas variáveis para verificar a distribuição delas
            variaveis_selecionadas = ['nome_cluster', 'ufcliente', 'gerentecarteira']
            for variavel in variaveis_selecionadas:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=variavel, data=df_filtrado)
                plt.title(f"Distribuição de {variavel}")
                plt.xlabel(variavel)
                plt.ylabel("Frequência")
                st.pyplot(plt)
        
    with tab4:    
        # Relacionamento e Padrões
        # Correção

        # Existe correlação entre as variáveis numéricas e a variável-alvo?
        st.subheader("Existe correlação entre as variáveis numéricas e a variável-alvo?")

        # st.write("Colunas disponíveis no DataFrame:", df_filtrado.columns.tolist())

        # Verificar se as variáveis numéricas e a variável alvo estão presentes no DataFrame
        variaveis_a_verificar = variaveis_numericas #+ [variavel_alvo]
        variaveis_presentes = [var for var in variaveis_a_verificar if var in df_filtrado.columns]

        if len(variaveis_presentes) == len(variaveis_a_verificar):
            # Calcular a matriz de correlação
            matriz_correlacao = df_filtrado[variaveis_presentes].corr()
            # Mostrar a matriz de correlação
            st.dataframe(matriz_correlacao)
        else:
            st.warning("Algumas variáveis não estão presentes no DataFrame. Verifique os nomes das colunas.")
        
        st.markdown("""
        **Correlação:**
        
        - Valor 1: Indica uma correlação positiva perfeita. Isso significa que, à medida que uma variável aumenta, a outra também aumenta de forma proporcional.
        - Valor -1: Indica uma correlação negativa perfeita. Isso significa que, à medida que uma variável aumenta, a outra diminui de forma proporcional.
        - Valor 0: Indica que não há correlação linear entre as variáveis. Isso significa que as mudanças em uma variável não estão relacionadas às mudanças na outra.
        """)


    with tab5:
        st.title("Fase de Pré-processamento")
        st.subheader("Preparação dos Dados para Análise")

        # Rafa comentou sobre não comparar mês a mês anos anteriores a 2023 com 2025 e considerar somente o crescimento anual
        # Ao contrário de 2024

        # Tratamento de dados ausentes na coluna 'rede'
        df_filtrado['rede'].fillna('sem rede', inplace=True)

        # Tratamento de dados ausentes nas colunas 'custo' e 'margem'
        df_filtrado['custo'].fillna(method='bfill', inplace=True)
        df_filtrado['margem'].fillna(method='bfill', inplace=True)

        df_processado = df_filtrado.copy()
        
        # Excluir registros com 'rede' igual a VILLA ou TOQUE DE PEIXE
        st.warning("Excluindo rede Villa e Toque de Peixe")
        # Verificar se df_filtrado foi definido antes de processamento
        if df_filtrado is not None:
            st.success("DataFrame filtrado carregado com sucesso.")
            df_processado = df_filtrado.copy()
            df_processado = df_processado[df_processado['rede'].isin(['VILLA', 'TOQUE DE PEIXE']) == False]
        else:
            st.error("DataFrame filtrado não foi definido. Verifique as etapas anteriores de processamento.")

        # Calcular o preço médio unitário vendido
        if df_processado is not None:
            df_processado['preco_medio_unitario'] = df_processado['faturamento'] / df_processado['quantidade']
            

        st.subheader("Tratamento de Outliers")


        # Vamos criar uma coluna cod_gerente_carteira apenas com o numeral encontrado à direta da dos valores da variável 
        df_processado['cod_gerente_carteira'] = df_processado['gerentecarteira'].apply(lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else None)
        # Fazer o mesmo para nome_cluster
        df_processado['cod_nome_cluster'] = df_processado['nome_cluster'].apply(lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else None)
        # Fazer o mesmo para supervisor_carteira
        df_processado['cod_supervisor_carteira'] = df_processado['supervisorcarteira'].apply(lambda x: re.search(r'\d+', x).group() if re.search(r'\d+', x) else None)


        # Tratar as variáveis cod_gerente_carteira, cod_nome_cluster e cod_supervisor_carteira como categóricas
        df_processado['cod_gerente_carteira'] = df_processado['cod_gerente_carteira'].astype(str)
        df_processado['cod_nome_cluster'] = df_processado['cod_nome_cluster'].astype(str)
        df_processado['cod_supervisor_carteira'] = df_processado['cod_supervisor_carteira'].astype(str)


        colunas_a_remover = ['cliente', 'nome_cluster', 'rede', 'gerentevenda', 'supervisorvenda', 'consultorvenda', 'gerentecarteira', 'supervisorcarteira', 'consultorcarteira', 'ufcliente', 'codigocliente', 'codigofilial', 'notasaida']
        for coluna in colunas_a_remover:
            if coluna in df_processado.columns:
                df_processado.drop(columns=[coluna], inplace=True)


        # Identificar as variáveis categóricas e numéricas, excluindo a variável alvo
        variaveis_categoricas_p = df_processado.select_dtypes(include=['object']).columns.difference([variavel_alvo])
        variaveis_numericas_p = df_processado.select_dtypes(include=['int64', 'float64']).columns.difference([variavel_alvo])


        # Aplicar escalas logarítmicas às variáveis numéricas
        for var in variaveis_numericas_p:
            df_processado[var] = df_processado[var].apply(lambda x: np.log(x) if x > 0 else x)

        st.success("Escalas logarítmicas aplicadas às variáveis numéricas com sucesso.")

        # Mostrar o DataFrame após o tratamento de outliers
        st.dataframe(df_processado.head())

        # Adicionar variáveis temporais
        if df_processado is not None:
            # Converter DATAMOVIMENTO para datetime se ainda não estiver
            df_processado['datamovimento'] = pd.to_datetime(df_processado['datamovimento'])
            
            # Criar novas variáveis temporais
            df_processado['dia_da_semana'] = df_processado['datamovimento'].dt.weekday  # Dia da semana
            df_processado['mes'] = df_processado['datamovimento'].dt.month  # Mês
            df_processado['sazonalidade'] = df_processado['datamovimento'].dt.month % 12 // 3 + 1  # Sazonalidade (1: Primavera, 2: Verão, 3: Outono, 4: Inverno)


            # Mostrar o DataFrame após a criação de variáveis temporais
            st.success("Criação de variáveis temporais com sucesso.")
            st.dataframe(df_processado.head())


    
        st.subheader("Fase de Modelagem e Treinamento do Modelo")
        st.write("Divisão dos dados")

        from sklearn.model_selection import train_test_split

        # Divisão dos dados em treino, validação e teste
        X = df_processado.drop(columns=[variavel_alvo])
        y = df_processado[variavel_alvo]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.143, random_state=42)

        st.success("Divisão dos dados em treino, validação e teste com sucesso.")
        # st.write("Índices de treino:", X_train.index)
        # st.write("Índices de validação:", X_val.index)
        # st.write("Índices de teste:", X_test.index)
        # Salvar df_processado na sessão
        st.session_state.df_processado = df_processado  # Salvar df_processado na sessão



if pagina == "Modelagem e Treinamento":
    if 'df_processado' in st.session_state:
        # Iniciar a fase de modelagem
        st.title("Fase de Modelagem e Treinamento do Modelo")
        st.subheader("Divisão dos dados")


        # Recuperar df_processado da sessão
        df_processado = st.session_state.df_processado
        # Remover registros do mês corrente de df_processado
        mes_atual = pd.Timestamp.now().to_period('M')
        df_processado = df_processado[df_processado['datamovimento'].dt.to_period('M') != mes_atual]
        
        st.success(f"Registros do mês atual ({mes_atual}) removidos do DataFrame.")

        st.subheader("Definição da Variável Alvo")
        variavel_alvo = st.selectbox("Selecione a Variável Alvo:", ["faturamento", "margem", "quantidade"])
        st.write(f"Variável Alvo Selecionada: {variavel_alvo}")

        from sklearn.model_selection import train_test_split

        # Divisão dos dados em treino, validação e teste
        X = df_processado.drop(columns=[variavel_alvo])
        y = df_processado[variavel_alvo]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.143, random_state=42)

        st.success("Divisão dos dados em treino, validação e teste com sucesso.")

        # Inclusão da seleção do modelo
        st.subheader("Escolha do Modelo")

        modelos_disponiveis = [
            'Nenhum',
            'Prophet',
            'Regressão Linear', 
            'Árvore de Decisão', 
            'Random Forest', 
            'Gradient Boosting', 
            'XGBoost'
        ]
        # Adicionar selectbox na sidebar para escolha do modelo
        modelo_selecionado = st.selectbox(
            "Escolha o Modelo de Machine Learning", 
            modelos_disponiveis
        )


    # Exibir o modelo selecionado
    st.write(f"Modelo selecionado: {modelo_selecionado}")

    if modelo_selecionado == 'Nenhum':
        st.subheader("Nenhum modelo selecionado")



    if modelo_selecionado == 'Prophet':
        # Preparando dados para o Prophet
        df_prophet = df_processado[['datamovimento', variavel_alvo]].copy()
        df_prophet.columns = ['ds', 'y']  # Renomeando colunas conforme requerido pelo Prophet
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        # Agregar dados diários para mensais
        df_prophet = df_prophet.resample('M', on='ds').sum().reset_index()

        # Adicionando feriados brasileiros usando make_holidays
        from prophet import Prophet
        from prophet.make_holidays import make_holidays_df
        from prophet.diagnostics import cross_validation, performance_metrics

        # Criar dataframe de feriados brasileiros
        feriados_brasileiros = make_holidays_df(year_list=[2018, 2019, 2020, 2021, 2022, 2023, 2024], country='BR')  # Adicione os anos desejados
        feriados_brasileiros['holiday'] = 'feriado'

        # Inicializando e treinando o modelo Prophet
        modelo_prophet = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            holidays=feriados_brasileiros,
            changepoint_prior_scale=0.2,  # Ajuste conforme necessário
            seasonality_prior_scale=10.0   # Ajuste conforme necessário
        )

        # Adicionando sazonalidade mensal
        # modelo_prophet.add_seasonality(name='mensal', period=30.5, fourier_order=5)

        # Ajustando o modelo
        modelo_prophet.fit(df_prophet)

        # Criando período futuro para previsão
        futuro = modelo_prophet.make_future_dataframe(periods=12, freq='ME')  # Previsão para os próximos 12 meses
        previsao = modelo_prophet.predict(futuro)

        # Visualizando resultados com Plotly
        import plotly.graph_objects as go

        fig = go.Figure()

        # Gráfico de linha para a previsão
        fig.add_trace(go.Scatter(x=previsao['ds'], y=previsao['yhat'], mode='lines', name='Previsão', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=previsao['ds'], y=previsao['yhat_lower'], mode='lines', name='Limite Inferior', line=dict(color='lightblue', dash='dash')))
        fig.add_trace(go.Scatter(x=previsao['ds'], y=previsao['yhat_upper'], mode='lines', name='Limite Superior', line=dict(color='lightblue', dash='dash')))
        
        # Adicionando dados reais
        fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='markers', name='Dados Reais', marker=dict(color='red')))

        # Atualizando layout
        fig.update_layout(title='Previsão com Prophet', xaxis_title='Data', yaxis_title=variavel_alvo)
        
        # Exibir gráfico interativo
        st.plotly_chart(fig)

        
        # Adicionar checkbox para validação cruzada
        aplicar_validacao_cruzada = st.checkbox("Aplicar Validação Cruzada", value=False)

        if aplicar_validacao_cruzada:
            # Aplicar validação cruzada
            df_cv = cross_validation(modelo_prophet, initial='365 days', period='30 days', horizon='90 days')

            # Calcular métricas de desempenho
            df_p = performance_metrics(df_cv)

            # Exibir as métricas de desempenho
            st.write("### Métricas de Desempenho da Validação Cruzada")
            st.dataframe(df_p)

            # Visualizar os resultados da validação cruzada
            fig_cv = modelo_prophet.plot(df_cv)
            st.pyplot(fig_cv)

        # Adicionar explicação sobre o Prophet e os resultados
        st.markdown(""" 
        ### Sobre o Prophet
        O Prophet é uma ferramenta de previsão desenvolvida pelo Facebook, projetada para lidar com séries temporais que apresentam padrões sazonais e tendências. Ele é especialmente útil para dados que têm períodos de sazonalidade diária, semanal ou anual, e pode lidar com dados ausentes e mudanças nas tendências.

        ### Resultados da Previsão
        A tabela abaixo apresenta as previsões mensais para o variável alvo ao longo dos anos. Cada célula representa a previsão do valor para um determinado mês e ano. A linha 'Total' fornece a soma das previsões para cada ano, permitindo uma visão geral do desempenho esperado ao longo do tempo.
        """)

        # Imprimir as colunas do DataFrame de previsão
        # st.subheader("Colunas do DataFrame de Previsão")
        # st.write(previsao.columns)

        # Criar tabela pivot com mês em linhas e ano em colunas
        previsao['ano'] = previsao['ds'].dt.year
        previsao['mes'] = previsao['ds'].dt.month
        tabela_pivot = previsao.pivot_table(index='mes', columns='ano', values='yhat', aggfunc='sum')

        # Adicionar linha de totais
        tabela_pivot.loc['Total'] = tabela_pivot.sum()

        # Exibir tabela pivot com formatação de duas casas decimais
        st.subheader("Tabela Pivot da Previsão")
        st.dataframe(tabela_pivot.style.format("{:.2f}"))

        # Componentes da previsão
        fig_componentes = modelo_prophet.plot_components(previsao)

        # Extrair os dados dos componentes
        trend = previsao[['ds', 'trend']]
        seasonal = previsao[['ds', 'weekly']]
        holidays = previsao[['ds', 'holidays']]
        yearly = previsao[['ds', 'yearly']]

        # Criar gráficos interativos com Plotly
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend['ds'], y=trend['trend'], mode='lines', name='Tendência', line=dict(color='blue')))
        fig_trend.update_layout(title='Tendência', xaxis_title='Data', yaxis_title='Valor')

        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Scatter(x=seasonal['ds'], y=seasonal['weekly'], mode='lines', name='Sazonalidade', line=dict(color='orange')))
        fig_seasonal.update_layout(title='Sazonalidade', xaxis_title='Data', yaxis_title='Valor')

        fig_yealy = go.Figure()
        fig_yealy.add_trace(go.Scatter(x=yearly['ds'], y=yearly['yearly'], mode='lines', name='Sazonalidade Anual', line=dict(color='green')))
        fig_yealy.update_layout(title='Sazonalidade Anual', xaxis_title='Data', yaxis_title='Valor')


        # Exibir gráficos interativos
        st.plotly_chart(fig_trend)
        st.plotly_chart(fig_seasonal)
        st.plotly_chart(fig_yealy)
        if 'holidays' in previsao.columns:
            fig_holidays = go.Figure()
            fig_holidays.add_trace(go.Scatter(x=holidays['ds'], y=holidays['holidays'], mode='lines', name='Feriados', line=dict(color='green')))
            fig_holidays.update_layout(title='Efeito dos Feriados', xaxis_title='Data', yaxis_title='Valor')
            
            # Exibir gráfico de feriados
            st.plotly_chart(fig_holidays)

        # Avaliação da acurácia
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

        # Calcular MSE para o horizonte desejado
        mse = mean_squared_error(df_prophet['y'], previsao['yhat'][:len(df_prophet)])
        # Calcular MAE (Mean Absolute Error)
        mae = mean_absolute_error(df_prophet['y'], previsao['yhat'][:len(df_prophet)])
        
        # Calcular MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(df_prophet['y'], previsao['yhat'][:len(df_prophet)])

        st.write(f"**Erro Quadrático Médio (MSE):** {mse:.2f}")
        st.write(f"**Erro Absoluto Médio (MSE):** {mae:.2f}")
        st.write(f"**Erro Absoluto Médio Percentual (MSE):** {mape:.2f}")

        # Explicação do Erro Quadrático Médio (MSE)
        st.markdown("### Interpretação do Erro Quadrático Médio (MSE)")
        
        if mse is not None:
            if mse < 0.1:
                st.success(f"O MSE de {mse:.2f} indica um modelo com excelente precisão. Quanto mais próximo de zero, melhor o desempenho do modelo.")
                st.markdown("""
                - **Interpretação:** Os valores previstos estão muito próximos dos valores reais
                - Baixa dispersão entre predições e valores observados
                - Alta confiabilidade nas previsões
                """)
            elif mse < 1:
                st.info(f"O MSE de {mse:.2f} sugere um modelo com boa precisão. Há uma variação moderada entre predições e valores reais.")
                st.markdown("""
                - **Interpretação:** Os valores previstos têm uma precisão razoável
                - Alguma dispersão entre predições e valores observados
                - Modelo funciona bem, mas pode ser aprimorado
                """)
            else:
                st.warning(f"O MSE de {mse:.2f} indica que o modelo tem espaço para melhorias significativas.")
                st.markdown("""
                - **Interpretação:** Existe uma diferença considerável entre valores previstos e reais
                - Alta dispersão nos resultados
                - Recomenda-se revisar features, técnicas de modelagem ou coletar mais dados
                """)

        # Análise de Resíduos para Regressão Linear
        st.subheader("Análise de Resíduos")
        
        # Calcular resíduos
        residuos = y_test - previsao
        
        # Gráfico de dispersão dos resíduos
        plt.figure(figsize=(10, 6))
        plt.scatter(previsao, residuos, color='green', alpha=0.7)
        plt.title("Gráfico de Dispersão dos Resíduos")
        plt.xlabel("Valores Previstos")
        plt.ylabel("Resíduos")
        plt.axhline(y=0, color='red', linestyle='--')
        st.pyplot(plt)
        
        # Histograma dos resíduos
        plt.figure(figsize=(10, 6))
        plt.hist(residuos, bins=30, color='purple', alpha=0.7)
        plt.title("Distribuição dos Resíduos")
        plt.xlabel("Resíduos")
        plt.ylabel("Frequência")
        st.pyplot(plt)
        
        # Explicação da análise de resíduos
        st.markdown("""
        ### Interpretação da Análise de Resíduos
        
        A análise de resíduos ajuda a avaliar a qualidade do modelo de regressão linear:
        
        1. **Gráfico de Dispersão dos Resíduos**:
        - Idealmente, os resíduos devem estar distribuídos aleatoriamente em torno da linha zero
        - Padrões ou tendências no gráfico indicam que o modelo pode estar deixando de capturar alguma relação importante
        
        2. **Histograma dos Resíduos**:
        - Os resíduos devem seguir aproximadamente uma distribuição normal
        - Distribuição simétrica em torno de zero sugere um bom ajuste do modelo
        
        #### O que procurar:
        - Resíduos concentrados próximos a zero
        - Distribuição aproximadamente simétrica 
        - Ausência de padrões sistemáticos
        
        #### Possíveis problemas:
        - Resíduos com padrão não aleatório: indica viés no modelo
        - Distribuição muito assimétrica: sugere que o modelo não captura bem a variação dos dados
        """)

    # Regressão Linear
    if modelo_selecionado == 'Regressão Linear':
        # Preparando dados para Regressão Linear
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns  # Importar seaborn para melhor visualização

        # Selecionar features para o modelo
        features = st.multiselect(
            "Selecione as Features para o Modelo de Regressão Linear", 
            list(df_processado.select_dtypes(include=['int64', 'float64']).columns)
        )

        # Preparar dados de treino e teste
        X = df_processado[features]
        y = df_processado[variavel_alvo]

        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar modelo de Regressão Linear
        modelo_linear = LinearRegression()
        modelo_linear.fit(X_train, y_train)

        # Fazer previsões
        previsoes = modelo_linear.predict(X_test)

        # Avaliar modelo
        mse = mean_squared_error(y_test, previsoes)
        r2 = r2_score(y_test, previsoes)

        # Visualizar resultados
        st.subheader("Resultados da Regressão Linear")
        
        # Gráfico de valores reais vs previstos
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=y_test, y=previsoes, color='blue', alpha=0.7)  # Usar seaborn para o gráfico de dispersão
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Linha de referência
        plt.title("Valores Reais vs Previstos", fontsize=16)  # Título com tamanho de fonte
        plt.xlabel("Valores Reais", fontsize=12)  # Rótulo do eixo x com tamanho de fonte
        plt.ylabel("Valores Previstos", fontsize=12)  # Rótulo do eixo y com tamanho de fonte
        plt.grid(True)  # Adicionar grade para melhor visualização
        st.pyplot(plt)

        # Métricas de desempenho
        st.write(f"**Erro Quadrático Médio (MSE):** {mse:.2f}")
        st.write(f"**Coeficiente de Determinação (R²):** {r2:.2f}")

        # Coeficientes do modelo
        coeficientes = pd.DataFrame({
            'Feature': features,
            'Coeficiente': modelo_linear.coef_
        })
        st.subheader("Coeficientes do Modelo")
        st.dataframe(coeficientes)

        # Previsão para 2025
        st.subheader("Previsão para 2025")
        
        # Criar DataFrame para os meses de 2025
        meses_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='M')
        df_previsao_2025 = pd.DataFrame(meses_2025, columns=['datamovimento'])
        
        # Adicionar colunas de características necessárias (exemplo)
        # Aqui você deve adicionar as colunas que seu modelo espera
        # Exemplo: df_previsao_2025['feature1'] = valor
        # df_previsao_2025['feature2'] = valor
        # Certifique-se de que as colunas correspondam às que foram usadas no treinamento do modelo

        # Exemplo de preenchimento de características (substitua com seus dados reais)
        df_previsao_2025[features] = 0  # Preencher com zeros para todas as features dinâmicas
        # Adicione mais características conforme necessário

        # Fazer previsões para 2025
        previsoes_2025 = modelo_linear.predict(df_previsao_2025[features])

        # Adicionar previsões ao DataFrame
        df_previsao_2025['Previsao'] = previsoes_2025

        # Exibir tabela de previsões
        st.dataframe(df_previsao_2025[['datamovimento', 'Previsao']].style.format({"Previsao": "${:,.2f}"}))



    # Arvore de decisão
    if modelo_selecionado == 'Árvore de Decisão':
        # Preparar dados para Árvore de Decisão
        
        features = st.multiselect(
            "Selecione as Features para o Modelo de Árvore de Decisão", 
            list(df_processado.select_dtypes(include=['int64', 'float64']).columns.difference([variavel_alvo]))
        )

        # Verificar se features foram selecionadas
        if not features:
            st.warning("Por favor, selecione pelo menos uma feature.")
        else:
            # Preparar dados de treino e teste
            X = df_processado[features]
            y = df_processado[variavel_alvo]

            # Dividir dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Treinar modelo de Árvore de Decisão
            arvore_decisao = DecisionTreeRegressor(random_state=42)
            arvore_decisao.fit(X_train, y_train)

            # Fazer previsões
            previsoes = arvore_decisao.predict(X_test)

            # Imprimir previsões em uma tabela
            previsoes_df = pd.DataFrame({
                'Valores Reais': y_test,
                'Valores Previstos': previsoes
            })
            st.subheader("Tabela de Previsões")
            st.dataframe(previsoes_df)

            # Avaliar modelo
            mse = mean_squared_error(y_test, previsoes)
            r2 = r2_score(y_test, previsoes)

            # Visualizar resultados
            st.subheader("Resultados da Árvore de Decisão")
            
            # Gráfico de valores reais vs previstos
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, previsoes, color='green', alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.title("Valores Reais vs Previstos (Árvore de Decisão)")
            plt.xlabel("Valores Reais")
            plt.ylabel("Valores Previstos")
            st.pyplot(plt)


            # Métricas de desempenho
            st.write(f"**Erro Quadrático Médio (MSE):** {mse:.2f}")
            st.write(f"**Coeficiente de Determinação (R²):** {r2:.2f}")

            # Importância das features
            importancia_features = pd.DataFrame({
                'Feature': features,
                'Importância': arvore_decisao.feature_importances_
            }).sort_values('Importância', ascending=False)
            
            st.subheader("Importância das Features")
            st.dataframe(importancia_features)

            
            # Visualização da árvore de decisão (opcional)
            st.subheader("Visualização da Árvore de Decisão")
            plt.figure(figsize=(20,10))
            plot_tree(arvore_decisao, feature_names=features, filled=True, rounded=True)
            plt.title("Estrutura da Árvore de Decisão")
            st.pyplot(plt)


    # Modelo XGBoost

    if modelo_selecionado == 'XGBoost':
        # Importar bibliotecas necessárias
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt
        import numpy as np

        # Preparar os dados para o XGBoost
        # Remover colunas não numéricas
        X = df_processado.drop(columns=[variavel_alvo, 'datamovimento', 'cod_gerente_carteira', 'cod_nome_cluster', 'cod_supervisor_carteira'])
        y = df_processado[variavel_alvo]

        # **Transformar dados diários em mensais**
        df_processado['datamovimento'] = df_processado['datamovimento'].dt.to_period('M').dt.to_timestamp()

        # **Adicionar lags de 3 e 6 meses**
        df_processado['lag_3'] = df_processado[variavel_alvo].shift(3)
        df_processado['lag_6'] = df_processado[variavel_alvo].shift(6)

        # **Adicionar média móvel das vendas**
        df_processado['media_movel_3'] = df_processado[variavel_alvo].rolling(window=3).mean()
        df_processado['media_movel_6'] = df_processado[variavel_alvo].rolling(window=6).mean()

        # **Adicionar indicadores sazonais**
        df_processado['mes'] = df_processado['datamovimento'].dt.month
        df_processado['sazonalidade'] = df_processado['mes'] % 12 // 3 + 1  # Sazonalidade (1: Primavera, 2: Verão, 3: Outono, 4: Inverno)

        # Se houver colunas categóricas, você pode usar one-hot encoding
        X = pd.get_dummies(X, drop_first=True)  # Converte colunas categóricas em variáveis dummy

        # Dividir os dados em treino, validação e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.143, random_state=42)

        # Inicializar e treinar o modelo XGBoost
        xgboost = XGBRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=42
        )
        xgboost.fit(X_train, y_train)

        # Fazer previsões
        previsoes = xgboost.predict(X_test)

        # Imprimir previsões em uma tabela
        previsoes_df = pd.DataFrame({
            'Valores Reais': y_test,
            'Valores Previstos': previsoes
        })
        st.subheader("Tabela de Previsões (XGBoost)")
        st.dataframe(previsoes_df)

        # Avaliar modelo
        mse = mean_squared_error(y_test, previsoes)
        r2 = r2_score(y_test, previsoes)

        # Visualizar resultados
        st.subheader("Resultados do XGBoost")
        
        # Gráfico de valores reais vs previstos
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, previsoes, color='purple', alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title("Valores Reais vs Previstos (XGBoost)")
        plt.xlabel("Valores Reais")
        plt.ylabel("Valores Previstos")
        st.pyplot(plt)

        # Métricas de desempenho
        st.write(f"**Erro Quadrático Médio (MSE):** {mse:.2f}")
        st.write(f"**Coeficiente de Determinação (R²):** {r2:.2f}")

        # Importância das features
        features = X.columns.tolist()
        importancia_features = pd.DataFrame({
            'Feature': features,
            'Importância': xgboost.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        st.subheader("Importância das Features (XGBoost)")
        st.dataframe(importancia_features)


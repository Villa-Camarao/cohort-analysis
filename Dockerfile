# Use uma imagem base oficial do Python. A versão 3.11 corresponde ao seu log de erro.
FROM python:3.11-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo de dependências PRIMEIRO.
# Isso aproveita o cache do Docker. Se o requirements.txt não mudar, esta camada não será reconstruída.
COPY requirements.txt .

# Instale as dependências. O --no-cache-dir garante que não haverá cache do pip.
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o resto do seu código para o diretório de trabalho
COPY . .

# Informe ao Docker que o contêiner escuta na porta 8501 (porta padrão do Streamlit)
EXPOSE 8501

# O comando para rodar sua aplicação quando o contêiner iniciar.
# O --server.address=0.0.0.0 é CRUCIAL para que o app seja acessível de fora do contêiner.
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

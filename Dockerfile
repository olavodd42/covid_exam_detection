FROM tensorflow/tensorflow:2.15.0-gpu

# Instala dependências adicionais
RUN apt-get update && apt-get install -y git wget unzip

# Cria diretório de trabalho
WORKDIR /app

# Copia arquivos para o container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#CMD ["python", "train.py"]
CMD ["python", "train_v2.py"]

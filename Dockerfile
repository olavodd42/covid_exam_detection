# Use a imagem oficial do TensorFlow com GPU
FROM tensorflow/tensorflow:2.15.0-gpu

# Metadados
LABEL maintainer="seu-email@exemplo.com"
LABEL description="COVID-19 X-Ray Classification Model Training"
LABEL version="2.0"

# Variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    curl \
    vim \
    htop \
    tree \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip e instala wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Cria usuário não-root para segurança
RUN useradd -m -u 1000 ml_user && \
    mkdir -p /app && \
    chown -R ml_user:ml_user /app

# Define diretório de trabalho
WORKDIR /app

# Copia requirements primeiro (para cache Docker)
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte
COPY . .

# Cria diretórios necessários
RUN mkdir -p logs results models plots && \
    chown -R ml_user:ml_user /app

# Muda para usuário não-root
USER ml_user

# Configuração de GPU para TensorFlow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import tensorflow as tf; print('TF version:', tf.__version__)"

# Volumes para dados persistentes
VOLUME ["/app/data", "/app/models", "/app/logs", "/app/results"]

# Expõe porta para TensorBoard (opcional)
EXPOSE 6006

# Comando padrão
CMD ["python", "train_improved.py"]

# Para desenvolvimento, use:
# CMD ["bash"]

# Para TensorBoard, adicione:
#CMD ["tensorboard", "--logdir=logs", "--host=0.0.0.0"]
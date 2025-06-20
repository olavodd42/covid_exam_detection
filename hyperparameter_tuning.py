"""
Script para otimização de hiperparâmetros usando Optuna
Permite busca automática dos melhores parâmetros para o modelo
"""

import optuna
import logging
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, VGG16
from tensorflow.keras.callbacks import EarlyStopping
import kagglehub

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Classe para otimização de hiperparâmetros"""
    
    def __init__(self, study_name: str = "covid_classification_optimization"):
        self.study_name = study_name
        self.train_ds = None
        self.val_ds = None
        self.class_names = None
        self.img_size = (224, 224)
        
        # Configuração de GPU
        self._setup_gpu()
        
        # Carrega dataset uma vez
        self._load_dataset()
    
    def _setup_gpu(self):
        """Configuração de GPU"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    
    def _load_dataset(self):
        """Carrega dataset para otimização"""
        logger.info("Carregando dataset...")
        path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
        dataset_dir = Path(f'{path}/COVID-19_Radiography_Dataset')
        
        # Usa batch size menor para otimização mais rápida
        batch_size = 32
        seed = 123
        
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=seed,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )
        
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=seed,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )
        
        self.class_names = self.train_ds.class_names
        
        # Otimização de performance
        autotune = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(autotune)
        self.val_ds = self.val_ds.cache().prefetch(autotune)
        
        logger.info(f"Dataset carregado. Classes: {self.class_names}")
    
    def create_model(self, trial) -> tf.keras.Model:
        """Cria modelo com hiperparâmetros sugeridos pelo trial"""
        
        # Hiperparâmetros para otimização
        model_type = trial.suggest_categorical('model_type', ['efficientnet', 'resnet', 'vgg'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        dense_units = trial.suggest_categorical('dense_units', [128, 256, 512, 1024])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Data augmentation parameters
        rotation_range = trial.suggest_float('rotation_range', 0.05, 0.3)
        zoom_range = trial.suggest_float('zoom_range', 0.05, 0.2)
        contrast_range = trial.suggest_float('contrast_range', 0.05, 0.3)
        
        # Arquitetura do modelo
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = layers.Rescaling(1./255)(inputs)
        
        # Data augmentation
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(rotation_range),
            layers.RandomZoom(zoom_range),
            layers.RandomContrast(contrast_range),
        ])
        x = augmentation(x)
        
        # Base model
        if model_type == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
        elif model_type == 'resnet':
            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
        else:  # vgg
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=x)
        
        base_model.trainable = False
        
        # Classifier head
        x = layers.GlobalAveragePooling2D()(base_model.output)
        
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Camada adicional (opcional)
        if trial.suggest_categorical('use_additional_layer', [True, False]):
            additional_units = trial.suggest_categorical('additional_units', [64, 128, 256])
            x = layers.Dense(additional_units, activation='relu')(x)
            x = layers.Dropout(dropout_rate * 0.5)(x)
        
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        return model
    
    def objective(self, trial):
        """Função objetivo para otimização"""
        
        # Hiperparâmetros de treinamento
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        
        # Cria modelo
        model = self.create_model(trial)
        
        # Configura otimizador
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            )
        else:  # rmsprop
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        # Compila modelo
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        # Treinamento rápido para otimização
        epochs = 15  # Reduzido para otimização mais rápida
        
        try:
            history = model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Retorna melhor acurácia de validação
            best_accuracy = max(history.history['val_accuracy'])
            
            # Limpeza de memória
            del model
            tf.keras.backend.clear_session()
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Erro durante treinamento: {e}")
            return 0.0
    
    def optimize(self, n_trials: int = 50, timeout: int = None):
        """
        Executa otimização de hiperparâmetros
        
        Args:
            n_trials: Número de trials para executar
            timeout: Timeout em segundos (opcional)
        """
        
        # Cria estudo
        study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            storage=f'sqlite:///{self.study_name}.db',
            load_if_exists=True
        )
        
        logger.info(f"Iniciando otimização com {n_trials} trials...")
        
        # Executa otimização
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._trial_callback]
        )
        
        # Salva resultados
        self._save_results(study)
        
        return study
    
    def _trial_callback(self, study, trial):
        """Callback executado após cada trial"""
        logger.info(
            f"Trial {trial.number}: "
            f"Accuracy = {trial.value:.4f} "
            f"(Best: {study.best_value:.4f})"
        )
    
    def _save_results(self, study):
        """Salva resultados da otimização"""
        
        # Melhores parâmetros
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Melhor acurácia: {best_value:.4f}")
        logger.info(f"Melhores parâmetros: {best_params}")
        
        # Salva em arquivo JSON
        results = {
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'study_name': self.study_name
        }
        
        with open(f'{self.study_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Salva trials completos
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            }
            trials_data.append(trial_data)
        
        with open(f'{self.study_name}_trials.json', 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        logger.info(f"Resultados salvos em {self.study_name}_results.json")
    
    def create_optimized_model(self, best_params: Dict[str, Any]) -> tf.keras.Model:
        """
        Cria modelo final com os melhores parâmetros encontrados
        
        Args:
            best_params: Dicionário com os melhores parâmetros
            
        Returns:
            Modelo TensorFlow otimizado
        """
        
        # Cria um trial mock com os melhores parâmetros
        class MockTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_categorical(self, name, choices):
                return self.params[name]
            
            def suggest_float(self, name, low, high, log=False):
                return self.params[name]
        
        mock_trial = MockTrial(best_params)
        model = self.create_model(mock_trial)
        
        # Configura otimizador
        optimizer_type = best_params['optimizer']
        learning_rate = best_params['learning_rate']
        
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'adamw':
            weight_decay = best_params.get('weight_decay', 1e-4)
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model


def analyze_optimization_results(study_name: str):
    """Analisa e visualiza resultados da otimização"""
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Carrega resultados
        with open(f'{study_name}_trials.json', 'r') as f:
            trials_data = json.load(f)
        
        df = pd.DataFrame(trials_data)
        completed_trials = df[df['state'] == 'COMPLETE']
        
        if len(completed_trials) == 0:
            logger.warning("Nenhum trial completo encontrado")
            return
        
        # Gráfico de convergência
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(completed_trials['number'], completed_trials['value'])
        plt.xlabel('Trial')
        plt.ylabel('Validation Accuracy')
        plt.title('Convergência da Otimização')
        plt.grid(True)
        
        # Histograma dos resultados
        plt.subplot(1, 2, 2)
        plt.hist(completed_trials['value'], bins=20, alpha=0.7)
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Frequency')
        plt.title('Distribuição dos Resultados')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{study_name}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análise de importância dos parâmetros
        logger.info("Top 10 trials:")
        top_trials = completed_trials.nlargest(10, 'value')
        for _, trial in top_trials.iterrows():
            logger.info(f"Trial {trial['number']}: {trial['value']:.4f}")
        
    except ImportError:
        logger.warning("Matplotlib/Pandas não disponível para análise visual")
    except Exception as e:
        logger.error(f"Erro na análise: {e}")


def main():
    """Função principal para execução da otimização"""
    
    # Configurações
    study_name = "covid_classification_optimization"
    n_trials = 30  # Ajuste conforme necessário
    
    # Executa otimização
    tuner = HyperparameterTuner(study_name)
    study = tuner.optimize(n_trials=n_trials)
    
    # Analisa resultados
    analyze_optimization_results(study_name)
    
    # Cria modelo final com melhores parâmetros
    best_params = study.best_params
    final_model = tuner.create_optimized_model(best_params)
    
    logger.info("Modelo otimizado criado com sucesso!")
    logger.info(f"Use os parâmetros salvos em {study_name}_results.json para treinamento final")


if __name__ == "__main__":
    main()
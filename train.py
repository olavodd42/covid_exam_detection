"""
Treinamento melhorado para classificação de COVID-19 em imagens de radiografia
Inclui logging, configurações flexíveis, validação robusta e métricas avançadas
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score,
    precision_recall_curve,
    roc_curve
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.mixed_precision import set_global_policy
import kagglehub

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CovidClassifier:
    """Classe principal para treinamento do classificador de COVID-19"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None
        self.class_names = None
        
        # Configuração de GPU otimizada
        self._setup_gpu()
        
        # Mixed precision para melhor performance
        if config.get('use_mixed_precision', True):
            set_global_policy('mixed_float16')
            logger.info("Mixed precision habilitada")
    
    def _setup_gpu(self):
        """Configuração otimizada de GPU"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configurada: {len(gpus)} dispositivos encontrados")
            except RuntimeError as e:
                logger.error(f"Erro na configuração de GPU: {e}")
        else:
            logger.warning("Nenhuma GPU encontrada, usando CPU")
    
    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Carrega e prepara o dataset com split train/val/test"""
        logger.info("Baixando dataset...")
        path = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
        dataset_dir = Path(f'{path}/COVID-19_Radiography_Dataset')
        
        logger.info(f"Dataset localizado em: {dataset_dir}")
        
        # Primeiro split: 80% treino, 20% temp
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="training",
            seed=self.config['seed'],
            image_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            label_mode="categorical"
        )
        
        temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.2,
            subset="validation",
            seed=self.config['seed'],
            image_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            label_mode="categorical"
        )
        
        # Segundo split: divide temp_ds em 50% val, 50% test
        val_batches = tf.data.experimental.cardinality(temp_ds)
        test_ds = temp_ds.take(val_batches // 2)
        val_ds = temp_ds.skip(val_batches // 2)
        
        # Salva nomes das classes
        self.class_names = train_ds.class_names
        logger.info(f"Classes encontradas: {self.class_names}")
        
        # Otimização de performance
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(autotune)
        val_ds = val_ds.cache().prefetch(autotune)
        test_ds = test_ds.cache().prefetch(autotune)
        
        return train_ds, val_ds, test_ds
    
    def create_data_augmentation(self) -> tf.keras.Sequential:
        """Cria pipeline de data augmentation balanceado"""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ], name="data_augmentation")
    
    def create_model(self, model_type: str = "efficientnet") -> tf.keras.Model:
        """Cria modelo com arquitetura especificada"""
        input_shape = (*self.config['img_size'], 3)
        inputs = tf.keras.Input(shape=input_shape)
        
        # Normalização e augmentation
        x = layers.Rescaling(1./255)(inputs)
        
        if self.config.get('use_augmentation', True):
            augmentation = self.create_data_augmentation()
            x = augmentation(x)
        
        # Backbone model
        if model_type == "efficientnet":
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif model_type == "resnet":
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif model_type == "vgg":
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        else:
            # Modelo CNN customizado
            return self._create_custom_cnn(inputs)
        
        # Congela layers inicialmente
        base_model.trainable = False
        
        # Cabeça do classificador
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Camada de saída
        num_classes = len(self.class_names) if self.class_names else 4
        outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def _create_custom_cnn(self, inputs) -> tf.keras.Model:
        """Cria CNN customizada melhorada"""
        x = layers.Rescaling(1./255)(inputs)
        
        if self.config.get('use_augmentation', True):
            x = self.create_data_augmentation()(x)
        
        # Blocos convolucionais com BatchNorm
        for filters in [32, 64, 128, 256]:
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)
        
        # Classificador
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        num_classes = len(self.class_names) if self.class_names else 4
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def compile_model(self, model: tf.keras.Model):
        """Compila modelo com configurações otimizadas"""
        initial_lr = self.config.get('learning_rate', 1e-4)
        
        # Scheduler de learning rate
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=1.0,
            alpha=0.1
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Modelo compilado com sucesso")
        return model
    
    def get_callbacks(self) -> list:
        """Retorna lista de callbacks otimizados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f'best_model_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            callbacks.CSVLogger(f'training_log_{timestamp}.csv'),
            # TensorBoard para visualização
            callbacks.TensorBoard(
                log_dir=f'logs/{timestamp}',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train(self, train_ds, val_ds):
        """Executa treinamento completo"""
        logger.info("Iniciando treinamento...")
        
        # Cria e compila modelo
        model_type = self.config.get('model_type', 'efficientnet')
        self.model = self.create_model(model_type)
        self.model = self.compile_model(self.model)
        
        logger.info(f"Modelo {model_type} criado")
        logger.info(f"Parâmetros treináveis: {self.model.count_params():,}")
        
        # Treinamento
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.get('epochs', 50),
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        logger.info("Treinamento concluído")
        return self.history
    
    def fine_tune(self, train_ds, val_ds, unfreeze_layers: int = 20):
        """Fine-tuning do modelo pré-treinado"""
        if not self.model:
            raise ValueError("Modelo deve ser treinado antes do fine-tuning")
        
        logger.info(f"Iniciando fine-tuning (unfreeze_layers={unfreeze_layers})")
        
        # Descongela as últimas camadas
        base_model = self.model.layers[2]  # Assume que é o 3º layer
        base_model.trainable = True
        
        # Congela as primeiras camadas
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompila com learning rate menor
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tuning
        fine_tune_history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.get('fine_tune_epochs', 10),
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        # Combina históricos
        if self.history:
            for key in self.history.history:
                self.history.history[key].extend(fine_tune_history.history[key])
        
        logger.info("Fine-tuning concluído")
    
    def evaluate_model(self, test_ds) -> Dict[str, Any]:
        """Avaliação completa do modelo"""
        logger.info("Avaliando modelo...")
        
        # Avaliação básica
        test_results = self.model.evaluate(test_ds, verbose=0)
        metrics = dict(zip(self.model.metrics_names, test_results))
        
        # Predições para métricas avançadas
        y_pred_probs = []
        y_true = []
        
        for images, labels in test_ds:
            probs = self.model.predict(images, verbose=0)
            y_pred_probs.extend(probs)
            y_true.extend(labels.numpy())
        
        y_pred_probs = np.array(y_pred_probs)
        y_true = np.array(y_true)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        
        # Métricas detalhadas
        report = classification_report(
            y_true_labels, 
            y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # AUC para cada classe (One-vs-Rest)
        auc_scores = {}
        if len(self.class_names) > 2:
            try:
                auc_scores = roc_auc_score(
                    y_true, y_pred_probs, 
                    multi_class='ovr', 
                    average=None
                )
                auc_scores = dict(zip(self.class_names, auc_scores))
            except ValueError:
                logger.warning("Não foi possível calcular AUC scores")
        
        results = {
            'test_metrics': metrics,
            'classification_report': report,
            'auc_scores': auc_scores,
            'predictions': {
                'y_true': y_true_labels.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_probs': y_pred_probs.tolist()
            }
        }
        
        logger.info(f"Acurácia no teste: {metrics.get('accuracy', 0):.4f}")
        return results
    
    def plot_results(self, results: Dict[str, Any]):
        """Gera visualizações dos resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Training history
        if self.history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Accuracy
            axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            
            # Precision
            if 'precision' in self.history.history:
                axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
                axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
                axes[1, 0].set_title('Model Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].legend()
            
            # Recall
            if 'recall' in self.history.history:
                axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
                axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
                axes[1, 1].set_title('Model Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(f'training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Confusion Matrix
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred']
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Classification Report Heatmap
        report = results['classification_report']
        metrics_data = []
        classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        for class_name in classes:
            metrics_data.append([
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics_data,
            annot=True,
            fmt='.3f',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=classes,
            cmap='RdYlBu_r'
        )
        plt.title('Classification Metrics by Class')
        plt.savefig(f'classification_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizações salvas com timestamp {timestamp}")
    
    def save_results(self, results: Dict[str, Any]):
        """Salva resultados em arquivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Remove arrays numpy que não são JSON serializáveis
        clean_results = {
            'timestamp': timestamp,
            'config': self.config,
            'test_metrics': results['test_metrics'],
            'classification_report': results['classification_report'],
            'auc_scores': results['auc_scores']
        }
        
        with open(f'results_{timestamp}.json', 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Resultados salvos em results_{timestamp}.json")


def main():
    """Função principal de treinamento"""
    # Configurações do experimento
    config = {
        'img_size': (224, 224),
        'batch_size': 16,
        'epochs': 50,
        'fine_tune_epochs': 10,
        'learning_rate': 1e-4,
        'seed': 123,
        'model_type': 'efficientnet',  # 'efficientnet', 'resnet', 'vgg', 'custom'
        'use_augmentation': True,
        'use_mixed_precision': True
    }
    
    # Cria classificador
    classifier = CovidClassifier(config)
    
    try:
        # Carrega dados
        train_ds, val_ds, test_ds = classifier.load_dataset()
        
        # Treinamento inicial
        classifier.train(train_ds, val_ds)
        
        # Fine-tuning (opcional)
        if config['model_type'] in ['efficientnet', 'resnet', 'vgg']:
            classifier.fine_tune(train_ds, val_ds)
        
        # Avaliação final
        results = classifier.evaluate_model(test_ds)
        
        # Visualizações e salvamento
        classifier.plot_results(results)
        classifier.save_results(results)
        
        # Salva modelo final
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classifier.model.save(f'covid_model_final_{timestamp}.h5')
        
        logger.info("Treinamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        raise


if __name__ == "__main__":
    main()
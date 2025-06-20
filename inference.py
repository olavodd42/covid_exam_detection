"""
Script de inferência para o modelo de classificação COVID-19
Permite fazer predições em imagens individuais ou lotes
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidPredictor:
    """Classe para fazer predições de COVID-19 em imagens de raio-X"""
    
    def __init__(self, model_path: str, class_names: List[str] = None):
        """
        Inicializa o preditor
        
        Args:
            model_path: Caminho para o modelo treinado (.h5)
            class_names: Lista com nomes das classes
        """
        self.model_path = model_path
        self.class_names = class_names or ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.model = None
        self.img_size = (224, 224)
        
        self._load_model()
        logger.info(f"Modelo carregado: {model_path}")
        logger.info(f"Classes: {self.class_names}")
    
    def _load_model(self):
        """Carrega o modelo treinado"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def preprocess_image(self, img_path: Union[str, Path]) -> np.ndarray:
        """
        Pré-processa uma imagem para inferência
        
        Args:
            img_path: Caminho para a imagem
            
        Returns:
            Array numpy com a imagem processada
        """
        try:
            # Carrega e redimensiona a imagem
            img = image.load_img(img_path, target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalização (importante: deve ser a mesma do treinamento)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem {img_path}: {e}")
            raise
    
    def predict_single(self, img_path: Union[str, Path]) -> Dict[str, Union[str, float, Dict]]:
        """
        Faz predição para uma única imagem
        
        Args:
            img_path: Caminho para a imagem
            
        Returns:
            Dicionário com resultados da predição
        """
        # Pré-processa imagem
        img_array = self.preprocess_image(img_path)
        
        # Faz predição
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Organiza resultados
        result = {
            'image_path': str(img_path),
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': confidence,
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, predictions[0])
            }
        }
        
        return result
    
    def predict_batch(self, img_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Faz predições para um lote de imagens
        
        Args:
            img_paths: Lista de caminhos para as imagens
            
        Returns:
            Lista com resultados das predições
        """
        results = []
        
        logger.info(f"Processando {len(img_paths)} imagens...")
        
        for img_path in img_paths:
            try:
                result = self.predict_single(img_path)
                results.append(result)
                logger.info(f"✓ {Path(img_path).name}: {result['predicted_class']} ({result['confidence']:.3f})")
            except Exception as e:
                logger.error(f"✗ Erro em {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory: Union[str, Path], 
                         extensions: List[str] = None) -> List[Dict]:
        """
        Faz predições para todas as imagens em um diretório
        
        Args:
            directory: Caminho para o diretório
            extensions: Extensões de arquivo aceitas
            
        Returns:
            Lista com resultados das predições
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        directory = Path(directory)
        img_paths = []
        
        for ext in extensions:
            img_paths.extend(directory.glob(f"*{ext}"))
            img_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        if not img_paths:
            logger.warning(f"Nenhuma imagem encontrada em {directory}")
            return []
        
        return self.predict_batch(img_paths)
    
    def visualize_prediction(self, img_path: Union[str, Path], 
                           save_path: str = None) -> None:
        """
        Visualiza a predição com a imagem original
        
        Args:
            img_path: Caminho para a imagem
            save_path: Caminho para salvar a visualização (opcional)
        """
        # Faz predição
        result = self.predict_single(img_path)
        
        # Carrega imagem original
        img = Image.open(img_path)
        
        # Cria visualização
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Mostra imagem original
        ax1.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax1.set_title(f"Imagem Original\n{Path(img_path).name}")
        ax1.axis('off')
        
        # Mostra probabilidades
        classes = list(result['all_probabilities'].keys())
        probs = list(result['all_probabilities'].values())
        
        bars = ax2.barh(classes, probs)
        ax2.set_xlabel('Probabilidade')
        ax2.set_title(f'Predição: {result["predicted_class"]}\n'
                     f'Confiança: {result["confidence"]:.3f}')
        
        # Destaca a classe predita
        predicted_idx = classes.index(result['predicted_class'])
        bars[predicted_idx].set_color('red')
        
        # Adiciona valores nas barras
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.01, i, f'{prob:.3f}', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, results: List[Dict], 
                       output_path: str = "prediction_report.txt") -> None:
        """
        Gera relatório das predições
        
        Args:
            results: Lista com resultados das predições
            output_path: Caminho para salvar o relatório
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE PREDIÇÕES - COVID-19 X-RAY CLASSIFIER\n")
            f.write("=" * 60 + "\n\n")
            
            # Estatísticas gerais
            total_images = len(results)
            successful_predictions = len([r for r in results if 'error' not in r])
            error_count = total_images - successful_predictions
            
            f.write(f"Total de imagens processadas: {total_images}\n")
            f.write(f"Predições bem-sucedidas: {successful_predictions}\n")
            f.write(f"Erros: {error_count}\n\n")
            
            # Distribuição de classes
            if successful_predictions > 0:
                class_counts = {}
                for result in results:
                    if 'error' not in result:
                        pred_class = result['predicted_class']
                        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                
                f.write("DISTRIBUIÇÃO DE CLASSES PREDITAS:\n")
                f.write("-" * 30 + "\n")
                for class_name, count in class_counts.items():
                    percentage = (count / successful_predictions) * 100
                    f.write(f"{class_name}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Detalhes das predições
            f.write("DETALHES DAS PREDIÇÕES:\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {Path(result['image_path']).name}\n")
                
                if 'error' in result:
                    f.write(f"   ERRO: {result['error']}\n")
                else:
                    f.write(f"   Classe: {result['predicted_class']}\n")
                    f.write(f"   Confiança: {result['confidence']:.3f}\n")
                    f.write("   Probabilidades:\n")
                    for class_name, prob in result['all_probabilities'].items():
                        f.write(f"     {class_name}: {prob:.3f}\n")
                f.write("\n")
        
        logger.info(f"Relatório salvo em: {output_path}")


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(
        description='Inferência para classificação de COVID-19 em raios-X'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Caminho para o modelo treinado (.h5)'
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Caminho para imagem ou diretório de imagens'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        default='predictions',
        help='Diretório para salvar resultados'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Gerar visualizações das predições'
    )
    
    parser.add_argument(
        '--classes',
        nargs='+',
        default=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'],
        help='Nomes das classes'
    )
    
    args = parser.parse_args()
    
    # Cria diretório de saída
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Inicializa preditor
    predictor = CovidPredictor(args.model, args.classes)
    
    # Verifica se input é arquivo ou diretório
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Predição para uma única imagem
        logger.info(f"Processando imagem: {input_path}")
        result = predictor.predict_single(input_path)
        results = [result]
        
        # Visualização (se solicitada)
        if args.visualize:
            vis_path = output_dir / f"prediction_{input_path.stem}.png"
            predictor.visualize_prediction(input_path, vis_path)
            
    elif input_path.is_dir():
        # Predição para diretório
        logger.info(f"Processando diretório: {input_path}")
        results = predictor.predict_directory(input_path)
        
        # Visualizações (se solicitadas)
        if args.visualize:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            for result in results:
                if 'error' not in result:
                    img_path = result['image_path']
                    vis_path = vis_dir / f"prediction_{Path(img_path).stem}.png"
                    predictor.visualize_prediction(img_path, vis_path)
    else:
        logger.error(f"Caminho inválido: {input_path}")
        return
    
    # Gera relatório
    report_path = output_dir / "prediction_report.txt"
    predictor.generate_report(results, report_path)
    
    # Salva resultados em JSON
    import json
    json_path = output_dir / "predictions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processamento concluído. Resultados salvos em: {output_dir}")
    
    # Mostra resumo
    successful = len([r for r in results if 'error' not in r])
    total = len(results)
    logger.info(f"Resumo: {successful}/{total} predições bem-sucedidas")


if __name__ == "__main__":
    main()
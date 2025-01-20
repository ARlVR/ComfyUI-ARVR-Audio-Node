import os
import torch
import torchaudio
import logging
from typing import Tuple, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class AudioManagerNode:
    """Node para gerenciar áudios gerados no ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {
                    "default": "arvr_audio",
                    "multiline": False
                }),
                "format": (["wav", "mp3", "ogg", "flac"], {
                    "default": "wav",
                    "description": "Formato do arquivo de áudio"
                }),
                "output_dir": ("STRING", {
                    "default": "outputs/arvr_audio",
                    "description": "Diretório para salvar o áudio"
                }),
                "sample_rate": ("INT", {
                    "default": 44100,
                    "min": 8000,
                    "max": 48000,
                    "step": 100,
                    "description": "Taxa de amostragem do áudio"
                }),
                "preview_audio": ("BOOLEAN", {
                    "default": True,
                    "description": "Reproduzir áudio após geração"
                }),
            },
            "optional": {
                "metadata": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "Metadados do áudio (formato JSON)"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "process_audio"
    CATEGORY = "audio"
    OUTPUT_NODE = True

    def process_audio(self, audio, filename, format, output_dir, sample_rate, preview_audio, metadata=""):
        try:
            # Debug: imprimir informações sobre o áudio recebido
            print(f"Tipo do áudio recebido: {type(audio)}")
            if isinstance(audio, dict):
                print(f"Chaves do dicionário: {audio.keys()}")
            
            # Criar diretório de saída se não existir
            os.makedirs(output_dir, exist_ok=True)
            
            # Gerar caminho completo do arquivo
            file_path = os.path.join(output_dir, f"{filename}.{format}")
            
            # Extrair o tensor de áudio e a taxa de amostragem do dicionário
            if isinstance(audio, dict):
                if 'waveform' in audio:
                    audio_data = audio['waveform']
                    # Usar a taxa de amostragem do áudio se disponível
                    if 'sample_rate' in audio:
                        sample_rate = audio['sample_rate']
                else:
                    raise ValueError(f"Formato de áudio não reconhecido. Chaves disponíveis: {list(audio.keys())}")
                
                if isinstance(audio_data, (list, np.ndarray)):
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                elif isinstance(audio_data, torch.Tensor):
                    audio_tensor = audio_data
                else:
                    raise ValueError(f"Formato de áudio não suportado: {type(audio_data)}")
            else:
                audio_tensor = audio
            
            # Converter o tensor de áudio para 2D se necessário
            if audio_tensor.ndim == 3:
                # Remove a dimensão do batch (primeira dimensão)
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.ndim == 1:
                # Adiciona dimensão do canal se for 1D
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Garantir que o áudio está no formato correto (2D)
            if audio_tensor.ndim != 2:
                raise ValueError(f"Formato de áudio inválido: esperado 2D, recebido {audio_tensor.ndim}D")
            
            # Normalizar o áudio se necessário
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()
            
            # Garantir que o áudio está entre -1 e 1
            if audio_tensor.abs().max() > 1:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Salvar o áudio
            torchaudio.save(
                file_path,
                audio_tensor,
                sample_rate,
                format=format
            )
            
            logger.info(f"Áudio salvo em: {file_path}")
            
            # Adicionar metadados se fornecidos
            if metadata:
                try:
                    from mutagen import File
                    audio_file = File(file_path)
                    metadata_dict = eval(metadata)
                    if audio_file is not None:
                        for key, value in metadata_dict.items():
                            if hasattr(audio_file.tags, 'add'):
                                audio_file.tags.add(key, str(value))
                            else:
                                audio_file[key] = str(value)
                        audio_file.save()
                except ImportError:
                    logger.warning("mutagen não instalado. Metadados não foram salvos.")
                except Exception as e:
                    logger.error(f"Erro ao salvar metadados: {str(e)}")
            
            # Preview do áudio se solicitado
            if preview_audio:
                try:
                    import sounddevice as sd
                    audio_numpy = audio_tensor.cpu().numpy()
                    sd.play(audio_numpy.T, sample_rate)
                except Exception as e:
                    print(f"Erro ao reproduzir áudio: {str(e)}")
            
            return (file_path,)
            
        except Exception as e:
            print(f"Erro no processamento do áudio: {str(e)}")
            raise e

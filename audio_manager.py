import os
import torch
import torchaudio
import logging
from typing import Tuple, Dict, Any

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

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "process_audio"
    CATEGORY = "ARVR Audio"

    def process_audio(self, audio: Dict[str, torch.Tensor], filename: str, 
                     format: str, output_dir: str, sample_rate: int,
                     preview_audio: bool, metadata: str = "") -> Tuple[Dict[str, Any], str]:
        try:
            # Criar diretório se não existir
            os.makedirs(output_dir, exist_ok=True)
            
            # Garantir extensão correta
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"
            
            # Caminho completo do arquivo
            file_path = os.path.join(output_dir, filename)
            
            # Processar áudio
            waveform = audio["waveform"]
            original_sr = audio.get("sample_rate", 44100)
            
            # Resample se necessário
            if original_sr != sample_rate:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=original_sr,
                    new_freq=sample_rate
                )(waveform)
            
            # Salvar áudio
            torchaudio.save(
                file_path,
                waveform,
                sample_rate,
                format=format
            )
            
            logger.info(f"Áudio salvo em: {file_path}")
            
            # Adicionar metadados se fornecidos
            if metadata:
                try:
                    import taglib
                    audio_file = taglib.File(file_path)
                    metadata_dict = eval(metadata)
                    for key, value in metadata_dict.items():
                        audio_file.tags[key] = [str(value)]
                    audio_file.save()
                except ImportError:
                    logger.warning("taglib não instalado. Metadados não foram salvos.")
                except Exception as e:
                    logger.error(f"Erro ao salvar metadados: {str(e)}")
            
            # Reproduzir áudio se solicitado
            if preview_audio:
                try:
                    import sounddevice as sd
                    waveform_numpy = waveform.numpy()
                    sd.play(waveform_numpy.T, sample_rate)
                    sd.wait()
                except ImportError:
                    logger.warning("sounddevice não instalado. Prévia de áudio não disponível.")
                except Exception as e:
                    logger.error(f"Erro ao reproduzir áudio: {str(e)}")
            
            return (audio, file_path)
            
        except Exception as e:
            logger.error(f"Erro no processamento do áudio: {str(e)}")
            raise

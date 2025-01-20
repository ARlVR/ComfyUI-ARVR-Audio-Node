"""
@title: ComfyUI ARVR Audio Node
@author: ARVR
@description: Node para gerenciamento de áudio no ComfyUI
"""

from .audio_manager import AudioManagerNode

NODE_CLASS_MAPPINGS = {
    "ARVRAudioManagerNode": AudioManagerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ARVRAudioManagerNode": "ARVR Audio Manager"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
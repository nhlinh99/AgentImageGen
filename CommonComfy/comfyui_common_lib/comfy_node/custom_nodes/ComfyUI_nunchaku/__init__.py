"""Vendored subset of ComfyUI-nunchaku (https://github.com/nunchaku-tech/ComfyUI-nunchaku).

Only the Qwen-Image path is copied: ``nodes/models/qwenimage.py`` and the
``models`` / ``model_base`` / ``model_configs`` / ``mixins`` / ``model_patcher``
modules it pulls in. The upstream Flux, PuLID, IPAdapter, text-encoder,
depth-preprocessor and installer nodes are deliberately not vendored -- nothing
in this project calls them and they drag in insightface/facexlib/onnxruntime.

Upstream's ``__init__`` also runs a nunchaku version check and imports every node
at module scope. That is dropped here: this package is imported by ComfyUI's
``load_custom_node`` at worker start, where an ImportError would take the whole
scan down. The node is registered lazily-safely below instead, and the
device-explicit wrapper the workers actually use lives in
``custom_nodes/custom_modules/nunchaku/qwenimage.py``.

Sync note: when refreshing against upstream, keep the file list above in step and
re-check ``nodes/models/qwenimage.py`` against ``comfy.model_detection`` /
``comfy.model_management``, which move between ComfyUI releases.
"""

import logging

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}

try:
    from .nodes.models.qwenimage import NunchakuQwenImageDiTLoader

    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader
except ImportError as e:
    # nunchaku itself is a CUDA-only wheel installed from libs/; on a box without it
    # the rest of the node scan must still succeed.
    logger.warning("Node `NunchakuQwenImageDiTLoader` import failed: %s", e)

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

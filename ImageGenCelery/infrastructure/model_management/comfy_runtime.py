"""ComfyUI runtime flags that must be set before anything imports ``comfy``.

``comfy/ldm/modules/attention.py`` picks its attention implementation at *module
import* time from ``comfy.cli_args.args``, and nothing in this service calls
``comfy.options.enable_args_parsing()`` -- so ``args`` comes from
``parser.parse_args([])``, i.e. pure defaults. The SageAttention kernels shipped in
``libs/`` are therefore registered but never selected, and the whole stack runs on
``attention_pytorch``.

Flipping that has to happen before comfy's attention module is first imported,
which is why this lives on its own (no comfy imports at module scope) and is called
from ``celery_app/__init__.py`` -- every worker's first import -- as well as from
``infrastructure/model_management/models.py`` for entrypoints that skip celery_app.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

_ATTENTION_MODULE = "comfy.ldm.modules.attention"

_applied = False


def _comfy_node_dir() -> str:
    """comfy_node's own directory, which has to be on sys.path for bare `comfy.*` imports."""
    import comfyui_common_lib.comfy_node as _comfy_node_pkg

    return os.path.dirname(_comfy_node_pkg.__file__)


def apply_comfy_runtime_args(use_sage_attention: bool = None) -> None:
    """Set comfy's global args. Idempotent; safe to call from several entrypoints.

    ``use_sage_attention`` defaults to Config.use_sage_attention.
    """
    global _applied
    if _applied:
        return
    _applied = True

    if use_sage_attention is None:
        from config.config import Config

        use_sage_attention = Config().use_sage_attention

    if not use_sage_attention:
        logger.info("comfy runtime: sage attention disabled by config, using pytorch attention")
        return

    if _ATTENTION_MODULE in sys.modules:
        # The backend was already chosen at import time; setting the flag now is a no-op
        # that would silently look like it worked. Say so instead.
        logger.warning(
            "comfy runtime: %s is already imported, too late to enable sage attention. "
            "apply_comfy_runtime_args() must run before the first comfy import.",
            _ATTENTION_MODULE,
        )
        return

    # comfy's attention module calls exit(-1) when --use-sage-attention is set but the
    # package is missing, which would take the worker down at import. Only set the flag
    # once the import is known to work.
    #
    # Broad except on purpose: celery_app is imported by the orchestrator deployment too,
    # which has no GPU (values-prod worker vs orchestrator), and sageattention's CUDA
    # extension can fail there with more than just ImportError. Losing sage attention is
    # acceptable; failing the import of celery_app is not.
    try:
        import sageattention  # noqa: F401
    except Exception as e:
        logger.warning(
            "comfy runtime: sageattention unavailable (%s: %s), staying on pytorch attention",
            type(e).__name__, e,
        )
        return

    try:
        comfy_dir = _comfy_node_dir()
        if comfy_dir not in sys.path:
            sys.path.insert(0, comfy_dir)

        from comfy.cli_args import args

        args.use_sage_attention = True
    except Exception as e:
        logger.warning(
            "comfy runtime: could not set the sage attention flag (%s: %s), staying on pytorch attention",
            type(e).__name__, e,
        )
        return

    logger.info("comfy runtime: sage attention enabled")

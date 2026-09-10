"""
Storage manager that provides a unified interface for storage operations.
Combines local and remote storage providers with a clean API.

Canonical, shared between ImageGenBackend and ImageGenCelery.
"""

from typing import Any, Dict, Optional, Union
from PIL import Image
import cv2
import numpy as np
import pickle
import os
import json
import time
import threading
from pathlib import Path

from .base import StorageResult, ImageFormat
from .providers import LocalStorageProvider, RemoteStorageProvider, resolve_storage_mode
from common_lib.base_services import BaseServiceSingleton
from common_lib.logging_format import module_logger


def _default_config() -> Any:
    """Construct the calling app's own default config object. Each app
    (ImageGenBackend, ImageGenCelery) owns a top-level ``config`` package;
    resolved lazily at runtime so this module never imports either app's
    config type directly."""
    try:
        from config.config import Config  # ImageGenCelery
        return Config()
    except ImportError:
        from config.settings import Settings  # ImageGenBackend
        return Settings()


class StorageManager(BaseServiceSingleton):
    """
    High-level storage manager that provides a unified interface
    for both local and remote storage operations.
    """

    def __init__(self, config: Any = None):
        if config is None:
            config = _default_config()
        super(StorageManager, self).__init__(config)
        self.logger = module_logger("StorageManager")

        # 'local' or 'remote' -- see resolve_storage_mode. Gates every remote-only
        # method below; remote_provider is constructed lazily (only when mode is
        # actually 'remote') so a deployment with no FILE_SERVER_* credentials
        # configured (e.g. ImageGenCelery, which has none at all) never touches it.
        self.mode = resolve_storage_mode(self.config)
        self.local_provider = LocalStorageProvider(self.config)
        self._remote_provider: Optional[RemoteStorageProvider] = None

        # Background thread for cleaning up expired pickle and TTL image cache files (*.meta companions)
        self._cleanup_thread = None
        self._cleanup_interval = 60  # Check every 1 minutes

    @property
    def remote_provider(self) -> RemoteStorageProvider:
        if self._remote_provider is None:
            self._remote_provider = RemoteStorageProvider(self.config)
        return self._remote_provider

    def _remote_disabled_result(self, op: str) -> StorageResult:
        message = (
            f"'{op}' requires remote file storage, but storage mode is 'local' "
            f"(FILE_SERVER_MODE={self.mode}). Set FILE_SERVER_MODE=remote and configure "
            f"FILE_SERVER_* to enable it."
        )
        self.logger.warning(message)
        return StorageResult(success=False, error=message)

    def save_image_local(self, image: Union[Image.Image, np.ndarray],
                        filename: str,
                        folder: str = "",
                        format: ImageFormat = ImageFormat.JPEG) -> StorageResult:
        """
        Save an image to local storage.

        Args:
            image: PIL Image or numpy array
            filename: Name of the file to save
            folder: Subfolder within storage
            format: Image format to save as

        Returns:
            StorageResult with operation details
        """
        return self.local_provider.save_image(image, filename, folder, format)

    def save_image_remote(self, image: Union[Image.Image, np.ndarray],
                         filename: str,
                         folder: str = "",
                         format: ImageFormat = ImageFormat.JPEG,
                         max_attempts: int = 3) -> StorageResult:
        """
        Save an image to remote storage.

        Args:
            image: PIL Image or numpy array
            filename: Name of the file to save
            folder: Subfolder within storage
            format: Image format to save as
            max_attempts: Upload attempts for transient remote failures

        Returns:
            StorageResult with operation details
        """
        if self.mode != "remote":
            return self._remote_disabled_result("save_image_remote")
        return self.remote_provider.save_image(
            image, filename, folder, format, max_attempts=max_attempts
        )

    def save_video_remote(
        self,
        file_data: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "video/mp4",
        max_attempts: int = 3,
    ) -> StorageResult:
        """
        Upload raw video bytes to remote file storage (TDS /api/v3/Medias).

        Args:
            file_data: Encoded video file contents
            filename: Destination filename (e.g. recording.mp4)
            folder: Remote subdirectory (e.g. storage bucket folder name)
            content_type: MIME type such as video/mp4, video/webm, video/quicktime
            max_attempts: Upload attempts for transient remote failures (default 3).

        Returns:
            StorageResult with ``file_url`` for the playable asset; ``metadata`` may include
            ``preview_url`` when the API returns a separate poster/thumbnail URL.
        """
        if self.mode != "remote":
            return self._remote_disabled_result("save_video_remote")
        return self.remote_provider.save_video(
            file_data, filename, folder, content_type, max_attempts=max_attempts
        )

    def save_image_both(self, image: Union[Image.Image, np.ndarray],
                       filename: str,
                       folder: str = "",
                       format: ImageFormat = ImageFormat.JPEG,
                       max_attempts: int = 1) -> Dict[str, StorageResult]:
        """
        Save an image to both local and remote storage.

        Args:
            image: PIL Image or numpy array
            filename: Name of the file to save
            folder: Subfolder within storage
            format: Image format to save as
            max_attempts: Upload attempts for the remote save (transient failures)

        Returns:
            Dictionary with results from both storage operations
        """

        local_result = self.save_image_local(image, filename, folder, format)
        remote_result = self.save_image_remote(
            image, filename, folder, format, max_attempts=max_attempts
        )

        return {
            "local": local_result,
            "remote": remote_result
        }

    def save_pickle_local(self, data: Any,
                         filename: str,
                         folder: str = "",
                         timeout: Optional[int] = None) -> StorageResult:
        """
        Save data to a pickle file in local storage with optional timeout.

        Args:
            data: Any Python object that can be pickled
            filename: Name of the file to save (should include .pkl extension)
            folder: Subfolder within storage
            timeout: Optional timeout in seconds. File will be marked for deletion after this time.

        Returns:
            StorageResult with operation details
        """
        try:
            # Validate filename
            if not self.local_provider.validate_filename(filename):
                return StorageResult(
                    success=False,
                    error="Invalid filename"
                )

            # Ensure filename has .pkl extension
            if not filename.endswith('.pkl'):
                filename = f"{filename}.pkl"

            # Create directory if it doesn't exist
            folder_path = os.path.join(self.local_provider.base_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            # Save pickle file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            # Save expiration metadata if timeout is provided
            expiration_time = None
            if timeout is not None:
                expiration_time = time.time() + timeout
                metadata_path = file_path + '.meta'
                metadata = {
                    "expiration_time": expiration_time,
                    "timeout": timeout,
                    "created_time": time.time()
                }
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)

            # Get file size
            file_size = os.path.getsize(file_path)

            self.logger.info(f"Successfully saved pickle file to local storage: {file_path}" +
                           (f" (expires in {timeout}s)" if timeout else ""))

            return StorageResult(
                success=True,
                file_path=file_path,
                file_size=file_size,
                metadata={
                    "format": "pickle",
                    "data_type": str(type(data).__name__),
                    "expiration_time": expiration_time,
                    "timeout": timeout
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to save pickle file {filename}: {str(e)}")
            return StorageResult(
                success=False,
                error=str(e)
            )

    def load_pickle_local(self, filename: str, folder: str = "") -> StorageResult:
        """
        Load data from a pickle file in local storage.
        Checks expiration if metadata exists.

        Args:
            filename: Name of the file to load
            folder: Subfolder within storage

        Returns:
            StorageResult with loaded data
        """
        try:
            # Ensure filename has .pkl extension
            if not filename.endswith('.pkl'):
                filename = f"{filename}.pkl"

            file_path = self.local_provider.get_image_path(filename, folder)

            if not os.path.exists(file_path):
                return StorageResult(
                    success=False,
                    error="File not found"
                )

            # Check expiration if metadata exists
            metadata_path = file_path + '.meta'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    expiration_time = metadata.get("expiration_time")
                    if expiration_time and time.time() > expiration_time:
                        # File has expired, delete it
                        try:
                            os.remove(file_path)
                            os.remove(metadata_path)
                            self.logger.info(f"Deleted expired pickle file: {file_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to delete expired file {file_path}: {str(e)}")
                        return StorageResult(
                            success=False,
                            error="File has expired"
                        )

            # Load pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Get file size
            file_size = os.path.getsize(file_path)

            self.logger.info(f"Successfully loaded pickle file from local storage: {file_path}")

            return StorageResult(
                success=True,
                file_path=file_path,
                file_size=file_size,
                metadata={
                    "format": "pickle",
                    "data_type": str(type(data).__name__)
                },
                image=data  # Store data in image field for consistency with StorageResult
            )

        except Exception as e:
            self.logger.error(f"Failed to load pickle file {filename}: {str(e)}")
            return StorageResult(
                success=False,
                error=str(e)
            )

    def load_image_local(self, filename: str, folder: str = "") -> StorageResult:
        """
        Load an image from local storage.

        Args:
            filename: Name of the file to load
            folder: Subfolder within storage

        Returns:
            StorageResult with image data
        """
        return self.local_provider.load_image(filename, folder)

    def load_image_remote(self, url: str, method: str = "opencv") -> StorageResult:
        """
        Load an image from a remote URL.

        Args:
            url: URL of the image to load
            method: Method to use for loading ("opencv", "pil", or "requests")

        Returns:
            StorageResult with image data and metadata
        """
        try:
            # Both apps provide their own infrastructure.image_loader.utils
            # with the same read_image_from_url/validate_image_url API;
            # imported lazily so this resolves to whichever app is running.
            from infrastructure.image_loader.utils import read_image_from_url, validate_image_url  # pylint: disable=import-error

            # Validate URL
            if not validate_image_url(url):
                return StorageResult(
                    success=False,
                    error="Invalid image URL or URL does not point to an image"
                )

            # Load image using image_loader utility
            image_array = read_image_from_url(url, method)

            if image_array is None:
                return StorageResult(
                    success=False,
                    error="Failed to load image from URL"
                )

            # Convert numpy array to PIL Image for consistency
            if method.lower() == "opencv":
                # OpenCV returns BGR, convert to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image_array)

            return StorageResult(
                success=True,
                file_url=url,
                image=pil_image,  # Include the image in the result
                metadata={
                    "method": method,
                    "size": pil_image.size,
                    "mode": pil_image.mode,
                    "shape": image_array.shape,
                    "dtype": str(image_array.dtype)
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to load image from URL {url}: {str(e)}")
            return StorageResult(
                success=False,
                error=str(e)
            )

    def get_image_path(self, filename: str, folder: str = "") -> str:
        """
        Get the full file path for a local image.

        Args:
            filename: Name of the file
            folder: Subfolder within storage

        Returns:
            Full file path
        """
        return self.local_provider.get_image_path(filename, folder)

    def image_exists_local(self, filename: str, folder: str = "") -> bool:
        """
        Check if an image exists in local storage.

        Args:
            filename: Name of the file to check
            folder: Subfolder within storage

        Returns:
            True if image exists, False otherwise
        """
        return self.local_provider.image_exists(filename, folder)

    def save_image_local_ttl(
        self,
        image: Union[Image.Image, np.ndarray],
        filename: str,
        folder: str,
        fmt: ImageFormat,
        ttl_seconds: int,
    ) -> StorageResult:
        """Save image locally and write sibling ``{file}.meta`` for TTL eviction (cleanup thread)."""
        result = self.save_image_local(image, filename, folder, fmt)
        if not result.success or not result.file_path or ttl_seconds is None:
            return result
        try:
            file_path = result.file_path
            expiration_time = time.time() + ttl_seconds
            metadata_path = file_path + ".meta"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "expiration_time": expiration_time,
                        "timeout": ttl_seconds,
                        "created_time": time.time(),
                    },
                    f,
                )
            self.logger.info("Wrote TTL metadata for %s (expires in %ss)", file_path, ttl_seconds)
        except Exception as e:
            self.logger.warning("Failed to write TTL metadata for %s: %s", result.file_path, str(e))
        return result

    def load_image_local_ttl(self, filename: str, folder: str = "") -> StorageResult:
        """Load image; if sibling ``.meta`` exists and TTL expired, delete image + meta and return failure."""
        try:
            file_path = self.local_provider.get_image_path(filename, folder)
            if not os.path.exists(file_path):
                return StorageResult(success=False, error="File not found")
            meta_path = file_path + ".meta"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    expiration_time = metadata.get("expiration_time")
                    if expiration_time and time.time() > expiration_time:
                        try:
                            os.remove(file_path)
                            os.remove(meta_path)
                            self.logger.info("Deleted expired TTL image cache: %s", file_path)
                        except OSError as e:
                            self.logger.warning("Failed to delete expired TTL image %s: %s", file_path, str(e))
                        return StorageResult(success=False, error="File has expired")
                except Exception as e:
                    self.logger.warning("Invalid TTL metadata for %s: %s", meta_path, str(e))
            return self.local_provider.load_image(filename, folder)
        except Exception as e:
            self.logger.error("Failed to load TTL image %s: %s", filename, str(e))
            return StorageResult(success=False, error=str(e))

    def start_cleanup_thread(self):
        """Start background thread to clean up expired TTL cache files (*.pkl/*.png + sibling ``*.meta``)."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_expired_files,
                daemon=True,
                name="StorageManager-CleanupThread"
            )
            self._cleanup_thread.start()
            self.logger.info("Started background thread for cleaning up expired TTL cache files")

    def _cleanup_expired_files(self):
        """Background thread loop: delete expired TTL cache entries (paired data file + ``.meta``)."""
        while True:
            try:
                self._scan_and_cleanup_expired_files()
            except Exception as e:
                self.logger.error(f"Error in cleanup thread: {str(e)}")

            # Sleep for the cleanup interval
            time.sleep(self._cleanup_interval)

    def _scan_and_cleanup_expired_files(self):
        """Scan sibling ``*.meta`` TTL manifests and delete expired data files (``.pkl``, ``.png``, etc.)."""
        base_path = Path(self.local_provider.base_path)
        if not base_path.exists():
            return

        current_time = time.time()
        deleted_count = 0

        for meta_file in base_path.rglob("*.meta"):
            name = meta_file.name
            if not name.endswith(".meta"):
                continue
            data_file = meta_file.parent / name[: -len(".meta")]
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                expiration_time = metadata.get("expiration_time")
                if expiration_time and current_time > expiration_time:
                    if data_file.exists():
                        try:
                            os.remove(str(data_file))
                            os.remove(str(meta_file))
                            deleted_count += 1
                            self.logger.debug("Deleted expired TTL cache file: %s", data_file)
                        except OSError as e:
                            self.logger.warning("Failed to delete expired TTL file %s: %s", data_file, str(e))
            except Exception as e:
                self.logger.warning("Error processing TTL metadata %s: %s", meta_file, str(e))

        if deleted_count > 0:
            self.logger.info("Cleaned up %s expired TTL cache file(s)", deleted_count)

    def stop_cleanup_thread(self):
        """Stop the background cleanup thread.
        Note: Since it's a daemon thread, it will automatically stop when the main process exits.
        This method is kept for compatibility but daemon threads cannot be forcefully stopped.
        """
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self.logger.info("Cleanup thread is a daemon thread and will stop automatically with the main process")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about available storage providers.

        Returns:
            Dictionary with storage provider information
        """
        info = {
            "mode": self.mode,
            "local": {
                "type": self.local_provider.get_storage_type().value,
                "base_path": self.local_provider.base_path,
                "available": True
            },
        }
        if self.mode == "remote":
            info["remote"] = {
                "type": self.remote_provider.get_storage_type().value,
                "configured": bool(self.remote_provider.fileserver_secret_key and
                                 self.remote_provider.fileserver_url),
                "available": bool(self.remote_provider.fileserver_secret_key and
                                self.remote_provider.fileserver_url)
            }
        else:
            info["remote"] = {"available": False, "reason": "storage mode is 'local'"}
        return info

"""
Specialized storage provider implementations.
Each provider handles a specific type of storage backend.

Canonical, shared between ImageGenBackend and ImageGenCelery. The two apps'
config objects differ (pydantic ``Settings`` vs plain ``Config``, and the
remote-storage group is named ``storage`` vs ``file_server``) so provider
config is duck-typed via the resolver helpers below rather than importing
either app's config type.
"""

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from io import BytesIO
import requests

from PIL import Image
import numpy as np

from .base import (
    BaseStorageProvider, StorageType, StorageResult, ImageFormat,
    StorageError
)


def _exception_source_line(exc: BaseException) -> str:
    """Innermost traceback frame as ``file:line`` for log context."""
    if exc.__traceback__ is None:
        return "unknown"
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "unknown"
    last = frames[-1]
    return f"{os.path.basename(last.filename)}:{last.lineno}"


def _resolve_local_base_path(config: Any) -> str:
    """ImageGenCelery's Config exposes ``storage_path``; ImageGenBackend's
    Settings exposes ``base_url`` (its locally-served files root) for the
    same purpose. Accept either."""
    for attr in ("storage_path", "base_url"):
        value = getattr(config, attr, None)
        if value:
            return value
    raise AttributeError(
        "Config has neither 'storage_path' nor 'base_url' for local storage base path"
    )


def resolve_storage_mode(config: Any) -> str:
    """'local' or 'remote', read wherever the calling app put it.

    ImageGenBackend: config.storage.mode (StorageSettings, env FILE_SERVER_MODE).
    ImageGenCelery: config.file_server_mode (plain field, same env var name --
    Celery has no FILE_SERVER_* credential fields at all, so 'remote' isn't
    actually usable there; the flag exists only so shared storage code never
    has to special-case which app it's running in).
    Defaults to 'local' (safe: no external file-server dependency) if neither is set.
    """
    remote_group = getattr(config, "storage", None)
    if remote_group is not None and hasattr(remote_group, "mode"):
        return str(remote_group.mode or "local").strip().lower()
    top_level = getattr(config, "file_server_mode", None)
    if top_level:
        return str(top_level).strip().lower()
    return "local"


def _resolve_remote_settings(config: Any) -> Any:
    """ImageGenCelery's Config groups remote-storage fields under
    ``file_server``; ImageGenBackend's Settings uses ``storage``. Accept
    either group object, or a settings object that already has the fields
    directly (app_id, app_folder, secret, url, service_name, tenant_id,
    root_directory_name)."""
    for attr in ("file_server", "storage"):
        group = getattr(config, attr, None)
        if group is not None:
            return group
    return config


class LocalStorageProvider(BaseStorageProvider):
    """Local file system storage provider."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.base_path = _resolve_local_base_path(config)

    def get_storage_type(self) -> StorageType:
        return StorageType.LOCAL

    def save_image(self, image: Union[Image.Image, np.ndarray],
                   filename: str,
                   folder: str = "",
                   format: ImageFormat = ImageFormat.JPEG) -> StorageResult:
        """
        Save an image to local file system.

        Args:
            image: PIL Image or numpy array
            filename: Name of the file to save
            folder: Subfolder within storage
            format: Image format to save as

        Returns:
            StorageResult with operation details
        """
        try:
            # Validate inputs
            if not self.validate_filename(filename):
                return StorageResult(
                    success=False,
                    error="Invalid filename"
                )

            if not self.validate_image(image):
                return StorageResult(
                    success=False,
                    error="Invalid image data"
                )

            # Convert to PIL Image
            pil_image = self.convert_to_pil_image(image)
            # pil_image = pil_image.convert('RGB')
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert('RGBA')
            elif pil_image.mode == "RGB":
                pil_image = pil_image.convert('RGB')

            # Create directory if it doesn't exist
            folder_path = os.path.join(self.base_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            # Save image
            file_path = os.path.join(folder_path, filename)
            save_kwargs = {}
            if format == ImageFormat.JPEG:
                save_kwargs['quality'] = 85
                save_kwargs['optimize'] = True

            pil_image.save(file_path, **save_kwargs)

            # Get file size
            file_size = os.path.getsize(file_path)

            self.logger.info(f"Successfully saved image to local storage: {file_path}")

            return StorageResult(
                success=True,
                file_path=file_path,
                file_size=file_size,
                metadata={
                    "format": format.value,
                    "size": pil_image.size,
                    "mode": pil_image.mode
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to save image {filename}: {str(e)}")
            return StorageResult(
                success=False,
                error=str(e)
            )

    def get_image_path(self, filename: str, folder: str = "") -> str:
        """
        Get the full file path for a given filename and folder.

        Args:
            filename: Name of the file
            folder: Subfolder within storage

        Returns:
            Full file path
        """
        if folder:
            return os.path.join(self.base_path, folder, filename)
        return os.path.join(self.base_path, filename)

    def image_exists(self, filename: str, folder: str = "") -> bool:
        """
        Check if an image exists in local file system.

        Args:
            filename: Name of the file to check
            folder: Subfolder within storage

        Returns:
            True if image exists, False otherwise
        """
        file_path = self.get_image_path(filename, folder)
        return os.path.exists(file_path)

    def load_image(self, filename: str, folder: str = "") -> StorageResult:
        """
        Load an image from local file system.

        Args:
            filename: Name of the file to load
            folder: Subfolder within storage

        Returns:
            StorageResult with image data
        """
        try:
            file_path = self.get_image_path(filename, folder)

            if not os.path.exists(file_path):
                return StorageResult(
                    success=False,
                    error="File not found"
                )

            # Load image
            image = Image.open(file_path)
            image.load()  # Ensure image is loaded

            # Get file size
            file_size = os.path.getsize(file_path)

            self.logger.info(f"Successfully loaded image from local storage: {file_path}")

            return StorageResult(
                success=True,
                file_path=file_path,
                file_size=file_size,
                image=image,
                metadata={
                    "format": image.format,
                    "size": image.size,
                    "mode": image.mode
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to load image {filename}: {str(e)}")
            return StorageResult(
                success=False,
                error=str(e)
            )


def remote_video_preview_and_file_urls(uploaded: Dict[str, Any]) -> Tuple[str, str]:
    """
    Parse /api/v3/Medias merged response after GET by id: ``downloadUrl`` = playable video;
    ``thumbnail`` / ``thumbnails`` = poster image from the API. Returns (preview_url, file_url).
    """
    if not isinstance(uploaded, dict):
        return "", ""

    video_ext = (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv", ".m3u8")
    image_ext = (".png", ".jpg", ".jpeg", ".webp", ".gif")

    def path_lower(u: str) -> str:
        return (u.split("?")[0].split("#")[0] or "").lower()

    def is_video_url(u: str) -> bool:
        p = path_lower(u)
        return any(p.endswith(ext) for ext in video_ext)

    def is_image_url(u: str) -> bool:
        p = path_lower(u)
        return any(p.endswith(ext) for ext in image_ext)

    def http_str(v: Any) -> str:
        if isinstance(v, str) and v.strip().startswith("http"):
            return v.strip()
        return ""

    file_url = ""
    for key in (
        "downloadUrl",
        "url",
        "fileUrl",
        "mediaUrl",
        "mediaUrlOriginal",
        "originalUrl",
        "path",
        "contentPath",
        "fullPath",
        "contentUrl",
    ):
        u = http_str(uploaded.get(key))
        if u:
            file_url = u
            break

    if not file_url:
        for _key, u in uploaded.items():
            u = http_str(u)
            if u and is_video_url(u):
                file_url = u
                break

    if not file_url:
        for key in ("originalThumbnail", "thumbnail"):
            u = http_str(uploaded.get(key))
            if u and is_video_url(u):
                file_url = u
                break

    preview_url = ""
    for key in ("thumbnail", "poster", "posterThumbnail", "previewThumbnail", "previewUrl"):
        u = http_str(uploaded.get(key))
        if u and (is_image_url(u) or (file_url and u != file_url)):
            preview_url = u
            break

    if not preview_url:
        for key in ("originalThumbnail", "thumbnail"):
            u = http_str(uploaded.get(key))
            if u and is_image_url(u):
                preview_url = u
                break

    if not preview_url:
        for key in ("thumbnail", "originalThumbnail"):
            u = http_str(uploaded.get(key))
            if u and u != file_url:
                preview_url = u
                break

    if not preview_url:
        thumbs = uploaded.get("thumbnails")
        if isinstance(thumbs, list) and thumbs:
            u = http_str(thumbs[0])
            if u and u != file_url:
                preview_url = u

    if not file_url:
        u = http_str(uploaded.get("originalThumbnail")) or http_str(uploaded.get("thumbnail"))
        if u:
            file_url = u

    if file_url and is_image_url(file_url) and not is_video_url(file_url):
        for _key, u in uploaded.items():
            u = http_str(u)
            if u and is_video_url(u):
                file_url = u
                break

    if not preview_url and file_url:
        preview_url = file_url

    return preview_url, file_url


class RemoteStorageProvider(BaseStorageProvider):
    """Remote file server storage provider."""

    image_app_id: Optional[str]
    image_app_folder: Optional[str]
    service_name: Optional[str]
    tenant_id: Optional[str]
    fileserver_url: Optional[str]
    fileserver_secret_key: Optional[str]

    def __init__(self, config: Any):
        super().__init__(config)

        # Remote storage specific configuration
        remote = _resolve_remote_settings(config)
        self.image_app_id = remote.app_id
        self.image_app_folder = remote.app_folder
        self.fileserver_secret_key = remote.secret
        self.fileserver_url = remote.url
        self.service_name = remote.service_name
        self.tenant_id = remote.tenant_id or "ImageGen"
        self._root_directory_name = remote.root_directory_name
        self._root_directory_ids_by_name: Dict[str, str] = {}
        self._root_directory_id: Optional[str] = None
        self._subdirectory_ids_by_name: Dict[str, str] = {}

        # Prime directory caches on init (best-effort)
        try:
            self._root_directory_id = self.get_root_directory_id("ImageGen")
            for entry in self.list_subdirectories(self._root_directory_id):
                name = entry.get("displayName")
                sub_id = entry.get("id")
                if name and sub_id:
                    self._subdirectory_ids_by_name[name] = sub_id
        except Exception as e:
            self.logger.warning(f"Failed to prefetch directory ids: {e}")

    def get_storage_type(self) -> StorageType:
        return StorageType.REMOTE

    def _build_api_url(self, api_path: str) -> str:
        if not self.fileserver_url:
            raise StorageError("Remote storage base URL is not configured")
        return os.path.join(str(self.fileserver_url).rstrip("/"), api_path.lstrip("/"))

    def get_root_directory_id(self, folder_display_name: str = "ImageGen") -> str:
        """
        Resolve a root directory id by its display name.

        Calls GET /api/v3/Directories/root-directories and returns the `id`
        for the item whose `displayName` matches `folder_display_name`.
        """
        cached = self._root_directory_ids_by_name.get(folder_display_name)
        if cached:
            return cached

        if not self.fileserver_url:
            raise StorageError("Remote storage base URL is not configured")

        url = self._build_api_url("/api/v3/Directories/root-directories")
        headers = {
            "accept": "application/json",
            "App-Id": self.image_app_id,
            "App-Secret": self.fileserver_secret_key,
            "App-Folder": self.image_app_folder,
            "Tenant-Id": self.tenant_id,
            "Service-Name": self.service_name,
        }

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as e:
            self.logger.error(f"Failed to fetch root directories: {e}")
            raise StorageError(f"Failed to fetch root directories: {e}")

        items: List[Dict[str, Any]] = payload.get("items") or []

        for item in items:
            if item.get("displayName") == folder_display_name:
                directory_id = item.get("id")
                if not directory_id:
                    break
                self._root_directory_ids_by_name[folder_display_name] = directory_id
                return directory_id

        # If root directory doesn't exist yet, create system directory and return id
        created_id = self.create_system_root_directory(folder_display_name)
        self._root_directory_ids_by_name[folder_display_name] = created_id
        return created_id

    def create_system_root_directory(self, root_directory_name: str) -> str:
        """
        Create a system root directory.

        Calls POST /api/v3/Directories/create-system with body {"name": <root_directory_name>}.
        Returns the created directory id.
        """
        url = self._build_api_url("/api/v3/Directories/create-system")
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "App-Id": self.image_app_id,
            "App-Secret": self.fileserver_secret_key,
            "App-Folder": self.image_app_folder,
            "Tenant-Id": self.tenant_id,
            "Service-Name": self.service_name,
        }
        body = {"name": root_directory_name}

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as e:
            self.logger.error(
                f"Failed to create system root directory '{root_directory_name}': {e}"
            )
            raise StorageError(f"Failed to create system root directory: {e}")

        created_id = payload.get("id")
        if not created_id:
            raise StorageError("Create system directory succeeded but response has no id")
        return created_id

    def list_subdirectories(self, directory_id: str) -> List[Dict[str, str]]:
        """
        List immediate subdirectories for a given directory id.

        Calls GET /api/v3/Directories/contents with params {"DirectoryId": directory_id}
        and returns a list of {"id": "...", "displayName": "..."} entries.
        """
        if not self.fileserver_url:
            raise StorageError("Remote storage base URL is not configured")

        url = self._build_api_url("/api/v3/Directories/contents")
        headers = {
            "accept": "application/json",
            "App-Id": self.image_app_id,
            "App-Secret": self.fileserver_secret_key,
            "App-Folder": self.image_app_folder,
            "Tenant-Id": self.tenant_id,
            "Service-Name": self.service_name,
        }
        params = {"DirectoryId": directory_id}

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as e:
            self.logger.error(f"Failed to fetch directory contents for {directory_id}: {e}")
            raise StorageError(f"Failed to fetch directory contents: {e}")

        results: List[Dict[str, str]] = []
        for item in (payload.get("items") or []):
            if not item.get("isDir", False):
                continue
            sub_id = item.get("id")
            sub_name = item.get("displayName")
            if sub_id and sub_name:
                results.append({"id": sub_id, "displayName": sub_name})
        return results

    def get_subdirectory_id(
        self,
        root_directory_id: str,
        subfolder_display_name: str,
    ) -> Dict[str, str]:
        """
        Convenience lookup: find a subdirectory by displayName under a parent directory.

        Returns {"id": "...", "displayName": "..."}.
        """
        for entry in self.list_subdirectories(root_directory_id):
            if entry.get("displayName") == subfolder_display_name:
                return entry
        raise StorageError(
            f'Subdirectory "{subfolder_display_name}" not found under "{root_directory_id}"'
        )

    def create_subdirectory(
        self,
        root_directory_id: str,
        subfolder_name: str,
        description: str = "",
        is_lock: bool = True,
    ) -> str:
        """
        Create a subdirectory under a parent directory.

        Calls POST /api/v3/Directories with body:
        {"parentId": root_directory_id, "name": subfolder_name, "description": "", "isLock": True}

        Returns the created directory id.
        """
        if not self.fileserver_url:
            raise StorageError("Remote storage base URL is not configured")

        url = f"{str(self.fileserver_url).rstrip('/')}/api/v3/Directories"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "App-Id": self.image_app_id,
            "App-Secret": self.fileserver_secret_key,
            "App-Folder": self.image_app_folder,
            "Tenant-Id": self.tenant_id,
            "Service-Name": self.service_name,
        }
        body = {
            "parentId": root_directory_id,
            "name": subfolder_name,
            "description": description,
            "isLock": is_lock,
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            payload = resp.json() or {}
        except Exception as e:
            self.logger.error(f"Failed to create subdirectory '{subfolder_name}': {e}")
            raise StorageError(f"Failed to create subdirectory: {e}")

        created_id = payload.get("id")
        if not created_id:
            raise StorageError("Create directory succeeded but response has no id")

        # Best-effort cache for follow-up lookups
        self._subdirectory_ids_by_name[subfolder_name] = created_id
        return created_id

    def get_or_create_subdirectory_id(self, subfolder_name: str) -> str:
        """
        Resolve a subfolder id by name under the ImageGen root.

        Uses cached ids loaded at init; if missing, it creates the folder and returns its id.
        """
        cached = self._subdirectory_ids_by_name.get(subfolder_name)
        if cached:
            return cached

        root_id = self._root_directory_id or self.get_root_directory_id(self._root_directory_name)
        self._root_directory_id = root_id

        # Refresh cache from server before creating
        for entry in self.list_subdirectories(root_id):
            name = entry.get("displayName")
            sub_id = entry.get("id")
            if name and sub_id:
                self._subdirectory_ids_by_name[name] = sub_id
        cached = self._subdirectory_ids_by_name.get(subfolder_name)
        if cached:
            return cached

        created_id = self.create_subdirectory(root_id, subfolder_name, description="", is_lock=True)
        self._subdirectory_ids_by_name[subfolder_name] = created_id
        return created_id

    def _fetch_media_detail_after_upload(
        self,
        uploaded_file: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        For video uploads, POST /api/v3/Medias may omit ``downloadUrl`` or a stable poster.
        GET /api/v3/Medias/{id} returns the full Media object: use ``downloadUrl`` for the
        video URL and ``thumbnail`` for the poster image (first-frame preview from the service).
        """
        if not self.fileserver_url:
            raise StorageError("Remote storage base URL is not configured")
        media_id = uploaded_file.get("id")
        if not media_id:
            return uploaded_file
        detail_url = f"{str(self.fileserver_url).rstrip('/')}/api/v3/Medias/{media_id}"
        try:
            detail_resp = requests.get(detail_url, headers=headers, timeout=60)
            detail_resp.raise_for_status()
            detail = detail_resp.json()
        except Exception as e:
            self.logger.error(f"Medias GET {detail_url} failed: {e}")
            raise StorageError(f"Failed to fetch media detail after video upload: {e}") from e
        if not isinstance(detail, dict):
            raise StorageError("Medias GET returned invalid JSON")
        return {**uploaded_file, **detail}

    def _upload_to_remote(
        self,
        file_data: bytes,
        content_type: str,
        filename: str,
        folder: str,
        max_attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        Upload file data to remote server.

        Args:
            file_data: File data as bytes
            content_type: MIME type of the file
            filename: Name of the file
            folder: Folder path on remote server
            max_attempts: Number of attempts for transient HTTP/network failures (minimum 1).

        Returns:
            Response from remote server
        """
        if not self.fileserver_url:
            raise StorageError("Remote storage base URL is not configured")

        directory_id = self.get_or_create_subdirectory_id(folder) if folder else (
            self._root_directory_id or self.get_root_directory_id("ImageGen")
        )
        self._root_directory_id = self._root_directory_id or (
            directory_id if not folder else self._root_directory_id
        )

        url = f"{str(self.fileserver_url).rstrip('/')}/api/v3/Medias"
        headers = {
            "accept": "application/json",
            "App-Id": self.image_app_id,
            "App-Secret": self.fileserver_secret_key,
            "App-Folder": self.image_app_folder,
            "Tenant-Id": self.tenant_id,
            "Service-Name": self.service_name,
        }
        files = {
            "Files": (filename, file_data, content_type),
        }
        data = {
            "DirectoryId": directory_id,
        }

        ct_norm = (content_type or "").split(";")[0].strip().lower()
        is_video = ct_norm.startswith("video/")

        max_attempts = max(1, int(max_attempts))
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=1000,
                )
                response.raise_for_status()
                payload = response.json() or {}

                # /api/v3/Medias returns:
                # {
                #   "results": [{"uploadedFile": {"id","thumbnail","originalThumbnail",...}, ...}],
                #   ...
                # }
                if isinstance(payload, dict):
                    results = payload.get("results") or []
                    if isinstance(results, list) and results:
                        first = results[0] or {}
                        uploaded_file = first.get("uploadedFile") or {}
                        if isinstance(uploaded_file, dict) and uploaded_file.get("id"):
                            if is_video:
                                return self._fetch_media_detail_after_upload(
                                    uploaded_file, headers
                                )
                            return uploaded_file
                    err = StorageError("Upload succeeded but returned no uploadedFile")
                else:
                    err = StorageError("Unexpected upload response format")
                if attempt < max_attempts - 1:
                    self.logger.warning(
                        "Remote upload bad response, retry %s/%s: %s",
                        attempt + 1,
                        max_attempts,
                        err,
                    )
                    time.sleep(min(2**attempt, 30))
                    continue
                raise err
            except StorageError as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(
                        "Remote upload storage error, retry %s/%s: %s",
                        attempt + 1,
                        max_attempts,
                        e,
                    )
                    time.sleep(min(2**attempt, 30))
                    continue
                self.logger.error(f"Failed to upload to remote server: {str(e)}")
                raise
            except requests.HTTPError as e:
                if attempt < max_attempts - 1:
                    status = e.response.status_code if e.response is not None else None
                    self.logger.warning(
                        "Remote upload HTTP error (status=%s), retry %s/%s: %s",
                        status,
                        attempt + 1,
                        max_attempts,
                        e,
                    )
                    time.sleep(min(2**attempt, 30))
                    continue
                self.logger.error(f"Failed to upload to remote server: {str(e)}")
                raise StorageError(f"Remote upload failed: {str(e)}") from e
            except requests.RequestException as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(
                        "Remote upload request error, retry %s/%s: %s",
                        attempt + 1,
                        max_attempts,
                        e,
                    )
                    time.sleep(min(2**attempt, 30))
                    continue
                self.logger.error(f"Failed to upload to remote server: {str(e)}")
                raise StorageError(f"Remote upload failed: {str(e)}") from e
            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(
                        "Remote upload error, retry %s/%s: %s",
                        attempt + 1,
                        max_attempts,
                        e,
                    )
                    time.sleep(min(2**attempt, 30))
                    continue
                self.logger.error(f"Failed to upload to remote server: {str(e)}")
                raise StorageError(f"Remote upload failed: {str(e)}") from e
        raise StorageError("Remote upload failed after retries")

    def save_image(self, image: Union[Image.Image, np.ndarray],
                   filename: str,
                   folder: str = "",
                   format: ImageFormat = ImageFormat.JPEG,
                   max_attempts: int = 1) -> StorageResult:
        """
        Save an image to remote storage.

        Args:
            image: PIL Image or numpy array
            filename: Name of the file to save
            folder: Subfolder within storage
            format: Image format to save as
            max_attempts: Upload attempts for transient remote failures (passed to Medias upload).

        Returns:
            StorageResult with operation details
        """
        try:
            # Check if remote storage is configured
            if not self.fileserver_secret_key or not self.fileserver_url:
                return StorageResult(
                    success=False,
                    error="Remote storage not configured"
                )

            # Validate inputs
            if not self.validate_filename(filename):
                return StorageResult(
                    success=False,
                    error="Invalid filename"
                )

            if not self.validate_image(image):
                return StorageResult(
                    success=False,
                    error="Invalid image data"
                )

            # Convert to PIL Image
            pil_image = self.convert_to_pil_image(image)

            # Prepare image data
            save_kwargs = {}
            if format == ImageFormat.JPEG:
                save_kwargs['quality'] = 85
                save_kwargs['optimize'] = True

            # Save to temporary buffer
            buffer = BytesIO()
            pil_image.save(buffer, format=format.value.upper(), **save_kwargs)
            file_data = buffer.getvalue()

            # Upload to remote server
            content_type = f"image/{format.value}"
            response = self._upload_to_remote(
                file_data, content_type, filename, folder, max_attempts=max_attempts
            )
            return StorageResult(
                success=True,
                file_url=response.get("originalThumbnail") or response.get("thumbnail"),
                file_size=len(file_data),
                metadata={
                    "format": format.value,
                    "remote_id": response.get("id"),
                    "thumbnail": response.get("thumbnail"),
                    "original_thumbnail": response.get("originalThumbnail"),
                }
            )

        except Exception as e:
            self.logger.error(
                "Failed to save image %r (folder=%r, format=%s, error_at=%s): %s: %s",
                filename,
                folder,
                format.value,
                _exception_source_line(e),
                type(e).__name__,
                e,
                exc_info=True,
            )
            return StorageResult(
                success=False,
                error=str(e)
            )

    def save_video(
        self,
        file_data: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "video/mp4",
        max_attempts: int = 3,
    ) -> StorageResult:
        """
        Upload raw video bytes to remote storage (same /api/v3/Medias endpoint as images).

        Args:
            file_data: Encoded video bytes
            filename: Destination filename (e.g. clip.mp4)
            folder: Subdirectory name under the remote root (e.g. StableDiffusion)
            content_type: MIME type (video/mp4, video/webm, video/quicktime, ...)
            max_attempts: Upload attempts for transient remote failures (default 3).

        Returns:
            StorageResult with file_url pointing at the playable video when the API exposes it;
            metadata includes preview_url (poster) when separate from file_url.
        """
        try:
            if not self.fileserver_secret_key or not self.fileserver_url:
                return StorageResult(
                    success=False,
                    error="Remote storage not configured",
                )
            if not self.validate_filename(filename):
                return StorageResult(
                    success=False,
                    error="Invalid filename",
                )
            if not file_data:
                return StorageResult(
                    success=False,
                    error="Empty video data",
                )

            ct = (content_type or "video/mp4").split(";")[0].strip().lower()
            if not ct.startswith("video/"):
                ct = "video/mp4"

            response = self._upload_to_remote(
                file_data, ct, filename, folder, max_attempts=max_attempts
            )
            preview_url, video_url = remote_video_preview_and_file_urls(response)
            if not video_url:
                return StorageResult(
                    success=False,
                    error="Video upload failed (no URL returned)",
                )

            meta: Dict[str, Any] = {
                "content_type": ct,
                "preview_url": preview_url,
                "remote_id": response.get("id"),
                "thumbnail": response.get("thumbnail"),
                "original_thumbnail": response.get("originalThumbnail"),
            }
            for k in ("width", "height", "duration", "fps"):
                v = response.get(k)
                if v is not None:
                    meta[k] = v

            return StorageResult(
                success=True,
                file_url=video_url,
                file_size=len(file_data),
                metadata=meta,
            )
        except Exception as e:
            self.logger.error(f"Failed to save video {filename}: {str(e)}", exc_info=True)
            return StorageResult(
                success=False,
                error=str(e),
            )

    def upload_image(self, image: Union[Image.Image, np.ndarray],
                    image_name: str,
                    storage_id: str,
                    format: ImageFormat = ImageFormat.JPEG) -> Dict[str, str]:
        """
        Legacy method for backward compatibility.

        Args:
            image: PIL Image or numpy array
            image_name: Name of the image file
            storage_id: Storage folder ID
            format: Image format

        Returns:
            Dictionary with image URL
        """
        result = self.save_image(image, image_name, storage_id, format)

        if result.success and result.file_url:
            return {"image_url": result.file_url}
        return {}

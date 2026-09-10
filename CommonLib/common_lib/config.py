"""Shared infra settings (Mongo/Redis/RabbitMQ) for ImageGenBackend + ImageGenCelery.

Each group reads its own env prefix, matching the env var names already used by
both services in production — no .env / deployment env var needs to change.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONGO_", env_file=".env", extra="ignore")
    uri: str = ""

    # Collection names — not currently env-driven (plain constants), grouped here since
    # they're all Mongo collections. Overridable via MONGO_DATABASE_NAME / MONGO_COLLECTION_* if needed.
    database_name: str = "StableDiffusion"
    collection_diffusion: str = "jobs"
    collection_human_gen: str = "HumanGenResult"
    collection_human_pose: str = "HumanPose"
    collection_fashion_attributes_extraction: str = "FashionAttributeData"
    collection_face_reference: str = "FaceReference"
    collection_user_usage: str = "UserUsage"
    collection_brand_models: str = "BrandModels"
    collection_library_assets: str = "LibraryAssets"
    collection_library_collections: str = "LibraryCollections"
    # Slugs + labels for Reference tab sidebar (admin can add/remove; seeded from defaults if empty).
    collection_library_reference_categories: str = "LibraryReferenceCategories"
    collection_library_reference_subcategories: str = "LibraryReferenceSubcategories"
    # Grants (grant_id) and magic links (link_id + token) in one collection.
    collection_library_sharing: str = "LibrarySharing"
    collection_announcements: str = "Announcements"
    collection_announcement_reads: str = "AnnouncementReads"
    # Cached ecommerce category trees (omni-channel) + audit trail.
    collection_product_type: str = "ProductType"
    collection_product_type_history: str = "ProductTypeHistory"
    # Single-doc allow/deny config for which inference_v1 APIs (and, for
    # image_gen/upscale, which model_type sub-categories) are usable.
    collection_api_permissions: str = "ApiPermissions"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")
    host: str = "172.29.32.24"
    port: int = 6379
    db: int = 1
    password: str = ""
    prefix_key: str = "image_gen"


class RabbitMQSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RABBITMQ_", env_file=".env", extra="ignore")
    host: str = "172.29.32.24"
    user: str = "admin"
    password: str = "admin"
    port: int = 5672
    vhost: str = "tuannha"

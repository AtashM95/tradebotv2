"""
Static Assets Manager for Ultimate Trading Bot v2.2.

This module provides static asset management including:
- Asset versioning for cache busting
- CSS/JS bundling configuration
- Asset URL generation
- Content hash generation
"""

import logging
import hashlib
import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from functools import lru_cache


logger = logging.getLogger(__name__)


@dataclass
class AssetConfig:
    """Configuration for static assets."""

    static_folder: str = "static"
    static_url_path: str = "/static"
    version: str = "2.2.0"
    use_hash: bool = True
    minify: bool = True
    bundle_css: bool = True
    bundle_js: bool = True
    cdn_url: str | None = None


@dataclass
class CSSBundle:
    """CSS bundle configuration."""

    name: str
    files: list[str]
    output: str | None = None

    def __post_init__(self) -> None:
        """Set default output path."""
        if not self.output:
            self.output = f"dist/css/{self.name}.min.css"


@dataclass
class JSBundle:
    """JavaScript bundle configuration."""

    name: str
    files: list[str]
    output: str | None = None
    defer: bool = True
    async_load: bool = False

    def __post_init__(self) -> None:
        """Set default output path."""
        if not self.output:
            self.output = f"dist/js/{self.name}.min.js"


@dataclass
class AssetManifest:
    """Asset manifest for versioned files."""

    assets: dict[str, str] = field(default_factory=dict)
    generated_at: str = ""

    def get(self, path: str) -> str:
        """Get versioned path for asset."""
        return self.assets.get(path, path)


# Default CSS bundles
DEFAULT_CSS_BUNDLES = [
    CSSBundle(
        name="vendor",
        files=[
            "css/vendor/normalize.css",
            "css/vendor/grid.css",
        ],
    ),
    CSSBundle(
        name="main",
        files=[
            "css/base.css",
            "css/layout.css",
            "css/components.css",
            "css/utilities.css",
        ],
    ),
    CSSBundle(
        name="dashboard",
        files=[
            "css/pages/dashboard.css",
            "css/widgets.css",
            "css/charts.css",
        ],
    ),
    CSSBundle(
        name="trading",
        files=[
            "css/pages/trading.css",
            "css/order-form.css",
        ],
    ),
]

# Default JS bundles
DEFAULT_JS_BUNDLES = [
    JSBundle(
        name="vendor",
        files=[
            "js/vendor/chart.min.js",
            "js/vendor/dayjs.min.js",
        ],
        defer=True,
    ),
    JSBundle(
        name="main",
        files=[
            "js/utils.js",
            "js/api.js",
            "js/websocket.js",
            "js/notifications.js",
            "js/theme.js",
        ],
        defer=True,
    ),
    JSBundle(
        name="dashboard",
        files=[
            "js/pages/dashboard.js",
            "js/widgets.js",
            "js/charts.js",
        ],
        defer=True,
    ),
    JSBundle(
        name="trading",
        files=[
            "js/pages/trading.js",
            "js/order-form.js",
        ],
        defer=True,
    ),
]


class StaticAssetManager:
    """
    Manager for static assets.

    Handles versioning, bundling configuration, and URL generation.
    """

    def __init__(
        self,
        config: AssetConfig | None = None,
        css_bundles: list[CSSBundle] | None = None,
        js_bundles: list[JSBundle] | None = None,
    ) -> None:
        """
        Initialize asset manager.

        Args:
            config: Asset configuration
            css_bundles: CSS bundle definitions
            js_bundles: JS bundle definitions
        """
        self._config = config or AssetConfig()
        self._css_bundles = {b.name: b for b in (css_bundles or DEFAULT_CSS_BUNDLES)}
        self._js_bundles = {b.name: b for b in (js_bundles or DEFAULT_JS_BUNDLES)}
        self._manifest = AssetManifest()
        self._hash_cache: dict[str, str] = {}

        logger.info("StaticAssetManager initialized")

    @property
    def config(self) -> AssetConfig:
        """Get asset configuration."""
        return self._config

    @property
    def manifest(self) -> AssetManifest:
        """Get asset manifest."""
        return self._manifest

    def get_url(self, path: str) -> str:
        """
        Get URL for static asset.

        Args:
            path: Asset path relative to static folder

        Returns:
            Full URL for asset
        """
        # Check manifest first
        versioned_path = self._manifest.get(path)

        # Build URL
        if self._config.cdn_url:
            base_url = self._config.cdn_url.rstrip("/")
        else:
            base_url = self._config.static_url_path

        url = f"{base_url}/{versioned_path}"

        # Add version query string if not using hashed filenames
        if self._config.use_hash and path == versioned_path:
            file_hash = self._get_file_hash(path)
            if file_hash:
                url = f"{url}?v={file_hash[:8]}"
            else:
                url = f"{url}?v={self._config.version}"

        return url

    def get_css_url(self, bundle_name: str) -> str:
        """
        Get URL for CSS bundle.

        Args:
            bundle_name: Bundle name

        Returns:
            CSS bundle URL
        """
        bundle = self._css_bundles.get(bundle_name)
        if not bundle:
            logger.warning(f"CSS bundle not found: {bundle_name}")
            return ""

        if self._config.bundle_css and bundle.output:
            return self.get_url(bundle.output)

        # Return first file if not bundling
        if bundle.files:
            return self.get_url(bundle.files[0])

        return ""

    def get_js_url(self, bundle_name: str) -> str:
        """
        Get URL for JS bundle.

        Args:
            bundle_name: Bundle name

        Returns:
            JS bundle URL
        """
        bundle = self._js_bundles.get(bundle_name)
        if not bundle:
            logger.warning(f"JS bundle not found: {bundle_name}")
            return ""

        if self._config.bundle_js and bundle.output:
            return self.get_url(bundle.output)

        # Return first file if not bundling
        if bundle.files:
            return self.get_url(bundle.files[0])

        return ""

    def get_css_files(self, bundle_name: str) -> list[str]:
        """
        Get individual CSS file URLs for a bundle.

        Args:
            bundle_name: Bundle name

        Returns:
            List of CSS file URLs
        """
        bundle = self._css_bundles.get(bundle_name)
        if not bundle:
            return []

        return [self.get_url(f) for f in bundle.files]

    def get_js_files(self, bundle_name: str) -> list[str]:
        """
        Get individual JS file URLs for a bundle.

        Args:
            bundle_name: Bundle name

        Returns:
            List of JS file URLs
        """
        bundle = self._js_bundles.get(bundle_name)
        if not bundle:
            return []

        return [self.get_url(f) for f in bundle.files]

    def get_css_tag(
        self,
        bundle_name: str,
        media: str = "all",
    ) -> str:
        """
        Get CSS link tag for bundle.

        Args:
            bundle_name: Bundle name
            media: Media query

        Returns:
            HTML link tag
        """
        url = self.get_css_url(bundle_name)
        if not url:
            return ""

        return f'<link rel="stylesheet" href="{url}" media="{media}">'

    def get_js_tag(
        self,
        bundle_name: str,
        defer: bool | None = None,
        async_load: bool | None = None,
    ) -> str:
        """
        Get JS script tag for bundle.

        Args:
            bundle_name: Bundle name
            defer: Defer loading
            async_load: Async loading

        Returns:
            HTML script tag
        """
        bundle = self._js_bundles.get(bundle_name)
        url = self.get_js_url(bundle_name)
        if not url:
            return ""

        attrs = [f'src="{url}"']

        if defer is None and bundle:
            defer = bundle.defer
        if async_load is None and bundle:
            async_load = bundle.async_load

        if defer:
            attrs.append("defer")
        if async_load:
            attrs.append("async")

        return f'<script {" ".join(attrs)}></script>'

    def get_preload_tag(
        self,
        path: str,
        as_type: str = "script",
    ) -> str:
        """
        Get preload link tag.

        Args:
            path: Asset path
            as_type: Resource type

        Returns:
            HTML preload tag
        """
        url = self.get_url(path)
        return f'<link rel="preload" href="{url}" as="{as_type}">'

    def get_all_css_tags(self) -> str:
        """
        Get all CSS bundle tags.

        Returns:
            HTML link tags
        """
        tags = []
        for bundle_name in self._css_bundles:
            tag = self.get_css_tag(bundle_name)
            if tag:
                tags.append(tag)
        return "\n".join(tags)

    def get_all_js_tags(self) -> str:
        """
        Get all JS bundle tags.

        Returns:
            HTML script tags
        """
        tags = []
        for bundle_name in self._js_bundles:
            tag = self.get_js_tag(bundle_name)
            if tag:
                tags.append(tag)
        return "\n".join(tags)

    def _get_file_hash(self, path: str) -> str:
        """
        Get content hash for file.

        Args:
            path: File path

        Returns:
            Content hash or empty string
        """
        if path in self._hash_cache:
            return self._hash_cache[path]

        full_path = Path(self._config.static_folder) / path
        if not full_path.exists():
            return ""

        try:
            content = full_path.read_bytes()
            file_hash = hashlib.md5(content).hexdigest()
            self._hash_cache[path] = file_hash
            return file_hash
        except Exception as e:
            logger.warning(f"Failed to hash file {path}: {e}")
            return ""

    def load_manifest(self, manifest_path: str) -> None:
        """
        Load asset manifest from file.

        Args:
            manifest_path: Path to manifest JSON file
        """
        import json

        try:
            with open(manifest_path) as f:
                data = json.load(f)
                self._manifest = AssetManifest(
                    assets=data.get("assets", {}),
                    generated_at=data.get("generated_at", ""),
                )
            logger.info(f"Loaded asset manifest: {len(self._manifest.assets)} assets")
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")

    def save_manifest(self, manifest_path: str) -> None:
        """
        Save asset manifest to file.

        Args:
            manifest_path: Path to manifest JSON file
        """
        import json
        from datetime import datetime

        self._manifest.generated_at = datetime.now().isoformat()

        try:
            with open(manifest_path, "w") as f:
                json.dump({
                    "assets": self._manifest.assets,
                    "generated_at": self._manifest.generated_at,
                }, f, indent=2)
            logger.info(f"Saved asset manifest: {manifest_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def clear_cache(self) -> None:
        """Clear file hash cache."""
        self._hash_cache.clear()


# Global asset manager instance
_asset_manager: StaticAssetManager | None = None


def get_asset_manager() -> StaticAssetManager:
    """
    Get or create global asset manager.

    Returns:
        StaticAssetManager instance
    """
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = StaticAssetManager()
    return _asset_manager


def static_url(path: str) -> str:
    """
    Get versioned static URL.

    Args:
        path: Asset path

    Returns:
        Full URL
    """
    return get_asset_manager().get_url(path)


def css_url(bundle_name: str) -> str:
    """
    Get CSS bundle URL.

    Args:
        bundle_name: Bundle name

    Returns:
        CSS URL
    """
    return get_asset_manager().get_css_url(bundle_name)


def js_url(bundle_name: str) -> str:
    """
    Get JS bundle URL.

    Args:
        bundle_name: Bundle name

    Returns:
        JS URL
    """
    return get_asset_manager().get_js_url(bundle_name)


def css_tag(bundle_name: str, **kwargs: Any) -> str:
    """
    Get CSS link tag.

    Args:
        bundle_name: Bundle name
        **kwargs: Additional arguments

    Returns:
        HTML link tag
    """
    return get_asset_manager().get_css_tag(bundle_name, **kwargs)


def js_tag(bundle_name: str, **kwargs: Any) -> str:
    """
    Get JS script tag.

    Args:
        bundle_name: Bundle name
        **kwargs: Additional arguments

    Returns:
        HTML script tag
    """
    return get_asset_manager().get_js_tag(bundle_name, **kwargs)


def register_asset_helpers(app: Any) -> None:
    """
    Register asset helper functions with Flask app.

    Args:
        app: Flask application
    """
    manager = get_asset_manager()

    @app.context_processor
    def inject_asset_helpers() -> dict[str, Any]:
        """Inject asset helpers into templates."""
        return {
            "static_url": manager.get_url,
            "css_url": manager.get_css_url,
            "js_url": manager.get_js_url,
            "css_tag": manager.get_css_tag,
            "js_tag": manager.get_js_tag,
            "css_files": manager.get_css_files,
            "js_files": manager.get_js_files,
            "all_css_tags": manager.get_all_css_tags,
            "all_js_tags": manager.get_all_js_tags,
        }


# Module version
__version__ = "2.2.0"

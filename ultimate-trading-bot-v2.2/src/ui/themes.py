"""
Themes Module for Ultimate Trading Bot v2.2.

This module provides theme management including:
- Theme definitions
- Color schemes
- CSS variable generation
- User preference handling
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ThemeMode(str, Enum):
    """Theme mode enumeration."""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class ColorPalette:
    """Color palette definition."""

    # Primary colors
    primary: str = "#3b82f6"
    primary_light: str = "#60a5fa"
    primary_dark: str = "#2563eb"

    # Secondary colors
    secondary: str = "#6b7280"
    secondary_light: str = "#9ca3af"
    secondary_dark: str = "#4b5563"

    # Accent colors
    accent: str = "#f59e0b"
    accent_light: str = "#fbbf24"
    accent_dark: str = "#d97706"

    # Semantic colors
    success: str = "#10b981"
    success_light: str = "#34d399"
    success_dark: str = "#059669"

    danger: str = "#ef4444"
    danger_light: str = "#f87171"
    danger_dark: str = "#dc2626"

    warning: str = "#f59e0b"
    warning_light: str = "#fbbf24"
    warning_dark: str = "#d97706"

    info: str = "#3b82f6"
    info_light: str = "#60a5fa"
    info_dark: str = "#2563eb"

    # Trading colors
    positive: str = "#10b981"
    negative: str = "#ef4444"
    neutral: str = "#6b7280"


@dataclass
class ThemeColors:
    """Theme color configuration."""

    # Background colors
    bg_primary: str = "#ffffff"
    bg_secondary: str = "#f9fafb"
    bg_tertiary: str = "#f3f4f6"
    bg_card: str = "#ffffff"
    bg_hover: str = "#f3f4f6"
    bg_active: str = "#e5e7eb"

    # Text colors
    text_primary: str = "#111827"
    text_secondary: str = "#4b5563"
    text_muted: str = "#9ca3af"
    text_disabled: str = "#d1d5db"
    text_inverse: str = "#ffffff"

    # Border colors
    border_primary: str = "#e5e7eb"
    border_secondary: str = "#d1d5db"
    border_focus: str = "#3b82f6"

    # Shadow colors
    shadow_sm: str = "rgba(0, 0, 0, 0.05)"
    shadow_md: str = "rgba(0, 0, 0, 0.1)"
    shadow_lg: str = "rgba(0, 0, 0, 0.15)"

    # Overlay colors
    overlay: str = "rgba(0, 0, 0, 0.5)"

    # Color palette
    palette: ColorPalette = field(default_factory=ColorPalette)


@dataclass
class Theme:
    """Theme definition."""

    name: str
    mode: ThemeMode
    colors: ThemeColors
    font_family: str = "Inter, system-ui, sans-serif"
    font_family_mono: str = "JetBrains Mono, monospace"
    border_radius: str = "0.375rem"
    border_radius_sm: str = "0.25rem"
    border_radius_lg: str = "0.5rem"
    transition_duration: str = "150ms"
    sidebar_width: str = "280px"
    header_height: str = "64px"

    def to_css_variables(self) -> dict[str, str]:
        """Convert theme to CSS variables."""
        variables = {
            # Colors
            "--color-bg-primary": self.colors.bg_primary,
            "--color-bg-secondary": self.colors.bg_secondary,
            "--color-bg-tertiary": self.colors.bg_tertiary,
            "--color-bg-card": self.colors.bg_card,
            "--color-bg-hover": self.colors.bg_hover,
            "--color-bg-active": self.colors.bg_active,
            "--color-text-primary": self.colors.text_primary,
            "--color-text-secondary": self.colors.text_secondary,
            "--color-text-muted": self.colors.text_muted,
            "--color-text-disabled": self.colors.text_disabled,
            "--color-text-inverse": self.colors.text_inverse,
            "--color-border-primary": self.colors.border_primary,
            "--color-border-secondary": self.colors.border_secondary,
            "--color-border-focus": self.colors.border_focus,
            "--shadow-sm": self.colors.shadow_sm,
            "--shadow-md": self.colors.shadow_md,
            "--shadow-lg": self.colors.shadow_lg,
            "--color-overlay": self.colors.overlay,
            # Palette
            "--color-primary": self.colors.palette.primary,
            "--color-primary-light": self.colors.palette.primary_light,
            "--color-primary-dark": self.colors.palette.primary_dark,
            "--color-secondary": self.colors.palette.secondary,
            "--color-secondary-light": self.colors.palette.secondary_light,
            "--color-secondary-dark": self.colors.palette.secondary_dark,
            "--color-accent": self.colors.palette.accent,
            "--color-success": self.colors.palette.success,
            "--color-danger": self.colors.palette.danger,
            "--color-warning": self.colors.palette.warning,
            "--color-info": self.colors.palette.info,
            "--color-positive": self.colors.palette.positive,
            "--color-negative": self.colors.palette.negative,
            "--color-neutral": self.colors.palette.neutral,
            # Typography
            "--font-family": self.font_family,
            "--font-family-mono": self.font_family_mono,
            # Layout
            "--border-radius": self.border_radius,
            "--border-radius-sm": self.border_radius_sm,
            "--border-radius-lg": self.border_radius_lg,
            "--transition-duration": self.transition_duration,
            "--sidebar-width": self.sidebar_width,
            "--header-height": self.header_height,
        }

        return variables

    def to_css(self) -> str:
        """Generate CSS variables string."""
        variables = self.to_css_variables()
        lines = [f"  {key}: {value};" for key, value in variables.items()]
        return ":root {\n" + "\n".join(lines) + "\n}"


# Pre-defined themes
LIGHT_THEME = Theme(
    name="Light",
    mode=ThemeMode.LIGHT,
    colors=ThemeColors(),
)

DARK_THEME = Theme(
    name="Dark",
    mode=ThemeMode.DARK,
    colors=ThemeColors(
        bg_primary="#111827",
        bg_secondary="#1f2937",
        bg_tertiary="#374151",
        bg_card="#1f2937",
        bg_hover="#374151",
        bg_active="#4b5563",
        text_primary="#f9fafb",
        text_secondary="#d1d5db",
        text_muted="#9ca3af",
        text_disabled="#6b7280",
        text_inverse="#111827",
        border_primary="#374151",
        border_secondary="#4b5563",
        border_focus="#3b82f6",
        shadow_sm="rgba(0, 0, 0, 0.2)",
        shadow_md="rgba(0, 0, 0, 0.3)",
        shadow_lg="rgba(0, 0, 0, 0.4)",
        overlay="rgba(0, 0, 0, 0.7)",
    ),
)

# Alternative themes
NORD_THEME = Theme(
    name="Nord",
    mode=ThemeMode.DARK,
    colors=ThemeColors(
        bg_primary="#2e3440",
        bg_secondary="#3b4252",
        bg_tertiary="#434c5e",
        bg_card="#3b4252",
        bg_hover="#434c5e",
        bg_active="#4c566a",
        text_primary="#eceff4",
        text_secondary="#d8dee9",
        text_muted="#81a1c1",
        text_disabled="#4c566a",
        text_inverse="#2e3440",
        border_primary="#434c5e",
        border_secondary="#4c566a",
        border_focus="#88c0d0",
        palette=ColorPalette(
            primary="#88c0d0",
            primary_light="#8fbcbb",
            primary_dark="#5e81ac",
            secondary="#81a1c1",
            success="#a3be8c",
            danger="#bf616a",
            warning="#ebcb8b",
            info="#88c0d0",
            positive="#a3be8c",
            negative="#bf616a",
        ),
    ),
)


class ThemeManager:
    """
    Theme manager.

    Handles theme loading, switching, and user preferences.
    """

    def __init__(self) -> None:
        """Initialize theme manager."""
        self._themes: dict[str, Theme] = {
            "light": LIGHT_THEME,
            "dark": DARK_THEME,
            "nord": NORD_THEME,
        }
        self._default_theme = "dark"
        self._user_themes: dict[str, str] = {}

        logger.info("ThemeManager initialized")

    def get_theme(self, name: str) -> Theme | None:
        """Get theme by name."""
        return self._themes.get(name)

    def get_default_theme(self) -> Theme:
        """Get default theme."""
        return self._themes[self._default_theme]

    def get_all_themes(self) -> list[Theme]:
        """Get all available themes."""
        return list(self._themes.values())

    def get_theme_names(self) -> list[str]:
        """Get all theme names."""
        return list(self._themes.keys())

    def register_theme(self, theme: Theme) -> None:
        """Register a custom theme."""
        theme_key = theme.name.lower().replace(" ", "_")
        self._themes[theme_key] = theme
        logger.info(f"Theme registered: {theme.name}")

    def set_default_theme(self, name: str) -> bool:
        """Set default theme."""
        if name in self._themes:
            self._default_theme = name
            return True
        return False

    def get_user_theme(self, user_id: str) -> Theme:
        """Get theme for user."""
        theme_name = self._user_themes.get(user_id, self._default_theme)
        return self._themes.get(theme_name, self.get_default_theme())

    def set_user_theme(self, user_id: str, theme_name: str) -> bool:
        """Set theme for user."""
        if theme_name in self._themes:
            self._user_themes[user_id] = theme_name
            return True
        return False

    def get_css(self, theme_name: str | None = None) -> str:
        """Get CSS for theme."""
        theme = self.get_theme(theme_name) if theme_name else self.get_default_theme()
        if theme:
            return theme.to_css()
        return ""

    def get_css_variables(self, theme_name: str | None = None) -> dict[str, str]:
        """Get CSS variables for theme."""
        theme = self.get_theme(theme_name) if theme_name else self.get_default_theme()
        if theme:
            return theme.to_css_variables()
        return {}

    def create_custom_theme(
        self,
        name: str,
        base_theme: str = "dark",
        **overrides: Any,
    ) -> Theme:
        """
        Create a custom theme based on existing theme.

        Args:
            name: Theme name
            base_theme: Base theme to extend
            **overrides: Color/property overrides

        Returns:
            New Theme instance
        """
        base = self._themes.get(base_theme, DARK_THEME)

        # Create new colors with overrides
        colors_dict = {
            "bg_primary": base.colors.bg_primary,
            "bg_secondary": base.colors.bg_secondary,
            "bg_tertiary": base.colors.bg_tertiary,
            "bg_card": base.colors.bg_card,
            "text_primary": base.colors.text_primary,
            "text_secondary": base.colors.text_secondary,
            "text_muted": base.colors.text_muted,
            "border_primary": base.colors.border_primary,
        }

        # Apply overrides
        for key, value in overrides.items():
            if key in colors_dict:
                colors_dict[key] = value

        colors = ThemeColors(**colors_dict)

        theme = Theme(
            name=name,
            mode=base.mode,
            colors=colors,
            font_family=overrides.get("font_family", base.font_family),
            border_radius=overrides.get("border_radius", base.border_radius),
        )

        self.register_theme(theme)
        return theme


def get_theme_manager() -> ThemeManager:
    """Get global theme manager instance."""
    global _theme_manager
    if "_theme_manager" not in globals():
        _theme_manager = ThemeManager()
    return _theme_manager


def get_theme_css(theme_name: str = "dark") -> str:
    """Get CSS for specified theme."""
    return get_theme_manager().get_css(theme_name)


def get_available_themes() -> list[dict[str, str]]:
    """Get list of available themes."""
    manager = get_theme_manager()
    return [
        {"name": theme.name, "mode": theme.mode.value}
        for theme in manager.get_all_themes()
    ]


# Module version
__version__ = "2.2.0"

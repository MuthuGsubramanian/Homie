package com.heyhomie.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val RetroColorScheme = darkColorScheme(
    primary = RetroGreen,
    secondary = RetroAmber,
    tertiary = RetroCyan,
    background = RetroDark,
    surface = RetroDarkSurface,
    surfaceVariant = RetroDarkCard,
    onPrimary = RetroDark,
    onSecondary = RetroDark,
    onTertiary = RetroDark,
    onBackground = RetroWhite,
    onSurface = RetroWhite,
    error = RetroRed,
    onError = RetroDark
)

@Composable
fun HomieRetroTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = RetroColorScheme,
        typography = RetroTypography,
        content = content
    )
}

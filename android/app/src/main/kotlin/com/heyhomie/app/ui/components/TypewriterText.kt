package com.heyhomie.app.ui.components

import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import com.heyhomie.app.ui.theme.RetroGreen
import com.heyhomie.app.ui.theme.RetroTypography
import kotlinx.coroutines.delay

@Composable
fun TypewriterText(
    fullText: String,
    modifier: Modifier = Modifier,
    style: TextStyle = RetroTypography.bodyMedium,
    color: Color = RetroGreen,
    charDelayMs: Long = 20L,
    onComplete: () -> Unit = {}
) {
    var visibleCount by remember(fullText) { mutableIntStateOf(0) }

    LaunchedEffect(fullText) {
        visibleCount = 0
        for (i in fullText.indices) {
            delay(charDelayMs)
            visibleCount = i + 1
        }
        onComplete()
    }

    Text(
        text = fullText.take(visibleCount) + if (visibleCount < fullText.length) "\u2588" else "",
        style = style,
        color = color,
        modifier = modifier
    )
}

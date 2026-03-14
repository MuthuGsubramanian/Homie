package com.heyhomie.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.RetroDarkCard
import com.heyhomie.app.ui.theme.RetroGreen

@Composable
fun RetroProgressBar(
    progress: Float,
    modifier: Modifier = Modifier,
    color: Color = RetroGreen,
    backgroundColor: Color = RetroDarkCard,
    label: String? = null
) {
    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(20.dp)
            .pixelBorder(color = color, width = 1f)
    ) {
        // Background
        drawRect(backgroundColor, size = size)

        // Filled blocks (pixel-style segments)
        val blockWidth = 8f * density
        val gap = 2f * density
        val filledWidth = size.width * progress.coerceIn(0f, 1f)
        var x = gap
        while (x + blockWidth < filledWidth) {
            drawRect(
                color,
                topLeft = Offset(x, gap),
                size = Size(blockWidth, size.height - gap * 2)
            )
            x += blockWidth + gap
        }
    }
}

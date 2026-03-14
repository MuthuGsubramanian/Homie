package com.heyhomie.app.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color

@Composable
fun ScanlineOverlay(
    modifier: Modifier = Modifier,
    lineSpacing: Float = 4f,
    alpha: Float = 0.08f
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        var y = 0f
        while (y < size.height) {
            drawRect(
                color = Color.Black.copy(alpha = alpha),
                topLeft = Offset(0f, y),
                size = Size(size.width, lineSpacing / 2)
            )
            y += lineSpacing * density
        }
    }
}

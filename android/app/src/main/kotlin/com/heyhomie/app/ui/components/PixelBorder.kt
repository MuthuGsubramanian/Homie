package com.heyhomie.app.ui.components

import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import com.heyhomie.app.ui.theme.RetroGreen

data class PixelBorderConfig(
    val stepSize: Float = 4f,
    val borderColor: Long = 0xFF39FF14,
    val borderWidth: Float = 2f
)

fun Modifier.pixelBorder(
    color: Color = RetroGreen,
    stepSize: Float = 4f,
    width: Float = 2f
): Modifier = this.drawBehind {
    val w = size.width
    val h = size.height
    val step = stepSize * density

    // Top edge
    var x = 0f
    while (x < w) {
        drawRect(color, Offset(x, 0f), Size(minOf(step, w - x), width * density))
        x += step
    }
    // Bottom edge
    x = 0f
    while (x < w) {
        drawRect(color, Offset(x, h - width * density), Size(minOf(step, w - x), width * density))
        x += step
    }
    // Left edge
    var y = 0f
    while (y < h) {
        drawRect(color, Offset(0f, y), Size(width * density, minOf(step, h - y)))
        y += step
    }
    // Right edge
    y = 0f
    while (y < h) {
        drawRect(color, Offset(w - width * density, y), Size(width * density, minOf(step, h - y)))
        y += step
    }
    // Corner notches (staircase effect)
    val notch = step
    drawRect(Color.Transparent, Offset(0f, 0f), Size(notch, notch))
    drawRect(color, Offset(notch, 0f), Size(width * density, notch))
    drawRect(color, Offset(0f, notch), Size(notch, width * density))
}

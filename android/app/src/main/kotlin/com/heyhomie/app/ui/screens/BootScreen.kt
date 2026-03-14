package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.components.ScanlineOverlay
import com.heyhomie.app.ui.theme.*
import kotlinx.coroutines.delay

private val bootLines = listOf(
    "HOMIE AI v0.1.0",
    "==================",
    "Initializing neural core...",
    "Loading memory banks...",
    "Scanning local models...",
    "Calibrating personality matrix...",
    "Connecting to reality...",
    "",
    "> SYSTEM READY",
    "> Hello, friend."
)

@Composable
fun BootScreen(onBootComplete: () -> Unit) {
    var visibleLines by remember { mutableIntStateOf(0) }
    var cursorVisible by remember { mutableStateOf(true) }

    // Blink cursor
    LaunchedEffect(Unit) {
        while (true) {
            delay(500)
            cursorVisible = !cursorVisible
        }
    }

    // Reveal lines one by one
    LaunchedEffect(Unit) {
        for (i in bootLines.indices) {
            delay(if (i < 2) 200L else 400L)
            visibleLines = i + 1
        }
        delay(1000)
        onBootComplete()
    }

    Box(
        Modifier
            .fillMaxSize()
            .background(RetroDark)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center
        ) {
            bootLines.take(visibleLines).forEachIndexed { index, line ->
                val color = when {
                    index == 0 -> RetroGreen
                    line.startsWith(">") -> RetroAmber
                    line.startsWith("=") -> RetroGreen
                    else -> RetroCyan
                }
                Text(
                    text = line,
                    style = RetroTypography.labelMedium,
                    color = color
                )
                Spacer(Modifier.height(4.dp))
            }

            if (visibleLines > 0 && visibleLines <= bootLines.size) {
                Text(
                    text = if (cursorVisible) "\u2588" else " ",
                    style = RetroTypography.labelMedium,
                    color = RetroGreen
                )
            }
        }

        ScanlineOverlay(alpha = 0.05f)
    }
}

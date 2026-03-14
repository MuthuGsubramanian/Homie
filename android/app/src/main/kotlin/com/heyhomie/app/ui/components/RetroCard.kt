package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.RetroDarkCard
import com.heyhomie.app.ui.theme.RetroGreen

@Composable
fun RetroCard(
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Box(
        modifier = modifier
            .pixelBorder(color = RetroGreen)
            .crtGlow()
            .background(RetroDarkCard)
            .padding(12.dp)
    ) {
        content()
    }
}

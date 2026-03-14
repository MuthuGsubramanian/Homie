package com.heyhomie.app.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.heyhomie.app.ui.theme.RetroDark
import com.heyhomie.app.ui.theme.RetroTypography

@Composable
fun ChatScreen() {
    Box(Modifier.fillMaxSize().background(RetroDark), contentAlignment = Alignment.Center) {
        Text("CHAT", style = RetroTypography.headlineMedium)
    }
}

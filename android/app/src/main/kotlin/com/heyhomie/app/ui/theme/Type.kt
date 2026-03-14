package com.heyhomie.app.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.heyhomie.app.R

val PressStart2P = FontFamily(
    Font(R.font.press_start_2p, FontWeight.Normal)
)

val JetBrainsMono = FontFamily(
    Font(R.font.jetbrains_mono_regular, FontWeight.Normal),
    Font(R.font.jetbrains_mono_bold, FontWeight.Bold)
)

val RetroTypography = Typography(
    displayLarge = TextStyle(
        fontFamily = PressStart2P, fontSize = 24.sp, color = RetroGreen
    ),
    headlineMedium = TextStyle(
        fontFamily = PressStart2P, fontSize = 16.sp, color = RetroGreen
    ),
    titleMedium = TextStyle(
        fontFamily = PressStart2P, fontSize = 12.sp, color = RetroAmber
    ),
    bodyLarge = TextStyle(
        fontFamily = JetBrainsMono, fontSize = 16.sp, color = RetroWhite
    ),
    bodyMedium = TextStyle(
        fontFamily = JetBrainsMono, fontSize = 14.sp, color = RetroWhite
    ),
    labelMedium = TextStyle(
        fontFamily = PressStart2P, fontSize = 10.sp, color = RetroCyan
    )
)

package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole
import com.heyhomie.app.ui.theme.*

@Composable
fun ChatBubble(
    message: ChatMessage,
    modifier: Modifier = Modifier
) {
    val isUser = message.role == MessageRole.USER
    val alignment = if (isUser) Alignment.CenterEnd else Alignment.CenterStart
    val bgColor = if (isUser) RetroDimGreen else RetroDarkCard
    val textColor = if (isUser) RetroWhite else RetroGreen
    val prefix = if (isUser) "> " else "HOMIE> "

    Box(
        modifier = modifier.fillMaxWidth(),
        contentAlignment = alignment
    ) {
        Column(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .let { if (!isUser) it.crtGlow(RetroGreen, 8f) else it }
                .pixelBorder(color = if (isUser) RetroAmber else RetroGreen, width = 1f)
                .background(bgColor)
                .padding(10.dp)
        ) {
            if (!isUser) {
                Text("\uD83E\uDD16", style = RetroTypography.labelMedium)
                Spacer(Modifier.height(4.dp))
            }

            if (!isUser && message.isStreaming) {
                TypewriterText(
                    fullText = message.text,
                    color = textColor
                )
            } else {
                Text(
                    text = "$prefix${message.text}",
                    style = RetroTypography.bodyMedium,
                    color = textColor
                )
            }
        }
    }
}

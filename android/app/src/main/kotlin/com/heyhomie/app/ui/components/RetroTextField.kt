package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.theme.*

@Composable
fun RetroTextField(
    value: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .pixelBorder(color = RetroGreen, width = 1f)
            .background(RetroDark)
            .padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("> ", style = RetroTypography.bodyMedium, color = RetroGreen)
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            enabled = enabled,
            textStyle = RetroTypography.bodyMedium.copy(color = RetroWhite),
            cursorBrush = SolidColor(RetroGreen),
            modifier = Modifier.weight(1f),
            singleLine = true
        )
        Text(
            text = "[SEND]",
            style = RetroTypography.labelMedium,
            color = if (value.isNotBlank()) RetroAmber else RetroGray,
            modifier = Modifier
                .padding(start = 8.dp)
                .clickable(enabled = value.isNotBlank()) { onSend() }
                .then(
                    if (value.isNotBlank()) Modifier.pixelBorder(RetroAmber, width = 1f)
                    else Modifier
                )
                .padding(4.dp)
        )
    }
}

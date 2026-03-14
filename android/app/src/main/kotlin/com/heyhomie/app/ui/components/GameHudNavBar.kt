package com.heyhomie.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.heyhomie.app.ui.navigation.Screen
import com.heyhomie.app.ui.navigation.bottomNavScreens
import com.heyhomie.app.ui.theme.*

@Composable
fun GameHudNavBar(
    currentRoute: String?,
    onNavigate: (Screen) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .background(RetroDark)
            .pixelBorder(color = RetroGreen, width = 1f)
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        bottomNavScreens.forEach { screen ->
            val selected = currentRoute == screen.route
            val color = if (selected) RetroGreen else RetroGray
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .clickable { onNavigate(screen) }
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(
                    text = screen.icon,
                    style = RetroTypography.bodyLarge,
                    color = color
                )
                Text(
                    text = screen.label,
                    style = RetroTypography.labelMedium,
                    color = color,
                    textAlign = TextAlign.Center
                )
                if (selected) {
                    Spacer(Modifier.height(2.dp))
                    Box(
                        Modifier
                            .width(24.dp)
                            .height(2.dp)
                            .background(RetroGreen)
                    )
                }
            }
        }
    }
}

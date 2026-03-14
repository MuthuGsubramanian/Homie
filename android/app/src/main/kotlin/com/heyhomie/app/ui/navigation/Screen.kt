package com.heyhomie.app.ui.navigation

sealed class Screen(val route: String, val label: String, val icon: String) {
    data object Chat : Screen("chat", "CHAT", "\uD83D\uDCAC")
    data object PhoneStats : Screen("phone_stats", "STATS", "\uD83D\uDCF1")
    data object Network : Screen("network", "LAN", "\uD83C\uDF10")
    data object Settings : Screen("settings", "CONFIG", "\u2699\uFE0F")
}

val bottomNavScreens = listOf(Screen.Chat, Screen.PhoneStats, Screen.Network, Screen.Settings)

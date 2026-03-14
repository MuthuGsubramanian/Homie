package com.heyhomie.app.network

data class PeerDevice(
    val name: String,
    val host: String,
    val port: Int,
    val deviceId: String
) {
    val wsUrl: String get() = "ws://$host:$port"
}

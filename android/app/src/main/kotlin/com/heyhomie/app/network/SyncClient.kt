package com.heyhomie.app.network

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class SyncClient @Inject constructor() {

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val _connectionState = MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _incomingMessages = MutableStateFlow<SyncMessage?>(null)
    val incomingMessages: StateFlow<SyncMessage?> = _incomingMessages

    var deviceId: String = "android-${System.currentTimeMillis()}"

    fun connect(peer: PeerDevice) {
        _connectionState.value = ConnectionState.CONNECTING
        val request = Request.Builder().url(peer.wsUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                _connectionState.value = ConnectionState.CONNECTED
                val hello = SyncMessage.hello(deviceId, "1.0.0", "Android")
                ws.send(hello.toJson().toString())
            }

            override fun onMessage(ws: WebSocket, text: String) {
                val msg = SyncMessage.fromJson(JSONObject(text))
                _incomingMessages.value = msg
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                _connectionState.value = ConnectionState.DISCONNECTED
                scope.launch {
                    delay(5000)
                    connect(peer)
                }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                _connectionState.value = ConnectionState.DISCONNECTED
            }
        })
    }

    fun send(message: SyncMessage) {
        webSocket?.send(message.toJson().toString())
    }

    fun disconnect() {
        webSocket?.close(1000, "User disconnect")
        webSocket = null
        _connectionState.value = ConnectionState.DISCONNECTED
    }
}

enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED }

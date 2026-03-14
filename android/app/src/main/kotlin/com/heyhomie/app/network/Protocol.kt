package com.heyhomie.app.network

import org.json.JSONObject

data class SyncMessage(
    val type: String,
    val protocolVersion: String = "1.0.0",
    val deviceId: String = "",
    val data: JSONObject = JSONObject()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("type", type)
        put("protocol_version", protocolVersion)
        put("device_id", deviceId)
        data.keys().forEach { key -> put(key, data.get(key)) }
    }

    companion object {
        fun fromJson(json: JSONObject): SyncMessage = SyncMessage(
            type = json.getString("type"),
            protocolVersion = json.optString("protocol_version", "1.0.0"),
            deviceId = json.optString("device_id", ""),
            data = json
        )

        fun hello(deviceId: String, version: String, name: String) = SyncMessage(
            type = "hello", protocolVersion = version, deviceId = deviceId,
            data = JSONObject().put("device_name", name)
        )

        fun inferenceRequest(prompt: String, systemPrompt: String? = null) = SyncMessage(
            type = "inference_request",
            data = JSONObject().apply {
                put("prompt", prompt)
                systemPrompt?.let { put("system_prompt", it) }
            }
        )

        fun memorySyncRequest(since: Long) = SyncMessage(
            type = "memory_sync",
            data = JSONObject().put("since", since)
        )
    }
}

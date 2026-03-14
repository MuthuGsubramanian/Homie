package com.heyhomie.app.core.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

@Entity(tableName = "messages")
data class MessageEntity(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val conversationId: String,
    val role: String,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val deviceId: String = ""
)

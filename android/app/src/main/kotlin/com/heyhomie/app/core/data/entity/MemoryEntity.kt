package com.heyhomie.app.core.data.entity

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.UUID

@Entity(tableName = "memories")
data class MemoryEntity(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val type: String,
    val content: String,
    val contentHash: String = "",
    val deviceId: String,
    val timestamp: Long = System.currentTimeMillis(),
    val tombstone: Boolean = false,
    val lamportClock: Long = 0
)

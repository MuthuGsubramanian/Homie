package com.heyhomie.app.core.data

import androidx.room.Database
import androidx.room.RoomDatabase
import com.heyhomie.app.core.data.dao.MessageDao
import com.heyhomie.app.core.data.dao.MemoryDao
import com.heyhomie.app.core.data.entity.MessageEntity
import com.heyhomie.app.core.data.entity.MemoryEntity

@Database(
    entities = [MessageEntity::class, MemoryEntity::class],
    version = 1,
    exportSchema = false
)
abstract class HomieDatabase : RoomDatabase() {
    abstract fun messageDao(): MessageDao
    abstract fun memoryDao(): MemoryDao
}

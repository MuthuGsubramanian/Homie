package com.heyhomie.app.core.data.dao

import androidx.room.*
import com.heyhomie.app.core.data.entity.MemoryEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MemoryDao {
    @Query("SELECT * FROM memories WHERE type = :type AND tombstone = 0 ORDER BY timestamp DESC")
    fun getByType(type: String): Flow<List<MemoryEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(memory: MemoryEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(memories: List<MemoryEntity>)

    @Query("UPDATE memories SET tombstone = 1 WHERE id = :id")
    suspend fun markTombstone(id: String)

    @Query("SELECT * FROM memories WHERE timestamp > :since AND tombstone = 0")
    suspend fun getNewerThan(since: Long): List<MemoryEntity>
}

package com.heyhomie.app.core.data.di

import android.content.Context
import androidx.room.Room
import com.heyhomie.app.core.data.HomieDatabase
import com.heyhomie.app.core.data.dao.MessageDao
import com.heyhomie.app.core.data.dao.MemoryDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    @Provides @Singleton
    fun provideDatabase(@ApplicationContext ctx: Context): HomieDatabase =
        Room.databaseBuilder(ctx, HomieDatabase::class.java, "homie.db").build()

    @Provides fun provideMessageDao(db: HomieDatabase): MessageDao = db.messageDao()
    @Provides fun provideMemoryDao(db: HomieDatabase): MemoryDao = db.memoryDao()
}

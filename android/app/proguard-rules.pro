# Homie AI ProGuard Rules

# Keep Hilt-generated code
-keep class dagger.hilt.** { *; }
-keep class * extends dagger.hilt.android.internal.managers.ViewComponentManager$FragmentContextWrapper { *; }

# Keep Room entities
-keep class com.heyhomie.app.core.data.entity.** { *; }

# Keep Kotlin data classes used in JSON serialization
-keepclassmembers class com.heyhomie.app.network.SyncMessage { *; }
-keepclassmembers class com.heyhomie.app.network.PeerDevice { *; }
-keepclassmembers class com.heyhomie.app.core.inference.QubridClient { *; }

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-keep class okhttp3.** { *; }

# Keep notification listener
-keep class com.heyhomie.app.notifications.HomieNotificationListener { *; }

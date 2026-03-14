package com.heyhomie.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import com.heyhomie.app.core.model.ChatMessage
import com.heyhomie.app.core.model.MessageRole

class ChatViewModel : ViewModel() {
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    fun updateInput(text: String) {
        _inputText.value = text
    }

    fun sendMessage(text: String) {
        if (text.isBlank()) return
        val userMsg = ChatMessage(role = MessageRole.USER, text = text.trim())
        _messages.value = _messages.value + userMsg
        _inputText.value = ""
        // Placeholder response — will be wired to InferenceRouter in Task 18
        val placeholder = ChatMessage(
            role = MessageRole.ASSISTANT,
            text = "I'm Homie! Inference not connected yet. Stay tuned, friend.",
            isStreaming = true
        )
        _messages.value = _messages.value + placeholder
    }
}

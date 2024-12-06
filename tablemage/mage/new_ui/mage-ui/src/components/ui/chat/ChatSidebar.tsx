'use client'

import { useState } from 'react'
import { Send, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { sendChatMessage } from '@/lib/api'  // Import the chat API function

interface Message {
  type: 'user' | 'system'
  content: string
}
interface ChatSidebarProps {
    onAnalysisUpdated?: () => void;
  }
  
export function ChatSidebar({ onAnalysisUpdated }: ChatSidebarProps) {
  const [messages, setMessages] = useState<Message[]>([
    { type: 'system', content: 'Welcome! Upload your dataset to begin analysis. I can help guide you through the process.' }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const suggestionChips = [
    "What does the skewness in Age indicate?",
    "How should I interpret these statistics?",
    "What preprocessing steps do you recommend?",
    "Should I handle missing values?",
  ]

  const handleSendMessage = async () => {
    const message = inputMessage.trim()
    if (!message) return
    
    try {
      setIsLoading(true)
      // Add user message to chat
      setMessages(prev => [...prev, { type: 'user', content: message }])
      setInputMessage('') // Clear input immediately after sending

      // Send message to API
      const response = await sendChatMessage(message)

      if (response.hasAnalysis) {
        window.dispatchEvent(new CustomEvent('refreshAnalysis'))
        onAnalysisUpdated?.()  // Call the prop if provided
      }


      // Add system response to chat
      setMessages(prev => [...prev, { 
        type: 'system', 
        content: response.response 
      }])

      // If new analysis was generated, trigger refresh
      if (response.hasAnalysis) {
        // You'll need to implement this function to refresh the analysis panel
        // window.dispatchEvent(new CustomEvent('refreshAnalysis'))
      }

    } catch (error) {
      // Add error message to chat
      setMessages(prev => [...prev, { 
        type: 'system', 
        content: 'Sorry, I encountered an error processing your request. Please try again.' 
      }])
      console.error('Chat error:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="w-96 border-l bg-gray-50 flex flex-col">
      {/* Chat Messages */}
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`mb-4 ${
              message.type === 'user' ? 'flex justify-end' : ''
            }`}
          >
            <div
              className={`rounded-lg p-3 max-w-[80%] ${
                message.type === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white border'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
      </div>

      {/* Suggestion Chips */}
      <div className="p-4 border-t bg-white">
        <p className="text-sm text-gray-600 mb-2">Suggested questions:</p>
        <div className="flex flex-wrap gap-2">
          {suggestionChips.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => setInputMessage(suggestion)}
              className="bg-gray-100 hover:bg-gray-200 rounded-full px-3 py-1 text-sm text-gray-700 flex items-center"
            >
              {suggestion}
              <ChevronRight className="h-4 w-4 ml-1" />
            </button>
          ))}
        </div>
      </div>

      {/* Chat Input */}
      <div className="p-4 border-t bg-white">
        <div className="flex items-center space-x-2">
          <Input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
            placeholder="Ask a question about your dataset..."
            className="flex-1"
            disabled={isLoading}
          />
          <Button 
            size="icon" 
            onClick={handleSendMessage}
            disabled={isLoading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
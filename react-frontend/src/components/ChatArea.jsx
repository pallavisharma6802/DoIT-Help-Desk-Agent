import { useRef, useEffect, useState } from 'react'
import Message from './Message'
import './ChatArea.css'

const PLACEHOLDER = {
  student: 'Ask a UW-Madison IT question…',
  agent: 'Describe the issue you\'re troubleshooting…',
}

const SUGGESTIONS = {
  student: [
    'How do I connect to UWNet?',
    'I can\'t log into my NetID account',
    'How do I set up Duo two-factor authentication?',
    'Where can I get Office 365 for free?',
  ],
  agent: [
    'NetID locked — what\'s the reset procedure?',
    'UWNet not connecting on Mac',
    'Student can\'t access Canvas course',
  ],
}

export default function ChatArea({ mode, messages, loading, onSend, sessionId }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e) => {
    e.preventDefault()
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    onSend(q)
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <main className="chat-area">
      <div className="chat-messages">
        {isEmpty && !loading && (
          <div className="chat-empty">
            <div className="chat-empty-icon-wrap">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <path d="M6 16C6 10.477 10.477 6 16 6s10 4.477 10 10-4.477 10-10 10H6l3-3" stroke="var(--uw-red)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="12" cy="16" r="1.2" fill="var(--uw-red)"/>
                <circle cx="16" cy="16" r="1.2" fill="var(--uw-red)"/>
                <circle cx="20" cy="16" r="1.2" fill="var(--uw-red)"/>
              </svg>
            </div>
            <p className="chat-empty-title">
              {mode === 'agent' ? 'Agent mode active' : 'How can I help you today?'}
            </p>
            <p className="chat-empty-sub">
              {mode === 'agent'
                ? 'Describe an issue and get real-time KB results with streaming.'
                : 'Ask any UW-Madison IT support question and get answers backed by the DoIT Knowledge Base.'}
            </p>
            <div className="chat-empty-suggestions">
              {SUGGESTIONS[mode].map(s => (
                <button
                  key={s}
                  className="suggestion-chip"
                  onClick={() => onSend(s)}
                  disabled={loading}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <Message key={msg.id} message={msg} mode={mode} />
        ))}

        {loading && mode === 'student' && (
          <div className="chat-loading">
            <div className="chat-loading-avatar">AI</div>
            <div className="chat-loading-bubble">
              <span className="dot-pulse" />
              <span className="dot-pulse dot-pulse--2" />
              <span className="dot-pulse dot-pulse--3" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={PLACEHOLDER[mode]}
            rows={1}
            disabled={loading}
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={!input.trim() || loading}
            aria-label="Send"
          >
            <svg width="17" height="17" viewBox="0 0 17 17" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="8.5" y1="14" x2="8.5" y2="3" />
              <polyline points="4 7.5 8.5 3 13 7.5" />
            </svg>
          </button>
        </form>
        <p className="chat-hint">Enter to send · Shift+Enter for new line</p>
      </div>
    </main>
  )
}

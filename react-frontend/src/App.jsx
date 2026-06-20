import { useState, useEffect, useCallback } from 'react'
import { generateSessionId } from './utils/session'
import { useMode } from './hooks/useMode'
import { useCategories } from './hooks/useCategories'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import Toast from './components/Toast'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function App() {
  const [mode, setMode] = useMode()
  const [sessionId, setSessionId] = useState(generateSessionId)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [toast, setToast] = useState(null)
  const categories = useCategories(API_BASE)

  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 3500)
  }, [])

  const newSession = useCallback(() => {
    setSessionId(generateSessionId())
    setMessages([])
    setSidebarOpen(false)
  }, [])

  const endSession = useCallback(async () => {
    if (messages.length === 0) {
      showToast('No messages to log.', 'info')
      return
    }
    try {
      const res = await fetch(`${API_BASE}/end-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      showToast('Session logged successfully.')
    } catch (e) {
      showToast(`Failed to log session: ${e.message}`, 'error')
    }
    setSessionId(generateSessionId())
    setMessages([])
    setSidebarOpen(false)
  }, [sessionId, messages, showToast])

  const sendStudentMessage = useCallback(async (query) => {
    setLoading(true)
    const userMsg = { role: 'user', content: query, id: crypto.randomUUID() }
    setMessages(prev => [...prev, userMsg])

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId, user_type: 'student' }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      const assistantMsg = {
        role: 'assistant',
        id: crypto.randomUUID(),
        content: data.answer,
        citations: data.kb_citations || [],
        meta: {
          turn: data.turn,
          resolved: data.resolved,
          escalated: data.escalated,
          complexity: data.complexity,
        },
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (e) {
      const errMsg = {
        role: 'assistant',
        id: crypto.randomUUID(),
        content: null,
        error: `Request failed: ${e.message}`,
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  const sendAgentMessage = useCallback(async (query) => {
    setLoading(true)
    const userMsg = { role: 'user', content: query, id: crypto.randomUUID() }

    const placeholderId = crypto.randomUUID()
    const placeholder = {
      role: 'assistant',
      id: placeholderId,
      content: '',
      streaming: true,
      ttft: null,
    }
    setMessages(prev => [...prev, userMsg, placeholder])

    const t0 = performance.now()
    let firstTokenMs = null
    let fullText = ''

    try {
      const res = await fetch(`${API_BASE}/agent-assist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId, user_type: 'agent' }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const chunk = line.slice(6)
          if (chunk === '[DONE]') break
          if (chunk.startsWith('[ERROR]')) {
            setMessages(prev => prev.map(m =>
              m.id === placeholderId
                ? { ...m, streaming: false, error: chunk.slice(8), content: fullText }
                : m
            ))
            setLoading(false)
            return
          }
          if (firstTokenMs === null) {
            firstTokenMs = performance.now() - t0
          }
          fullText += chunk
          setMessages(prev => prev.map(m =>
            m.id === placeholderId ? { ...m, content: fullText } : m
          ))
        }
      }

      setMessages(prev => prev.map(m =>
        m.id === placeholderId
          ? { ...m, streaming: false, ttft: firstTokenMs }
          : m
      ))
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === placeholderId
          ? { ...m, streaming: false, error: `Request failed: ${e.message}`, content: fullText }
          : m
      ))
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  const handleSend = useCallback((query) => {
    if (mode === 'agent') {
      sendAgentMessage(query)
    } else {
      sendStudentMessage(query)
    }
  }, [mode, sendAgentMessage, sendStudentMessage])

  return (
    <div className="app-shell">
      <Header
        mode={mode}
        onMenuToggle={() => setSidebarOpen(o => !o)}
      />

      <div className="app-body">
        <Sidebar
          mode={mode}
          setMode={setMode}
          sessionId={sessionId}
          onNewSession={newSession}
          onEndSession={endSession}
          categories={categories}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />

        {sidebarOpen && (
          <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
        )}

        <ChatArea
          mode={mode}
          messages={messages}
          loading={loading}
          onSend={handleSend}
          sessionId={sessionId}
        />
      </div>

      {toast && <Toast message={toast.message} type={toast.type} />}
    </div>
  )
}

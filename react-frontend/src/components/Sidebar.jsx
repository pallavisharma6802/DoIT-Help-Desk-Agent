import './Sidebar.css'

export default function Sidebar({
  mode, setMode, sessionId, onNewSession, onEndSession,
  categories, open, onClose
}) {
  return (
    <aside className={`sidebar ${open ? 'sidebar--open' : ''}`}>
      <div className="sidebar-header">
        <div className="sidebar-wordmark">
          <div className="sidebar-wordmark-dot" />
          <span className="sidebar-logo-text">Controls</span>
        </div>
        <button className="sidebar-close" onClick={onClose} aria-label="Close sidebar">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
            <line x1="2" y1="2" x2="14" y2="14" />
            <line x1="14" y1="2" x2="2" y2="14" />
          </svg>
        </button>
      </div>

      <div className="sidebar-section">
        <p className="sidebar-label">Mode</p>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'student' ? 'mode-btn--active' : ''}`}
            onClick={() => { setMode('student'); onClose(); }}
          >
            Student
          </button>
          <button
            className={`mode-btn ${mode === 'agent' ? 'mode-btn--active' : ''}`}
            onClick={() => { setMode('agent'); onClose(); }}
          >
            Agent
          </button>
        </div>
      </div>

      <div className="sidebar-divider" />

      <div className="sidebar-section">
        <p className="sidebar-label">Session</p>
        <div className="session-id-box">
          <span className="session-id-label">ID</span>
          <code className="session-id-value">{sessionId.slice(0, 8)}</code>
        </div>
        <button className="sidebar-btn sidebar-btn--primary" onClick={onEndSession}>
          End &amp; Log Session
        </button>
        <button className="sidebar-btn sidebar-btn--secondary" onClick={onNewSession}>
          New Session
        </button>
      </div>

      {categories && categories.length > 0 && (
        <>
          <div className="sidebar-divider" />
          <div className="sidebar-section sidebar-section--categories">
            <p className="sidebar-label">KB Categories</p>
            <ul className="category-list">
              {categories.map(cat => (
                <li key={cat.category} className="category-item">
                  <span className="category-name">{cat.category}</span>
                  <span className="category-count">{cat.count}</span>
                </li>
              ))}
            </ul>
            <p className="category-total">
              {categories.reduce((s, c) => s + c.count, 0)} articles total
            </p>
          </div>
        </>
      )}
    </aside>
  )
}

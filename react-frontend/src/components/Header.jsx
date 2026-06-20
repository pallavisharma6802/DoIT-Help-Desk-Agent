import './Header.css'

export default function Header({ mode, onMenuToggle }) {
  const isAgent = mode === 'agent'

  return (
    <header className="header">
      <div className="header-inner">
        <button className="menu-btn" onClick={onMenuToggle} aria-label="Toggle sidebar">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="2" y1="5" x2="16" y2="5" />
            <line x1="2" y1="9" x2="16" y2="9" />
            <line x1="2" y1="13" x2="16" y2="13" />
          </svg>
        </button>

        <div className="header-brand">
          <div className="header-logo">W</div>
          <div className="header-titles">
            <span className="header-title">DoIT HD Agentic Assistant</span>
            <span className="header-subtitle">
              {isAgent
                ? 'Real-time KB lookup during live calls'
                : 'Ask any UW-Madison IT question — powered by the Knowledge Base'}
            </span>
          </div>
        </div>

        <div className="header-right">
          <span className="header-pill">UW–Madison</span>
          {isAgent && <span className="agent-badge">AGENT MODE</span>}
        </div>
      </div>
    </header>
  )
}

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './Message.css'

function MD({ children }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ node, ...props }) => <p className="md-p" {...props} />,
        strong: ({ node, ...props }) => <strong className="md-strong" {...props} />,
        em: ({ node, ...props }) => <em className="md-em" {...props} />,
        a: ({ node, href, children, ...props }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" className="md-link" {...props}>{children}</a>
        ),
        ul: ({ node, ...props }) => <ul className="md-ul" {...props} />,
        ol: ({ node, ...props }) => <ol className="md-ol" {...props} />,
        li: ({ node, ...props }) => <li className="md-li" {...props} />,
        hr: () => <hr className="md-hr" />,
        code: ({ node, inline, ...props }) =>
          inline
            ? <code className="md-code-inline" {...props} />
            : <code className="md-code-block" {...props} />,
        h1: ({ node, ...props }) => <h3 className="md-h" {...props} />,
        h2: ({ node, ...props }) => <h3 className="md-h" {...props} />,
        h3: ({ node, ...props }) => <h3 className="md-h" {...props} />,
      }}
    >
      {children}
    </ReactMarkdown>
  )
}

export default function Message({ message, mode }) {
  const { role, content, error, citations, meta, streaming, ttft } = message
  const isUser = role === 'user'

  if (isUser) {
    return (
      <div className="msg msg--user">
        <div className="msg-bubble msg-bubble--user">
          {content}
        </div>
      </div>
    )
  }

  /* Agent streaming */
  if (mode === 'agent') {
    return (
      <div className="msg msg--assistant">
        <div className="msg-avatar">AI</div>
        <div className="msg-content">
          <div className={`msg-bubble msg-bubble--assistant ${streaming ? 'msg-bubble--streaming' : ''}`}>
            {error ? (
              <span className="msg-error">{error}</span>
            ) : (
              <>
                <MD>{content}</MD>
                {streaming && <span className="msg-cursor" aria-hidden="true">▌</span>}
              </>
            )}
          </div>
          {!streaming && !error && ttft !== null && (
            <p className="msg-caption">Streamed · TTFT: {Math.round(ttft)}ms</p>
          )}
        </div>
      </div>
    )
  }

  /* Student full response */
  return (
    <div className="msg msg--assistant">
      <div className="msg-avatar">AI</div>
      <div className="msg-content">
        <div className="msg-bubble msg-bubble--assistant">
          {error ? (
            <span className="msg-error">{error}</span>
          ) : (
            <MD>{content}</MD>
          )}
        </div>

        {citations && citations.length > 0 && (
          <div className="msg-citations">
            {citations.map(c => (
              <a
                key={c.id}
                href={c.url}
                target="_blank"
                rel="noopener noreferrer"
                className="citation-card"
              >
                <span className="citation-id">KB-{c.id}</span>
                <span className="citation-url">{c.url}</span>
                <svg className="citation-icon" width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 2H2a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V7" />
                  <path d="M8 1h3v3" />
                  <line x1="11" y1="1" x2="5" y2="7" />
                </svg>
              </a>
            ))}
          </div>
        )}

        {meta && !error && (
          <div className="msg-meta">
            {meta.resolved && <span className="badge badge--resolved">Resolved</span>}
            {meta.escalated && !meta.resolved && <span className="badge badge--escalated">Escalated</span>}
            <span className="msg-caption-inline">
              Turn {meta.turn}{meta.complexity && ` · ${meta.complexity}`}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

import { useState, useEffect } from 'react'

export function useMode() {
  const getInitialMode = () => {
    const params = new URLSearchParams(window.location.search)
    const m = params.get('mode')
    return m === 'agent' ? 'agent' : 'student'
  }

  const [mode, setModeState] = useState(getInitialMode)

  const setMode = (newMode) => {
    setModeState(newMode)
    const url = new URL(window.location.href)
    url.searchParams.set('mode', newMode)
    window.history.replaceState(null, '', url.toString())
  }

  return [mode, setMode]
}

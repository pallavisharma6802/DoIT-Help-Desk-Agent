import { useState, useEffect } from 'react'

export function useCategories(apiBase) {
  const [categories, setCategories] = useState(null)

  useEffect(() => {
    if (!apiBase) return
    fetch(`${apiBase}/categories`)
      .then(res => {
        if (res.status === 404) return null
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(data => setCategories(data))
      .catch(() => setCategories(null))
  }, [apiBase])

  return categories
}

import { useEffect, useState } from 'react'
import type { ModelItem, ModelsResponse } from './protocol'

/**
 * Loads the model list from `GET /models`.
 *
 * Fetches once `enabled` is true and retries on the next change to true if
 * the request failed, so a backend that wasn't up yet is picked up once the
 * socket connects.
 *
 * @param enabled Whether the backend is reachable, i.e. the socket is open.
 * @returns The models in catalog order, or null until they have loaded.
 */
export function useModels(enabled: boolean): ModelItem[] | null {
  const [models, setModels] = useState<ModelItem[] | null>(null)

  useEffect(() => {
    if (!enabled || models) return
    const controller = new AbortController()
    fetch('/models', { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`GET /models failed with ${res.status}`)
        return res.json() as Promise<ModelsResponse>
      })
      .then((data) => setModels(data.models))
      .catch((err: unknown) => {
        if (!controller.signal.aborted) console.error(err)
      })
    return () => controller.abort()
  }, [enabled, models])

  return models
}

import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8001'

const Icon = ({ children, label }) => (
  <svg aria-hidden={label ? undefined : true} aria-label={label} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
)

function UploadIcon() {
  return <Icon><path d="M12 16V3" /><path d="m7 8 5-5 5 5" /><path d="M5 13v6h14v-6" /></Icon>
}

function PlayIcon({ pause }) {
  return pause ? <Icon><path d="M8 5v14M16 5v14" /></Icon> : <Icon><path d="m8 5 11 7-11 7V5Z" fill="currentColor" /></Icon>
}

function App() {
  const [mode, setMode] = useState('image')
  const [sourceUrl, setSourceUrl] = useState('')
  const [resultUrl, setResultUrl] = useState('')
  const [fileName, setFileName] = useState('')
  const [confidence, setConfidence] = useState(0.4)
  const [opacity, setOpacity] = useState(0.55)
  const [color, setColor] = useState('#20d7f5')
  const [enableTracking, setEnableTracking] = useState(true)
  const [status, setStatus] = useState('Ready for media')
  const [metrics, setMetrics] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  const fileRef = useRef(null)
  const videoRef = useRef(null)

  useEffect(() => () => sourceUrl && URL.revokeObjectURL(sourceUrl), [sourceUrl])

  const resetMedia = useCallback(() => {
    setResultUrl('')
    setMetrics(null)
    setVideoTime(0)
    setVideoDuration(0)
    setIsPlaying(false)
  }, [])

  const onPickFile = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    const nextMode = file.type.startsWith('video/') ? 'video' : 'image'
    setMode(nextMode)
    resetMedia()
    const localUrl = URL.createObjectURL(file)
    setSourceUrl(localUrl)
    setFileName(file.name)
    setStatus(`Processing ${file.name}`)
    setIsProcessing(true)

    const form = new FormData()
    form.append('file', file)
    form.append('confidence', confidence.toString())
    form.append('overlay_opacity', opacity.toString())
    form.append('overlay_color', color)
    if (nextMode === 'video') form.append('enable_tracking', enableTracking.toString())
    try {
      const endpoint = nextMode === 'image' ? '/api/segment/image' : '/api/segment/video'
      const response = await fetch(`${API_BASE}${endpoint}`, { method: 'POST', body: form })
      if (!response.ok) throw new Error((await response.json()).detail || 'Segmentation failed')
      const payload = await response.json()
      setResultUrl(`${API_BASE}${payload.result_url}?v=${Date.now()}`)
      setMetrics(payload.metrics)
      setStatus(nextMode === 'image' ? 'Mask ready' : 'Video mask ready')
    } catch (error) {
      setStatus(error.message || 'Could not process this media')
    } finally {
      setIsProcessing(false)
    }
  }

  const openPicker = () => fileRef.current?.click()
  const activeUrl = resultUrl || sourceUrl
  const formattedTime = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
  const toggleVideo = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) video.play(); else video.pause()
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-dot" />Human Mask Studio</div>
        <div className="mode-switch" role="tablist" aria-label="Media type">
          {['image', 'video'].map((item) => <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)} role="tab" aria-selected={mode === item}>{item === 'image' ? 'Image' : 'Video'}</button>)}
        </div>
        <div className="gpu-status"><span /> GPU · RTX 5080</div>
      </header>

      <section className="workspace">
        <section className="canvas-region">
          <div className={`preview ${activeUrl ? 'has-media' : ''}`}>
            {!activeUrl && <div className="empty-preview"><div className="empty-icon"><UploadIcon /></div><h1>Drop an image or video here</h1><p>YOLO will find every person and draw a mask over each one.</p><button className="link-button" onClick={openPicker}>Choose media</button></div>}
            {activeUrl && mode === 'image' && <img src={activeUrl} alt={resultUrl ? 'Person segmentation result' : 'Selected image'} />}
            {activeUrl && mode === 'video' && <video ref={videoRef} src={activeUrl} onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration)} onTimeUpdate={(e) => setVideoTime(e.currentTarget.currentTime)} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} />}
            {isProcessing && <div className="processing"><div className="spinner" />{mode === 'video' && enableTracking ? 'Segmenting and tracking IDs…' : 'Running YOLO segmentation…'}</div>}
            {resultUrl && <div className="mask-indicator"><span /> Mask overlay</div>}
          </div>
          {mode === 'video' && <div className="transport"><button onClick={toggleVideo} aria-label={isPlaying ? 'Pause video' : 'Play video'}><PlayIcon pause={isPlaying} /></button><input aria-label="Video position" type="range" min="0" max={videoDuration || 1} step="0.01" value={videoTime} onChange={(e) => { if (videoRef.current) videoRef.current.currentTime = Number(e.target.value) }} /><span>{formattedTime(videoTime)} / {formattedTime(videoDuration)}</span></div>}
          <div className="media-caption"><span>{fileName || 'No media selected'}</span><span>{status}</span></div>
        </section>

        <aside className="inspector">
          <input ref={fileRef} className="visually-hidden" type="file" accept="image/*,video/*" onChange={onPickFile} />
          <button className="upload-button" onClick={openPicker}><UploadIcon /> Upload media</button>
          <div className="control-group"><label htmlFor="model">Model</label><select id="model" defaultValue="yolo11n-seg"><option value="yolo11n-seg">YOLO11n Segment</option></select><p>Fast person instance masks</p></div>
          {mode === 'video' ? <div className="control-group tracking-control"><label className="toggle-row" htmlFor="tracking"><span><strong>Identity tracking</strong><small>BoT-SORT · ReID embeddings</small></span><input id="tracking" type="checkbox" checked={enableTracking} onChange={(event) => setEnableTracking(event.target.checked)} /></label></div> : null}
          <div className="control-group"><label htmlFor="confidence">Confidence <output>{confidence.toFixed(2)}</output></label><input id="confidence" type="range" min="0.1" max="0.9" step="0.05" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></div>
          <div className="control-group"><label htmlFor="opacity">Overlay opacity <output>{opacity.toFixed(2)}</output></label><input id="opacity" type="range" min="0.1" max="0.9" step="0.05" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} /></div>
          {mode === 'image' || !enableTracking ? <div className="control-group"><label htmlFor="color">Mask color</label><div className="color-field"><input id="color" type="color" value={color} onChange={(e) => setColor(e.target.value)} /><code>{color.toUpperCase()}</code></div></div> : null}
          <div className="metrics">
            <div><span>Inference</span><strong>{metrics ? `${metrics.inference_ms.toFixed(1)} ms` : '—'}</strong></div>
            <div><span>People</span><strong>{metrics ? metrics.people : '—'}</strong></div>
            {mode === 'video' && enableTracking ? <div><span>Track IDs</span><strong>{metrics ? metrics.unique_tracks : '—'}</strong></div> : null}
            <div><span>FPS</span><strong>{metrics?.fps ? metrics.fps.toFixed(1) : '—'}</strong></div>
          </div>
        </aside>
      </section>
      <footer><span className="footer-mark">?</span><span>Start with a photo to inspect mask quality, then try a short video to measure FPS.</span></footer>
    </main>
  )
}

export default App

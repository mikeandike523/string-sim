import { Div, Canvas } from "style-props-html"
import '../App.css'
import { useEffect, useRef, useCallback, useState } from "react"
import { throttle } from "lodash"

const NUM_POINTS = 100
const DIAGRAM_HEIGHT_MM = 7
// Physics constants needed on main thread to compute derived params for worklet
// Must stay in sync with string-processor.js
const STRING_TENSION = 50
const K_SPRING = 12_000
const BENDING_STIFFNESS = 1e-9

const NODE_RADIUS = 2
const LINE_THICKNESS = 3
const NODE_COLOR = "RED"
const LINE_COLOR = "BLACK"
const PICKUP_COLOR = "#2196F3"

// 45% of window width = 1 metre of string at reference scale
const PIXELS_PER_METER_FRACTION = 0.45

function computeCanvasDims(stringLength: number) {
  const ppm = window.innerWidth * PIXELS_PER_METER_FRACTION
  const width = Math.max(120, Math.min(Math.round(ppm * stringLength), window.innerWidth - 48))
  // Height is fixed to a comfortable fraction of the viewport — it represents
  // displacement amplitude, which is independent of string length
  const height = Math.min(Math.round(window.innerHeight * 0.30), 280)
  return { width, height }
}

export default function IndexPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const gainNodeRef = useRef<GainNode | null>(null)
  // Latest y-positions snapshot from worklet (or drag override)
  const latestPositionsRef = useRef<Float32Array | null>(null)
  // Local copy of last-sent positions, used for nearest-node search during drag
  const localPositionsRef = useRef<Float32Array>(new Float32Array(NUM_POINTS))
  const pickupIndexRef = useRef<number>(20)
  const audioStartedRef = useRef(false)
  const gainValueRef = useRef(0.7)
  const stringLengthRef = useRef(1.0)
  const slowMoRef = useRef(1)
  const rafRef = useRef<number>(0)

  const [audioStarted, setAudioStarted] = useState(false)
  const [pickupFraction, setPickupFraction] = useState(0.2)
  const [gainValue, setGainValue] = useState(0.7)
  const [stringLength, setStringLength] = useState(1.0)
  const [slowMo, setSlowMo] = useState(1)
  const [canvasDims, setCanvasDims] = useState(() => computeCanvasDims(1.0))

  // Compute and apply canvas CSS + resolution dimensions.
  // Reads stringLengthRef so it can be called stably from resize handler.
  const updateCanvasSize = useCallback(() => {
    const dims = computeCanvasDims(stringLengthRef.current)
    setCanvasDims(dims)
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio ?? 1
    canvas.width = Math.round(dims.width * dpr)
    canvas.height = Math.round(dims.height * dpr)
    if (!ctxRef.current) ctxRef.current = canvas.getContext("2d")
    ctxRef.current?.setTransform(dpr, 0, 0, dpr, 0, 0)
  }, [])

  // Render loop reads only refs — stable identity, no deps
  const render = useCallback(function renderFrame() {
    const ctx = ctxRef.current
    const canvas = canvasRef.current

    if (ctx && canvas) {
      const { width, height } = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, width, height)

      const positions = latestPositionsRef.current
      // Node i maps to x-fraction i/(N-1), independent of string length
      const xFrac = (i: number) => width * (i / (NUM_POINTS - 1))
      const yPx = (y: number) => height / 2 - (height / 2) * (y * 1000 / (DIAGRAM_HEIGHT_MM / 2))

      if (positions) {
        ctx.strokeStyle = LINE_COLOR
        ctx.lineWidth = LINE_THICKNESS
        for (let i = 0; i < NUM_POINTS - 1; i++) {
          ctx.beginPath()
          ctx.moveTo(xFrac(i), yPx(positions[i]))
          ctx.lineTo(xFrac(i + 1), yPx(positions[i + 1]))
          ctx.stroke()
        }

        ctx.fillStyle = NODE_COLOR
        for (let i = 0; i < NUM_POINTS; i++) {
          ctx.beginPath()
          ctx.arc(xFrac(i), yPx(positions[i]), NODE_RADIUS, 0, 2 * Math.PI)
          ctx.fill()
        }

        // Pickup marker: blue triangle below the pickup node
        const pi = pickupIndexRef.current
        const px = xFrac(pi)
        const py = yPx(positions[pi])
        ctx.fillStyle = PICKUP_COLOR
        ctx.beginPath()
        ctx.moveTo(px, py + 8)
        ctx.lineTo(px - 6, py + 18)
        ctx.lineTo(px + 6, py + 18)
        ctx.closePath()
        ctx.fill()
      } else {
        // Idle: flat string
        ctx.strokeStyle = LINE_COLOR
        ctx.lineWidth = LINE_THICKNESS
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.stroke()

        ctx.fillStyle = NODE_COLOR
        for (const x of [0, width]) {
          ctx.beginPath()
          ctx.arc(x, height / 2, NODE_RADIUS, 0, 2 * Math.PI)
          ctx.fill()
        }

        if (!audioStartedRef.current) {
          ctx.font = '14px sans-serif'
          ctx.fillStyle = '#666'
          ctx.textAlign = 'center'
          ctx.fillText('Click "Start Audio" to begin', width / 2, height / 2 - 20)
        }
      }
    }

    rafRef.current = requestAnimationFrame(renderFrame)
  }, [])

  // Initial sizing + window resize handler
  useEffect(() => {
    updateCanvasSize()
    const handleResize = throttle(updateCanvasSize, 100)
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('resize', handleResize); handleResize.cancel() }
  }, [updateCanvasSize])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(render)
    return () => cancelAnimationFrame(rafRef.current)
  }, [render])

  useEffect(() => {
    return () => { audioCtxRef.current?.close() }
  }, [])

  // Sync pickupFraction → ref + worklet
  useEffect(() => {
    const index = Math.max(1, Math.min(NUM_POINTS - 2, Math.round(pickupFraction * (NUM_POINTS - 1))))
    pickupIndexRef.current = index
    workletNodeRef.current?.port.postMessage({ type: 'set_pickup', index })
  }, [pickupFraction])

  // Sync gainValue → ref + GainNode
  useEffect(() => {
    gainValueRef.current = gainValue
    if (gainNodeRef.current) gainNodeRef.current.gain.value = gainValue
  }, [gainValue])

  // Sync slowMo → ref + worklet
  useEffect(() => {
    slowMoRef.current = slowMo
    workletNodeRef.current?.port.postMessage({ type: 'set_slowmo', value: slowMo })
  }, [slowMo])

  const startAudio = useCallback(async () => {
    const audioCtx = new AudioContext()
    await audioCtx.audioWorklet.addModule('/string-processor.js')
    const workletNode = new AudioWorkletNode(audioCtx, 'string-processor')
    const gainNode = audioCtx.createGain()
    gainNode.gain.value = gainValueRef.current
    workletNode.connect(gainNode)
    gainNode.connect(audioCtx.destination)

    workletNode.port.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'visual_update') {
        latestPositionsRef.current = e.data.positions as Float32Array
      }
    }

    // Sync current slider values to worklet
    workletNode.port.postMessage({ type: 'set_pickup', index: pickupIndexRef.current })
    if (slowMoRef.current !== 1) {
      workletNode.port.postMessage({ type: 'set_slowmo', value: slowMoRef.current })
    }
    const sl = stringLengthRef.current
    const segLength = sl / (NUM_POINTS - 1)
    workletNode.port.postMessage({
      type: 'set_params',
      segLength,
      bendingForceCoeff: BENDING_STIFFNESS / Math.pow(segLength, 3),
    })

    audioCtxRef.current = audioCtx
    workletNodeRef.current = workletNode
    gainNodeRef.current = gainNode
    audioStartedRef.current = true
    setAudioStarted(true)
  }, [])

  const handleStringLengthChange = useCallback((newLength: number) => {
    stringLengthRef.current = newLength
    setStringLength(newLength)
    updateCanvasSize()

    const segLength = newLength / (NUM_POINTS - 1)
    workletNodeRef.current?.port.postMessage({
      type: 'set_params',
      segLength,
      bendingForceCoeff: BENDING_STIFFNESS / Math.pow(segLength, 3),
    })

    // After reset, show flat string (not null) so the canvas doesn't revert to idle text
    localPositionsRef.current.fill(0)
    latestPositionsRef.current = audioStartedRef.current ? new Float32Array(NUM_POINTS) : null
  }, [updateCanvasSize])

  // Compute triangle pluck positions for the given pointer coordinates.
  // Uses localPositionsRef (last-sent state) for the nearest-node distance search.
  const computePluckPositions = useCallback((pointerX: number, pointerY: number): Float32Array => {
    const canvas = canvasRef.current
    const positions = new Float32Array(NUM_POINTS)
    if (!canvas) return positions

    const { width, height, left, top } = canvas.getBoundingClientRect()
    const L = stringLengthRef.current
    const segLength = L / (NUM_POINTS - 1)
    const pointerWorldX = ((pointerX - left) / width) * L
    const pointerWorldY = ((height / 2 - (pointerY - top)) / (height / 2)) * (DIAGRAM_HEIGHT_MM / 1000 / 2)
    const clampedY = Math.max(
      -(DIAGRAM_HEIGHT_MM / 1000 / 2),
      Math.min(DIAGRAM_HEIGHT_MM / 1000 / 2, pointerWorldY)
    )

    const currentY = localPositionsRef.current
    let nearestIndex = 1
    let nearestDist = Infinity

    for (let i = 1; i < NUM_POINTS - 1; i++) {
      const d = Math.sqrt(
        (segLength * i - pointerWorldX) ** 2 +
        (currentY[i] - clampedY) ** 2
      )
      if (d < nearestDist) {
        nearestDist = d
        nearestIndex = i
      }
    }

    const apexY = clampedY
    const lastIdx = NUM_POINTS - 1

    for (let i = 0; i <= nearestIndex; i++) {
      positions[i] = nearestIndex === 0 ? 0 : (i / nearestIndex) * apexY
    }
    for (let i = nearestIndex + 1; i <= lastIdx; i++) {
      const t = (i - nearestIndex) / (lastIdx - nearestIndex)
      positions[i] = (1 - t) * apexY
    }
    positions[0] = 0
    positions[lastIdx] = 0

    return positions
  }, [])

  // Send pluck state to worklet and update local refs for immediate visual feedback.
  const sendPluck = useCallback((positions: Float32Array) => {
    const workletNode = workletNodeRef.current
    if (!workletNode) return

    // Transfer copies to worklet (zero-copy via Transferable)
    const posCopy = new Float32Array(positions)
    const velCopy = new Float32Array(NUM_POINTS)
    workletNode.port.postMessage(
      { type: 'pluck', positions: posCopy, velocities: velCopy },
      [posCopy.buffer, velCopy.buffer]
    )

    localPositionsRef.current.set(positions)
    latestPositionsRef.current = positions
  }, [])

  const controlsWidth = Math.max(280, canvasDims.width)

  return (
    <Div
      width="100%"
      height="100%"
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      gap="12px"
    >
      <Canvas
        ref={canvasRef}
        style={{ width: canvasDims.width, height: canvasDims.height }}
        border="1px solid black"
        cursor={audioStarted ? "crosshair" : "default"}
        onPointerDown={(e) => {
          if (!workletNodeRef.current) return
          workletNodeRef.current.port.postMessage({ type: 'set_paused', paused: true })
          const positions = computePluckPositions(e.clientX, e.clientY)
          sendPluck(positions)
          e.currentTarget.setPointerCapture(e.pointerId)
        }}
        onPointerMove={(e) => {
          if (!e.currentTarget.hasPointerCapture(e.pointerId)) return
          const positions = computePluckPositions(e.clientX, e.clientY)
          sendPluck(positions)
        }}
        onPointerUp={(e) => {
          workletNodeRef.current?.port.postMessage({ type: 'set_paused', paused: false })
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId)
          }
        }}
        onPointerCancel={(e) => {
          workletNodeRef.current?.port.postMessage({ type: 'set_paused', paused: false })
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId)
          }
        }}
      />

      <Div
        display="flex"
        flexDirection="column"
        gap="8px"
        fontSize="14px"
        style={{ width: controlsWidth }}
      >
        <Div display="flex" alignItems="center" gap="8px">
          <label style={{ width: 80, color: '#555' }}>Length</label>
          <input
            type="range"
            min={Math.log10(0.1)}
            max={Math.log10(5)}
            step={0.005}
            value={Math.log10(stringLength)}
            onChange={(e) => handleStringLengthChange(Math.pow(10, parseFloat(e.target.value)))}
            style={{ flex: 1 }}
          />
          <span style={{ width: 44, textAlign: 'right', color: '#555', fontVariantNumeric: 'tabular-nums' }}>
            {stringLength.toFixed(2)} m
          </span>
        </Div>

        <Div display="flex" alignItems="center" gap="8px">
          <label style={{ width: 80, color: '#555' }}>Slow-mo</label>
          <input
            type="range" min={1} max={100} step={1}
            value={slowMo}
            onChange={(e) => setSlowMo(parseInt(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ width: 44, textAlign: 'right', color: '#555', fontVariantNumeric: 'tabular-nums' }}>
            {slowMo}×
          </span>
        </Div>

        <Div display="flex" alignItems="center" gap="8px">
          <label style={{ width: 80, color: '#555' }}>Pickup</label>
          <input
            type="range" min={0.05} max={0.95} step={0.01}
            value={pickupFraction}
            onChange={(e) => setPickupFraction(parseFloat(e.target.value))}
            style={{ flex: 1 }}
            disabled={!audioStarted}
          />
          <span style={{ width: 44, textAlign: 'right', color: '#555', fontVariantNumeric: 'tabular-nums' }}>
            {Math.round(pickupFraction * 100)}%
          </span>
        </Div>

        <Div display="flex" alignItems="center" gap="8px">
          <label style={{ width: 80, color: '#555' }}>Volume</label>
          <input
            type="range" min={0} max={1} step={0.01}
            value={gainValue}
            onChange={(e) => setGainValue(parseFloat(e.target.value))}
            style={{ flex: 1 }}
            disabled={!audioStarted}
          />
          <span style={{ width: 44, textAlign: 'right', color: '#555', fontVariantNumeric: 'tabular-nums' }}>
            {Math.round(gainValue * 100)}%
          </span>
        </Div>

        <Div display="flex" justifyContent="center" paddingTop="4px">
          {!audioStarted ? (
            <button
              onClick={startAudio}
              style={{ padding: '8px 24px', cursor: 'pointer', fontSize: '14px', borderRadius: 4 }}
            >
              Start Audio
            </button>
          ) : (
            <span style={{ color: '#4caf50', fontSize: 13 }}>
              ● Audio running — drag the string to pluck
            </span>
          )}
        </Div>
      </Div>
    </Div>
  )
}

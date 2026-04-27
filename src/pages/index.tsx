import { Div, Canvas } from "style-props-html"
import '../App.css'
import { useEffect, useRef, useCallback, useState } from "react"
import { throttle } from "lodash"

const DIAGRAM_HEIGHT_MM = 4
const MIN_STRING_LENGTH = 0.02
const MAX_STRING_LENGTH = 1.5
const CONTROL_PANEL_WIDTH = 420
const MIN_CANVAS_WIDTH = 24

const NODE_RADIUS = 2
const LINE_THICKNESS = 3
const NODE_COLOR = "RED"
const LINE_COLOR = "BLACK"
const PICKUP_COLOR = "#2196F3"

// 45% of window width = 1 metre of string at reference scale
const PIXELS_PER_METER_FRACTION = 0.45

type SimulationParameters = {
  pointCount: number
  stringMass: number
  stringLength: number
  stringTension: number
  springK: number
  bendingStiffness: number
  segmentDamping: number
  supportDampingRate: number
  pickupOutputGain: number
  visualUpdateInterval: number
  pickupFraction: number
  slowMo: number
  outputVolume: number
}

const DEFAULT_SIMULATION_PARAMETERS: SimulationParameters = {
  pointCount: 50,
  stringMass: 3e-3,
  stringLength: 1,
  stringTension: 40,
  springK: 30_000,
  bendingStiffness: 3e-8,
  segmentDamping: 0.25,
  supportDampingRate: 0.75,
  pickupOutputGain: 2.5,
  visualUpdateInterval: 512,
  pickupFraction: 0.2,
  slowMo: 1,
  outputVolume: 0.7,
}

type ParameterKey = keyof SimulationParameters
type WorkletParameterKey = Exclude<ParameterKey, 'outputVolume'>

type ParameterDefinition = {
  key: ParameterKey
  label: string
  unit?: string
  kind: 'slider' | 'fixed'
  min?: number
  max?: number
  step?: number
  format: (value: number) => string
  toSlider?: (value: number) => number
  fromSlider?: (value: number) => number
  sendToWorklet?: boolean
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function clampStringLength(length: number) {
  return clamp(length, MIN_STRING_LENGTH, MAX_STRING_LENGTH)
}

function sliderParamToStringLength(param: number) {
  const t = clamp(param, 0, 1)
  const logarithmicScale = (Math.pow(10, t) - 1) / 9
  return MIN_STRING_LENGTH + (MAX_STRING_LENGTH - MIN_STRING_LENGTH) * logarithmicScale
}

function stringLengthToSliderParam(length: number) {
  const normalizedLength = (clampStringLength(length) - MIN_STRING_LENGTH) / (MAX_STRING_LENGTH - MIN_STRING_LENGTH)
  return clamp(Math.log10(1 + 9 * normalizedLength), 0, 1)
}

const PARAMETER_DEFINITIONS: ParameterDefinition[] = [
  {
    key: 'stringLength',
    label: 'Length',
    unit: 'm',
    kind: 'slider',
    min: 0,
    max: 1,
    step: 0.001,
    toSlider: stringLengthToSliderParam,
    fromSlider: sliderParamToStringLength,
    format: value => `${value.toFixed(2)} m`,
  },
  {
    key: 'slowMo',
    label: 'Slow-mo',
    kind: 'slider',
    min: 1,
    max: 100,
    step: 1,
    format: value => `${Math.round(value)}x`,
  },
  {
    key: 'pickupFraction',
    label: 'Pickup',
    kind: 'slider',
    min: 0.05,
    max: 0.95,
    step: 0.01,
    format: value => `${Math.round(value * 100)}%`,
  },
  {
    key: 'outputVolume',
    label: 'Volume',
    kind: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
    sendToWorklet: false,
    format: value => `${Math.round(value * 100)}%`,
  },
  { key: 'pointCount', label: 'Points', kind: 'fixed', format: value => `${Math.round(value)}` },
  { key: 'stringMass', label: 'Mass', kind: 'fixed', format: value => `${value.toExponential(1)} kg` },
  { key: 'stringTension', label: 'Tension', kind: 'fixed', format: value => `${value.toFixed(0)} N` },
  { key: 'springK', label: 'Spring K', kind: 'fixed', format: value => `${value.toLocaleString()} N/m` },
  { key: 'bendingStiffness', label: 'Bending', kind: 'fixed', format: value => `${value.toExponential(1)} N m^2` },
  { key: 'segmentDamping', label: 'Damping', kind: 'fixed', format: value => value.toFixed(2) },
  { key: 'supportDampingRate', label: 'Support damp', kind: 'fixed', format: value => value.toFixed(2) },
  { key: 'pickupOutputGain', label: 'Pickup gain', kind: 'fixed', format: value => value.toFixed(1) },
  { key: 'visualUpdateInterval', label: 'Visual step', kind: 'fixed', format: value => `${Math.round(value)} samples` },
]

function getWorkletParameters(params: SimulationParameters): Record<WorkletParameterKey, number> {
  return {
    pointCount: params.pointCount,
    stringMass: params.stringMass,
    stringLength: params.stringLength,
    stringTension: params.stringTension,
    springK: params.springK,
    bendingStiffness: params.bendingStiffness,
    segmentDamping: params.segmentDamping,
    supportDampingRate: params.supportDampingRate,
    pickupOutputGain: params.pickupOutputGain,
    visualUpdateInterval: params.visualUpdateInterval,
    pickupFraction: params.pickupFraction,
    slowMo: params.slowMo,
  }
}

function computePickupIndex(pickupFraction: number, pointCount: number) {
  return Math.max(1, Math.min(pointCount - 2, Math.round(pickupFraction * (pointCount - 1))))
}

function computeCanvasDims(stringLength: number) {
  const ppm = window.innerWidth * PIXELS_PER_METER_FRACTION
  const width = Math.max(MIN_CANVAS_WIDTH, Math.min(Math.round(ppm * stringLength), window.innerWidth - 48))
  // Height is fixed to a comfortable fraction of the viewport. It represents
  // displacement amplitude, which is independent of string length.
  const height = Math.min(Math.round(window.innerHeight * 0.30), 280)
  return { width, height }
}

export default function IndexPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const gainNodeRef = useRef<GainNode | null>(null)
  const latestPositionsRef = useRef<Float32Array | null>(null)
  const localPositionsRef = useRef<Float32Array>(new Float32Array(DEFAULT_SIMULATION_PARAMETERS.pointCount))
  const pickupIndexRef = useRef<number>(computePickupIndex(
    DEFAULT_SIMULATION_PARAMETERS.pickupFraction,
    DEFAULT_SIMULATION_PARAMETERS.pointCount,
  ))
  const audioStartedRef = useRef(false)
  const simulationPausedRef = useRef(false)
  const dragWasPausedRef = useRef(false)
  const parametersRef = useRef<SimulationParameters>(DEFAULT_SIMULATION_PARAMETERS)
  const rafRef = useRef<number>(0)

  const [audioStarted, setAudioStarted] = useState(false)
  const [simulationPaused, setSimulationPaused] = useState(false)
  const [parameters, setParameters] = useState<SimulationParameters>(DEFAULT_SIMULATION_PARAMETERS)
  const [canvasDims, setCanvasDims] = useState(() => computeCanvasDims(DEFAULT_SIMULATION_PARAMETERS.stringLength))

  const pointCount = Math.round(parameters.pointCount)

  const postParameterInitialization = useCallback((node: AudioWorkletNode, params: SimulationParameters) => {
    node.port.postMessage({
      type: 'initialize_parameters',
      parameters: getWorkletParameters(params),
      paused: simulationPausedRef.current,
    })
  }, [])

  const postParameterValue = useCallback((key: ParameterKey, value: number) => {
    if (key === 'outputVolume') return
    workletNodeRef.current?.port.postMessage({
      type: 'set_parameter',
      key,
      value,
    })
  }, [])

  // Compute and apply canvas CSS + resolution dimensions.
  const updateCanvasSize = useCallback((stringLength = parametersRef.current.stringLength) => {
    const dims = computeCanvasDims(stringLength)
    setCanvasDims(dims)
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio ?? 1
    canvas.width = Math.round(dims.width * dpr)
    canvas.height = Math.round(dims.height * dpr)
    if (!ctxRef.current) ctxRef.current = canvas.getContext("2d")
    ctxRef.current?.setTransform(dpr, 0, 0, dpr, 0, 0)
  }, [])

  const setParameterValue = useCallback((key: ParameterKey, value: number) => {
    const definition = PARAMETER_DEFINITIONS.find(param => param.key === key)
    const min = definition?.kind === 'slider' ? definition.min : undefined
    const max = definition?.kind === 'slider' ? definition.max : undefined
    const nextValue = min != null && max != null ? clamp(value, min, max) : value
    const nextParameters = {
      ...parametersRef.current,
      [key]: nextValue,
    }

    parametersRef.current = nextParameters
    setParameters(nextParameters)

    if (key === 'stringLength') updateCanvasSize(nextValue)
    if (key === 'outputVolume' && gainNodeRef.current) gainNodeRef.current.gain.value = nextValue
    if (key === 'pointCount') {
      localPositionsRef.current = new Float32Array(Math.round(nextValue))
      latestPositionsRef.current = null
    }
    if (key === 'pickupFraction' || key === 'pointCount') {
      pickupIndexRef.current = computePickupIndex(nextParameters.pickupFraction, Math.round(nextParameters.pointCount))
    }

    postParameterValue(key, nextValue)
  }, [postParameterValue, updateCanvasSize])

  const setSimulationPausedState = useCallback((paused: boolean) => {
    simulationPausedRef.current = paused
    setSimulationPaused(paused)
    workletNodeRef.current?.port.postMessage({ type: 'set_simulation_paused', paused })
  }, [])

  // Render loop reads refs for audio state and latest positions.
  const render = useCallback(function renderFrame() {
    const ctx = ctxRef.current
    const canvas = canvasRef.current

    if (ctx && canvas) {
      const { width, height } = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, width, height)

      const positions = latestPositionsRef.current
      const count = positions?.length ?? pointCount
      const xFrac = (i: number) => count <= 1 ? 0 : width * (i / (count - 1))
      const yPx = (y: number) => height / 2 - (height / 2) * (y * 1000 / (DIAGRAM_HEIGHT_MM / 2))

      if (positions) {
        ctx.strokeStyle = LINE_COLOR
        ctx.lineWidth = LINE_THICKNESS
        for (let i = 0; i < count - 1; i++) {
          ctx.beginPath()
          ctx.moveTo(xFrac(i), yPx(positions[i]))
          ctx.lineTo(xFrac(i + 1), yPx(positions[i + 1]))
          ctx.stroke()
        }

        ctx.fillStyle = NODE_COLOR
        for (let i = 0; i < count; i++) {
          ctx.beginPath()
          ctx.arc(xFrac(i), yPx(positions[i]), NODE_RADIUS, 0, 2 * Math.PI)
          ctx.fill()
        }

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
        ctx.strokeStyle = LINE_COLOR
        ctx.lineWidth = LINE_THICKNESS
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.stroke()

        ctx.fillStyle = NODE_COLOR
        for (let i = 0; i < count; i++) {
          ctx.beginPath()
          ctx.arc(xFrac(i), height / 2, NODE_RADIUS, 0, 2 * Math.PI)
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
  }, [pointCount])

  useEffect(() => {
    updateCanvasSize()
    const handleResize = throttle(() => updateCanvasSize(), 100)
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

  const startAudio = useCallback(async () => {
    const audioCtx = new AudioContext()
    await audioCtx.audioWorklet.addModule('/string-processor.js')
    const workletNode = new AudioWorkletNode(audioCtx, 'string-processor')
    const gainNode = audioCtx.createGain()
    gainNode.gain.value = parametersRef.current.outputVolume
    workletNode.connect(gainNode)
    gainNode.connect(audioCtx.destination)

    workletNode.port.onmessage = (e: MessageEvent) => {
      if (e.data.type === 'initialized') {
        const nextPointCount = e.data.numPoints as number
        localPositionsRef.current = new Float32Array(nextPointCount)
        latestPositionsRef.current = null
        pickupIndexRef.current = computePickupIndex(parametersRef.current.pickupFraction, nextPointCount)
      } else if (e.data.type === 'visual_update') {
        const positions = e.data.positions as Float32Array
        latestPositionsRef.current = positions
        if (localPositionsRef.current.length === positions.length) {
          localPositionsRef.current.set(positions)
        }
      }
    }

    audioCtxRef.current = audioCtx
    workletNodeRef.current = workletNode
    gainNodeRef.current = gainNode
    postParameterInitialization(workletNode, parametersRef.current)
    audioStartedRef.current = true
    setAudioStarted(true)
  }, [postParameterInitialization])

  const computePluckPositions = useCallback((pointerX: number, pointerY: number): Float32Array => {
    const canvas = canvasRef.current
    const count = Math.round(parametersRef.current.pointCount)
    const positions = new Float32Array(count)
    if (!canvas) return positions

    const { width, height, left, top } = canvas.getBoundingClientRect()
    const stringLength = parametersRef.current.stringLength
    const segLength = stringLength / (count - 1)
    const pointerWorldX = ((pointerX - left) / width) * stringLength
    const pointerWorldY = ((height / 2 - (pointerY - top)) / (height / 2)) * (DIAGRAM_HEIGHT_MM / 1000 / 2)
    const clampedY = clamp(pointerWorldY, -(DIAGRAM_HEIGHT_MM / 1000 / 2), DIAGRAM_HEIGHT_MM / 1000 / 2)

    const currentY = localPositionsRef.current
    let nearestIndex = 1
    let nearestDist = Infinity

    for (let i = 1; i < count - 1; i++) {
      const d = Math.sqrt(
        (segLength * i - pointerWorldX) ** 2 +
        ((currentY[i] ?? 0) - clampedY) ** 2
      )
      if (d < nearestDist) {
        nearestDist = d
        nearestIndex = i
      }
    }

    const apexY = clampedY
    const lastIdx = count - 1

    for (let i = 0; i <= nearestIndex; i++) {
      positions[i] = (i / nearestIndex) * apexY
    }
    for (let i = nearestIndex + 1; i <= lastIdx; i++) {
      const t = (i - nearestIndex) / (lastIdx - nearestIndex)
      positions[i] = (1 - t) * apexY
    }
    positions[0] = 0
    positions[lastIdx] = 0

    return positions
  }, [])

  const sendPluck = useCallback((positions: Float32Array) => {
    const workletNode = workletNodeRef.current
    if (!workletNode) return

    const posCopy = new Float32Array(positions)
    const velCopy = new Float32Array(positions.length)
    workletNode.port.postMessage(
      { type: 'pluck', positions: posCopy, velocities: velCopy },
      [posCopy.buffer, velCopy.buffer]
    )

    if (localPositionsRef.current.length === positions.length) {
      localPositionsRef.current.set(positions)
    }
    latestPositionsRef.current = positions
  }, [])

  const renderParameterControl = (definition: ParameterDefinition) => {
    const value = parameters[definition.key]

    return (
      <Div key={definition.key} display="flex" alignItems="center" gap="8px">
        <label style={{ width: 96, color: '#555' }}>{definition.label}</label>
        {definition.kind === 'slider' ? (
          <input
            type="range"
            min={definition.min}
            max={definition.max}
            step={definition.step}
            value={definition.toSlider ? definition.toSlider(value) : value}
            onChange={(e) => {
              const sliderValue = parseFloat(e.target.value)
              setParameterValue(definition.key, definition.fromSlider ? definition.fromSlider(sliderValue) : sliderValue)
            }}
            style={{ flex: 1 }}
          />
        ) : (
          <Div flex="1" height="2px" background="#ddd" />
        )}
        <span style={{ width: 86, textAlign: 'right', color: '#555', fontVariantNumeric: 'tabular-nums' }}>
          {definition.format(value)}
        </span>
      </Div>
    )
  }

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
          dragWasPausedRef.current = simulationPausedRef.current
          setSimulationPausedState(true)
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
          setSimulationPausedState(dragWasPausedRef.current)
          if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId)
          }
        }}
        onPointerCancel={(e) => {
          setSimulationPausedState(dragWasPausedRef.current)
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
        style={{ width: CONTROL_PANEL_WIDTH, maxWidth: 'calc(100vw - 32px)' }}
      >
        {PARAMETER_DEFINITIONS.map(renderParameterControl)}

        <Div display="flex" justifyContent="center" gap="8px" paddingTop="4px">
          {!audioStarted ? (
            <button
              onClick={startAudio}
              style={{ padding: '8px 24px', cursor: 'pointer', fontSize: '14px', borderRadius: 4 }}
            >
              Start Audio
            </button>
          ) : (
            <>
              <button
                onClick={() => setSimulationPausedState(!simulationPaused)}
                style={{ padding: '8px 18px', cursor: 'pointer', fontSize: '14px', borderRadius: 4 }}
              >
                {simulationPaused ? 'Play' : 'Pause'}
              </button>
              <span style={{ color: '#4caf50', fontSize: 13, alignSelf: 'center' }}>
                Audio running - drag the string to pluck
              </span>
            </>
          )}
        </Div>
      </Div>
    </Div>
  )
}

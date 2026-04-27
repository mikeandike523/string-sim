'use strict';

const MIN_POINT_COUNT = 3;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function copyParameters(parameters) {
  return {
    pointCount: Math.round(parameters.pointCount),
    stringMass: parameters.stringMass,
    stringLength: parameters.stringLength,
    stringTension: parameters.stringTension,
    springK: parameters.springK,
    bendingStiffness: parameters.bendingStiffness,
    segmentDamping: parameters.segmentDamping,
    supportDampingRate: parameters.supportDampingRate,
    pickupOutputGain: parameters.pickupOutputGain,
    visualUpdateInterval: Math.max(1, Math.round(parameters.visualUpdateInterval)),
    pickupFraction: parameters.pickupFraction,
    slowMo: Math.max(1, Math.round(parameters.slowMo)),
  };
}

class StringProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._dt = 1 / sampleRate;
    this._initialized = false;
    this._paused = true;
    this._sampleCount = 0;
    this._slowMoCounter = 0;

    this._params = null;
    this._pointCount = 0;
    this._nodeMass = 0;
    this._segLength = 0;
    this._springRestLength = 0;
    this._bendingForceCoeff = 0;
    this._supportDampCoeff = 0;
    this._pickupIndex = 1;

    this._y = new Float64Array(0);
    this._vy = new Float64Array(0);
    this._ay = new Float64Array(0);
    this._yPred = new Float64Array(0);
    this._vyPred = new Float64Array(0);
    this._ayPred = new Float64Array(0);

    this.port.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'initialize_parameters') {
        this._initialize(e.data.parameters, Boolean(e.data.paused));
      } else if (type === 'set_parameter') {
        this._setParameter(e.data.key, e.data.value);
      } else if (type === 'set_parameters') {
        for (const [key, value] of Object.entries(e.data.parameters ?? {})) {
          this._setParameter(key, value, false);
        }
        this._applyDerivedParameters();
      } else if (type === 'set_simulation_paused') {
        this._paused = Boolean(e.data.paused);
      } else if (type === 'pluck') {
        this._applyPluck(e.data.positions, e.data.velocities);
      }
    };
  }

  _initialize(parameters, paused) {
    const nextParams = copyParameters(parameters);
    nextParams.pointCount = Math.max(MIN_POINT_COUNT, nextParams.pointCount);

    this._params = nextParams;
    this._paused = paused;
    this._sampleCount = 0;
    this._slowMoCounter = 0;
    this._allocateState(nextParams.pointCount, false);
    this._applyDerivedParameters();
    this._initialized = true;
    this.port.postMessage({
      type: 'initialized',
      numPoints: this._pointCount,
      parameters: { ...this._params },
      paused: this._paused,
    });
    this._sendVisual();
  }

  _allocateState(pointCount, preserveState) {
    const previousY = this._y;
    const previousVy = this._vy;
    this._pointCount = pointCount;
    this._y = new Float64Array(pointCount);
    this._vy = new Float64Array(pointCount);
    this._ay = new Float64Array(pointCount);
    this._yPred = new Float64Array(pointCount);
    this._vyPred = new Float64Array(pointCount);
    this._ayPred = new Float64Array(pointCount);

    if (preserveState) {
      const copyCount = Math.min(pointCount, previousY.length);
      for (let i = 0; i < copyCount; i++) {
        this._y[i] = previousY[i];
        this._vy[i] = previousVy[i];
      }
      this._y[0] = 0;
      this._vy[0] = 0;
      this._y[pointCount - 1] = 0;
      this._vy[pointCount - 1] = 0;
    }
  }

  _setParameter(key, value, applyDerived = true) {
    if (!this._params || !(key in this._params)) return;

    if (key === 'pointCount') {
      const nextPointCount = Math.max(MIN_POINT_COUNT, Math.round(value));
      if (nextPointCount !== this._pointCount) {
        this._params.pointCount = nextPointCount;
        this._allocateState(nextPointCount, true);
      }
    } else if (key === 'slowMo' || key === 'visualUpdateInterval') {
      this._params[key] = Math.max(1, Math.round(value));
      if (key === 'slowMo') this._slowMoCounter = 0;
    } else {
      this._params[key] = value;
    }

    if (applyDerived) this._applyDerivedParameters();
  }

  _applyDerivedParameters() {
    const p = this._params;
    if (!p) return;

    this._nodeMass = p.stringMass / this._pointCount;
    this._segLength = p.stringLength / (this._pointCount - 1);
    this._springRestLength = this._segLength - p.stringTension / p.springK;
    this._bendingForceCoeff = p.bendingStiffness / Math.pow(this._segLength, 3);
    this._supportDampCoeff = p.supportDampingRate * this._nodeMass * sampleRate;
    this._pickupIndex = clamp(
      Math.round(p.pickupFraction * (this._pointCount - 1)),
      1,
      this._pointCount - 2,
    );
  }

  _applyPluck(positions, velocities) {
    if (!this._initialized || !positions || positions.length !== this._pointCount) return;

    for (let i = 0; i < this._pointCount; i++) {
      this._y[i] = positions[i];
      this._vy[i] = velocities?.[i] ?? 0;
    }
    this._y[0] = 0;
    this._vy[0] = 0;
    this._y[this._pointCount - 1] = 0;
    this._vy[this._pointCount - 1] = 0;
  }

  _computeAccelerations(y, vy, ay) {
    ay.fill(0);
    const p = this._params;
    const sl = this._segLength;
    const srl = this._springRestLength;
    const bfc = this._bendingForceCoeff;
    const nodeMass = this._nodeMass;
    const N = this._pointCount;

    for (let i = 0; i < N - 1; i++) {
      const dy = y[i + 1] - y[i];
      const distance = Math.sqrt(sl * sl + dy * dy);
      if (distance === 0) continue;
      const excessExt = distance - sl;
      const relVy = vy[i + 1] - vy[i];
      const extensionRate = relVy * dy / distance;
      const totalForce = p.springK * excessExt + p.stringTension - p.segmentDamping * extensionRate;
      const forceY = totalForce * dy / distance;
      ay[i] += forceY / nodeMass;
      ay[i + 1] -= forceY / nodeMass;
    }

    for (let i = 2; i < N - 2; i++) {
      const biharmonicY =
        y[i - 2] - 4 * y[i - 1] + 6 * y[i] - 4 * y[i + 1] + y[i + 2];
      ay[i] += (-bfc * biharmonicY) / nodeMass;
    }

    ay[1] += (-this._supportDampCoeff * vy[1]) / nodeMass;
    ay[N - 2] += (-this._supportDampCoeff * vy[N - 2]) / nodeMass;

    ay[0] = 0;
    ay[N - 1] = 0;
  }

  _step() {
    const dt = this._dt;
    const N = this._pointCount;
    const { _y: y, _vy: vy, _ay: ay, _yPred: yp, _vyPred: vyp, _ayPred: ayp } = this;

    this._computeAccelerations(y, vy, ay);

    for (let i = 1; i < N - 1; i++) {
      vyp[i] = vy[i] + ay[i] * dt;
      yp[i] = y[i] + vy[i] * dt + 0.5 * ay[i] * dt * dt;
    }
    yp[0] = 0;
    vyp[0] = 0;
    yp[N - 1] = 0;
    vyp[N - 1] = 0;

    this._computeAccelerations(yp, vyp, ayp);

    for (let i = 1; i < N - 1; i++) {
      const vyNext = vy[i] + 0.5 * (ay[i] + ayp[i]) * dt;
      y[i] = y[i] + 0.5 * (vy[i] + vyNext) * dt;
      vy[i] = vyNext;
    }
    y[0] = 0;
    vy[0] = 0;
    y[N - 1] = 0;
    vy[N - 1] = 0;
  }

  _sendVisual() {
    if (!this._initialized) return;

    const pos = new Float32Array(this._pointCount);
    for (let i = 0; i < this._pointCount; i++) pos[i] = this._y[i];
    this.port.postMessage({ type: 'visual_update', positions: pos }, [pos.buffer]);
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    const len = out.length;

    if (!this._initialized) {
      out.fill(0);
      return true;
    }

    const slowMo = this._params.slowMo;
    for (let s = 0; s < len; s++) {
      if (!this._paused) {
        if (slowMo === 1) {
          this._step();
        } else if (++this._slowMoCounter >= slowMo) {
          this._slowMoCounter = 0;
          this._step();
        }
      }
      out[s] = this._vy[this._pickupIndex] * this._params.pickupOutputGain;
      if (++this._sampleCount % this._params.visualUpdateInterval === 0) {
        this._sendVisual();
      }
    }
    return true;
  }
}

registerProcessor('string-processor', StringProcessor);

'use strict';

const NUM_POINTS = 50;
const STRING_MASS = 3e-3;    // kg  (was 4.5e-3; lighter → higher pitch)
const INITIAL_STRING_LENGTH = 1;
const STRING_TENSION = 40;   // N   (was 15; tighter → higher pitch)
const K = 30_000;            // N/m (was 30_000; must decrease with mass to stay stable)
const BENDING_STIFFNESS = 3e-8; // N m² (was 2e-8; less inharmonicity → cleaner harmonics)
const SEGMENT_DAMPING = 0.25;  // was 0.5; r_rel_max = 2*c*(dy/d)²*dt/m — keeping below 2 prevents blowup at large pluck amplitudes
// SUPPORT_DAMPING is a dimensionless per-step rate (r parameter).
// There are two r values that give the same per-step decay of 0.883×:
//   r ≈ 1.876 (the old implicit value at 48 kHz) — goes unstable (r > 2) at 44100 Hz
//   r ≈ 0.124 (the stable twin)                  — unconditionally stable at all sample rates
// We store the coefficient as SUPPORT_DAMPING_RATE × NODE_MASS × sampleRate so r = SUPPORT_DAMPING_RATE
// regardless of hardware sample rate, and the acoustic decay is unchanged.
const SUPPORT_DAMPING_RATE = 0.75;

const NODE_MASS = STRING_MASS / NUM_POINTS;
// These three are length-dependent and become instance fields (see constructor)

// Send a visual snapshot every N samples (~11 ms at 48 kHz)
const VISUAL_UPDATE_INTERVAL = 512;
// Velocity pickup: scale vy to audio range; GainNode handles final volume.
// Keep this conservative so the output stays below full scale at typical pluck sizes.
const PICKUP_OUTPUT_GAIN = 2.5;

class StringProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._dt = 1 / sampleRate; // sampleRate is a global in AudioWorkletGlobalScope
    this._supportDampCoeff = SUPPORT_DAMPING_RATE * NODE_MASS * sampleRate;
    this._y = new Float64Array(NUM_POINTS);
    this._vy = new Float64Array(NUM_POINTS);
    this._ay = new Float64Array(NUM_POINTS);
    this._yPred = new Float64Array(NUM_POINTS);
    this._vyPred = new Float64Array(NUM_POINTS);
    this._ayPred = new Float64Array(NUM_POINTS);
    this._paused = false;
    this._pickupIndex = 20;
    this._sampleCount = 0;
    this._slowMo = 1;        // samples per sim step; 1 = real-time
    this._slowMoCounter = 0; // counts samples since last step (only used when _slowMo > 1)
    // Length-dependent physics — updatable via 'set_params' message
    const initialSegLength = INITIAL_STRING_LENGTH / (NUM_POINTS - 1);
    this._segLength = initialSegLength;
    this._springRestLength = initialSegLength - STRING_TENSION / K;
    this._bendingForceCoeff = BENDING_STIFFNESS / Math.pow(initialSegLength, 3);
    this.port.postMessage({ type: 'config', numPoints: NUM_POINTS });

    this.port.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'pluck') {
        const { positions, velocities } = e.data;
        for (let i = 0; i < NUM_POINTS; i++) {
          this._y[i] = positions[i];
          this._vy[i] = velocities[i];
        }
      } else if (type === 'set_pickup') {
        this._pickupIndex = e.data.index;
      } else if (type === 'set_paused') {
        this._paused = e.data.paused;
      } else if (type === 'set_slowmo') {
        this._slowMo = Math.max(1, Math.round(e.data.value));
        this._slowMoCounter = 0;
      } else if (type === 'set_params') {
        this._segLength = e.data.segLength;
        this._springRestLength = e.data.segLength - STRING_TENSION / K;
        this._bendingForceCoeff = e.data.bendingForceCoeff;
        // Preserve the current transverse state when the scale changes.
        // The geometry changes immediately and the simulation settles from there.
      }
    };
  }

  _computeAccelerations(y, vy, ay) {
    ay.fill(0);
    const sl = this._segLength;
    const srl = this._springRestLength;
    const bfc = this._bendingForceCoeff;

    for (let i = 0; i < NUM_POINTS - 1; i++) {
      const dy = y[i + 1] - y[i];
      const distance = Math.sqrt(sl * sl + dy * dy);
      if (distance === 0) continue;
      // Split K*(d − SRL) = K*(d − SEG_LENGTH) + STRING_TENSION.
      // K*(d − SEG_LENGTH) is the excess extension from neutral (small near equilibrium).
      // STRING_TENSION is the equilibrium tension contribution; keeping it separate
      // means the left/right tension terms cancel as a difference of near-unit vectors
      // rather than as a difference of two large force magnitudes — critical for 2D stability.
      const excessExt = distance - sl;
      const relVy = vy[i + 1] - vy[i];
      const extensionRate = relVy * dy / distance;
      const totalForce = K * excessExt + STRING_TENSION - SEGMENT_DAMPING * extensionRate;
      const forceY = totalForce * dy / distance;
      ay[i] += forceY / NODE_MASS;
      ay[i + 1] -= forceY / NODE_MASS;
    }

    for (let i = 2; i < NUM_POINTS - 2; i++) {
      const biharmonicY =
        y[i - 2] - 4 * y[i - 1] + 6 * y[i] - 4 * y[i + 1] + y[i + 2];
      ay[i] += (-bfc * biharmonicY) / NODE_MASS;
    }

    // Support damping at near-end nodes (energy leakage into supports)
    ay[1] += (-this._supportDampCoeff * vy[1]) / NODE_MASS;
    ay[NUM_POINTS - 2] += (-this._supportDampCoeff * vy[NUM_POINTS - 2]) / NODE_MASS;

    ay[0] = 0;
    ay[NUM_POINTS - 1] = 0;
  }

  _step() {
    const dt = this._dt;
    const N = NUM_POINTS;
    const { _y: y, _vy: vy, _ay: ay, _yPred: yp, _vyPred: vyp, _ayPred: ayp } = this;

    this._computeAccelerations(y, vy, ay);

    // Predictor
    for (let i = 1; i < N - 1; i++) {
      vyp[i] = vy[i] + ay[i] * dt;
      yp[i] = y[i] + vy[i] * dt + 0.5 * ay[i] * dt * dt;
    }
    yp[0] = 0; vyp[0] = 0;
    yp[N - 1] = 0; vyp[N - 1] = 0;

    this._computeAccelerations(yp, vyp, ayp);

    // Corrector (trapezoidal)
    for (let i = 1; i < N - 1; i++) {
      const vyNext = vy[i] + 0.5 * (ay[i] + ayp[i]) * dt;
      y[i] = y[i] + 0.5 * (vy[i] + vyNext) * dt;
      vy[i] = vyNext;
    }
    y[0] = 0; vy[0] = 0;
    y[N - 1] = 0; vy[N - 1] = 0;
  }

  _sendVisual() {
    const pos = new Float32Array(NUM_POINTS);
    for (let i = 0; i < NUM_POINTS; i++) pos[i] = this._y[i];
    // Transfer buffer to avoid copying on the receiving side
    this.port.postMessage({ type: 'visual_update', positions: pos }, [pos.buffer]);
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    const len = out.length;
    for (let s = 0; s < len; s++) {
      if (!this._paused) {
        if (this._slowMo === 1) {
          // Fast path: step every sample, no counter overhead
          this._step();
        } else if (++this._slowMoCounter >= this._slowMo) {
          this._slowMoCounter = 0;
          this._step();
        }
      }
      out[s] = this._vy[this._pickupIndex] * PICKUP_OUTPUT_GAIN;
      if (++this._sampleCount % VISUAL_UPDATE_INTERVAL === 0) {
        this._sendVisual();
      }
    }
    return true;
  }
}

registerProcessor('string-processor', StringProcessor);

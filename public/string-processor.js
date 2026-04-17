'use strict';

const NUM_POINTS = 100;
const STRING_MASS = 1e-3;    // kg  (was 4.5e-3; lighter → higher pitch)
const STRING_LENGTH = 1;
const STRING_TENSION = 50;   // N   (was 15; tighter → higher pitch)
const K = 12_000;            // N/m (was 30_000; must decrease with mass to stay stable)
const BENDING_STIFFNESS = 1e-9; // N m² (was 2e-8; less inharmonicity → cleaner harmonics)
const SEGMENT_DAMPING = 0.3;  // was 0.5; r_rel_max = 2*c*(dy/d)²*dt/m — keeping below 2 prevents blowup at large pluck amplitudes
// SUPPORT_DAMPING is a dimensionless per-step rate (r parameter).
// There are two r values that give the same per-step decay of 0.883×:
//   r ≈ 1.876 (the old implicit value at 48 kHz) — goes unstable (r > 2) at 44100 Hz
//   r ≈ 0.124 (the stable twin)                  — unconditionally stable at all sample rates
// We store the coefficient as SUPPORT_DAMPING_RATE × NODE_MASS × sampleRate so r = SUPPORT_DAMPING_RATE
// regardless of hardware sample rate, and the acoustic decay is unchanged.
const SUPPORT_DAMPING_RATE = 0.124;

const NODE_MASS = STRING_MASS / NUM_POINTS;
// These three are length-dependent and become instance fields (see constructor)
const DEFAULT_SEG_LENGTH = STRING_LENGTH / (NUM_POINTS - 1);
const DEFAULT_SPRING_REST_LENGTH = DEFAULT_SEG_LENGTH - STRING_TENSION / K;
const DEFAULT_BENDING_FORCE_COEFF = BENDING_STIFFNESS / Math.pow(DEFAULT_SEG_LENGTH, 3);

// Send a visual snapshot every N samples (~11 ms at 48 kHz)
const VISUAL_UPDATE_INTERVAL = 512;
// Velocity pickup: scale vy to audio range; GainNode handles final volume
const AUDIO_SCALE = 10;
// Toggle x-coordinate simulation: false = string moves in y only; true = 2D motion
const SIMULATE_X = true;

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
    // x-coordinate arrays (only used if SIMULATE_X is true)
    this._x = new Float64Array(NUM_POINTS);
    this._vx = new Float64Array(NUM_POINTS);
    this._ax = new Float64Array(NUM_POINTS);
    this._xPred = new Float64Array(NUM_POINTS);
    this._vxPred = new Float64Array(NUM_POINTS);
    this._axPred = new Float64Array(NUM_POINTS);
    this._paused = false;
    this._pickupIndex = 20;
    this._sampleCount = 0;
    this._slowMo = 1;        // samples per sim step; 1 = real-time
    this._slowMoCounter = 0; // counts samples since last step (only used when _slowMo > 1)
    // Length-dependent physics — updatable via 'set_params' message
    this._segLength = DEFAULT_SEG_LENGTH;
    this._springRestLength = DEFAULT_SPRING_REST_LENGTH;
    this._bendingForceCoeff = DEFAULT_BENDING_FORCE_COEFF;
    // Initialize x positions to equilibrium spacing
    if (SIMULATE_X) {
      for (let i = 0; i < NUM_POINTS; i++) {
        this._x[i] = i * this._segLength;
      }
    }

    this.port.onmessage = (e) => {
      const { type } = e.data;
      if (type === 'pluck') {
        const { positions, velocities } = e.data;
        for (let i = 0; i < NUM_POINTS; i++) {
          this._y[i] = positions[i];
          this._vy[i] = velocities[i];
          if (SIMULATE_X) {
            this._vx[i] = 0; // no x-velocity on pluck
          }
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
        // New length → old positions are physically meaningless; reset to rest
        this._y.fill(0);
        this._vy.fill(0);
        if (SIMULATE_X) {
          for (let i = 0; i < NUM_POINTS; i++) {
            this._x[i] = i * this._segLength;
          }
          this._vx.fill(0);
        }
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

  _computeAccelerationsX(x, vx, ax) {
    // Compute x-direction forces: springs and damping (no bending in x for simplicity)
    ax.fill(0);
    const sl = this._segLength;
    const srl = this._springRestLength;

    for (let i = 0; i < NUM_POINTS - 1; i++) {
      const dx = x[i + 1] - x[i];
      const dy = this._y[i + 1] - this._y[i];
      const distance = Math.sqrt(sl * sl + dy * dy);
      if (distance === 0) continue;
      const excessExt = distance - sl;
      const relVx = vx[i + 1] - vx[i];
      const extensionRate = relVx * dx / distance;
      const totalForce = K * excessExt + STRING_TENSION - SEGMENT_DAMPING * extensionRate;
      const forceX = totalForce * dx / distance;
      ax[i] += forceX / NODE_MASS;
      ax[i + 1] -= forceX / NODE_MASS;
    }

    // Support damping at near-end nodes
    ax[1] += (-this._supportDampCoeff * vx[1]) / NODE_MASS;
    ax[NUM_POINTS - 2] += (-this._supportDampCoeff * vx[NUM_POINTS - 2]) / NODE_MASS;

    ax[0] = 0;
    ax[NUM_POINTS - 1] = 0;
  }

  _step() {
    const dt = this._dt;
    const N = NUM_POINTS;
    const { _y: y, _vy: vy, _ay: ay, _yPred: yp, _vyPred: vyp, _ayPred: ayp } = this;

    this._computeAccelerations(y, vy, ay);
    if (SIMULATE_X) {
      this._computeAccelerationsX(this._x, this._vx, this._ax);
    }

    // Predictor
    for (let i = 1; i < N - 1; i++) {
      vyp[i] = vy[i] + ay[i] * dt;
      yp[i] = y[i] + vy[i] * dt + 0.5 * ay[i] * dt * dt;
      if (SIMULATE_X) {
        this._vxPred[i] = this._vx[i] + this._ax[i] * dt;
        this._xPred[i] = this._x[i] + this._vx[i] * dt + 0.5 * this._ax[i] * dt * dt;
      }
    }
    yp[0] = 0; vyp[0] = 0;
    yp[N - 1] = 0; vyp[N - 1] = 0;
    if (SIMULATE_X) {
      this._xPred[0] = 0; this._vxPred[0] = 0;
      this._xPred[N - 1] = (N - 1) * this._segLength; this._vxPred[N - 1] = 0;
    }

    this._computeAccelerations(yp, vyp, ayp);
    if (SIMULATE_X) {
      this._computeAccelerationsX(this._xPred, this._vxPred, this._axPred);
    }

    // Corrector (trapezoidal)
    for (let i = 1; i < N - 1; i++) {
      const vyNext = vy[i] + 0.5 * (ay[i] + ayp[i]) * dt;
      y[i] = y[i] + 0.5 * (vy[i] + vyNext) * dt;
      vy[i] = vyNext;
      if (SIMULATE_X) {
        const vxNext = this._vx[i] + 0.5 * (this._ax[i] + this._axPred[i]) * dt;
        this._x[i] = this._x[i] + 0.5 * (this._vx[i] + vxNext) * dt;
        this._vx[i] = vxNext;
      }
    }
    y[0] = 0; vy[0] = 0;
    y[N - 1] = 0; vy[N - 1] = 0;
    if (SIMULATE_X) {
      this._x[0] = 0; this._vx[0] = 0;
      this._x[N - 1] = (N - 1) * this._segLength; this._vx[N - 1] = 0;
    }
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
      out[s] = this._vy[this._pickupIndex] * AUDIO_SCALE;
      if (++this._sampleCount % VISUAL_UPDATE_INTERVAL === 0) {
        this._sendVisual();
      }
    }
    return true;
  }
}

registerProcessor('string-processor', StringProcessor);

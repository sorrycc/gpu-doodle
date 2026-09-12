// One workgroup evaluates one sketch; lane = hidden channel. Mirrors cpu.ts
// stage for stage, in the same accumulation order, so both paths agree to
// float rounding. Tensor offsets, the dimensions, the state type and the body
// of rounded() are spliced in by shader-source.ts before compilation.
struct Parameters {
  pointCount: u32,
  streamCount: u32,
  padding0: u32,
  padding1: u32,
}
struct Stream { start: u32, count: u32 }

@group(0) @binding(0) var<storage, read> features: array<f32>;
@group(0) @binding(1) var<storage, read> streams: array<Stream>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@group(0) @binding(3) var<storage, read> modelWeights: array<f32>;
@group(0) @binding(4) var<storage, read_write> stateData: array<STATE_TYPE>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;

var<workgroup> streamInfo: vec2<u32>;
var<workgroup> pooled: array<f32, HIDDEN2>;
var<workgroup> headHidden: array<f32, HEAD>;

fn rounded(value: f32) -> f32 { ROUND_BODY }
fn sigmoid(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
// Four buffers follow tensor lifetimes: embedded -> forward, encoded,
// gate, candidate -> backward. Barriers precede cross-lane reads.
fn readState(stage: u32, token: u32, channel: u32) -> f32 {
  return f32(stateData[(stage * parameters.pointCount + token) * HIDDEN + channel]);
}
fn writeState(stage: u32, token: u32, channel: u32, value: f32) {
  stateData[(stage * parameters.pointCount + token) * HIDDEN + channel] = STATE_TYPE(rounded(value));
}

@compute @workgroup_size(HIDDEN)
fn classify(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
  let streamIndex = group.x + group.y * 65535u;
  if (streamIndex >= parameters.streamCount) { return; }
  let lane = local.x;
  if (lane == 0u) { streamInfo = vec2<u32>(streams[streamIndex].start, streams[streamIndex].count); }
  workgroupBarrier();
  let info = workgroupUniformLoad(&streamInfo);
  let start = info.x;
  let count = info.y;

  // Input projection: dot first, bias last, as cpu.ts does.
  for (var position = 0u; position < count; position++) {
    let token = start + position;
    var value = 0.0;
    for (var feature = 0u; feature < FEATURES; feature++) {
      value += modelWeights[INPUT_WEIGHT_OFFSET + lane * FEATURES + feature] * features[token * FEATURES + feature];
    }
    writeState(0u, token, lane, value + modelWeights[INPUT_BIAS_OFFSET + lane]);
  }
  storageBarrier();
  workgroupBarrier();

  // Depthwise convolution, width 5, then tanh.
  for (var position = 0u; position < count; position++) {
    var value = modelWeights[ENCODER_BIAS_OFFSET + lane];
    for (var tap = 0u; tap < 5u; tap++) {
      let neighbor = i32(position) + i32(tap) - 2;
      if (neighbor >= 0 && neighbor < i32(count)) {
        value += readState(0u, start + u32(neighbor), lane) * modelWeights[CONVOLUTION_OFFSET + tap * HIDDEN + lane];
      }
    }
    writeState(1u, start + position, lane, tanh(value));
  }
  storageBarrier();
  workgroupBarrier();

  // Gate and candidate, then the forward scan into stage 0.
  var state = 0.0;
  for (var position = 0u; position < count; position++) {
    let token = start + position;
    var gateValue = 0.0;
    var candidateValue = 0.0;
    for (var channel = 0u; channel < HIDDEN; channel++) {
      let encoded = readState(1u, token, channel);
      gateValue += modelWeights[GATE_WEIGHT_OFFSET + lane * HIDDEN + channel] * encoded;
      candidateValue += modelWeights[CANDIDATE_WEIGHT_OFFSET + lane * HIDDEN + channel] * encoded;
    }
    let gate = rounded(sigmoid(gateValue + modelWeights[GATE_BIAS_OFFSET + lane]));
    let candidate = rounded((1.0 - gate) * tanh(candidateValue + modelWeights[CANDIDATE_BIAS_OFFSET + lane]));
    writeState(2u, token, lane, gate);
    writeState(3u, token, lane, candidate);
    state = rounded(gate * state + candidate);
    writeState(0u, token, lane, state);
  }
  // Backward scan into stage 3.
  state = 0.0;
  for (var position = i32(count) - 1; position >= 0; position--) {
    let token = start + u32(position);
    state = rounded(readState(2u, token, lane) * state + readState(3u, token, lane));
    writeState(3u, token, lane, state);
  }
  storageBarrier();
  workgroupBarrier();

  // Combine, with mean and max pooling over the sequence.
  var sum = 0.0;
  var maximum = -2.0;
  for (var position = 0u; position < count; position++) {
    let token = start + position;
    var value = modelWeights[COMBINE_BIAS_OFFSET + lane];
    for (var channel = 0u; channel < HIDDEN; channel++) {
      value += modelWeights[COMBINE_WEIGHT_OFFSET + lane * HIDDEN2 + channel] * readState(0u, token, channel);
      value += modelWeights[COMBINE_WEIGHT_OFFSET + lane * HIDDEN2 + HIDDEN + channel] * readState(3u, token, channel);
    }
    let combined = rounded(tanh(readState(1u, token, lane) + value));
    sum += combined;
    maximum = max(maximum, combined);
  }
  pooled[lane] = rounded(sum / f32(count));
  pooled[HIDDEN + lane] = rounded(maximum);
  workgroupBarrier();

  // Head: one lane per unit.
  for (var unit = lane; unit < HEAD; unit += HIDDEN) {
    var value = 0.0;
    for (var index = 0u; index < HIDDEN2; index++) {
      value += modelWeights[HEAD_WEIGHT_OFFSET + unit * HIDDEN2 + index] * pooled[index];
    }
    headHidden[unit] = rounded(tanh(value + modelWeights[HEAD_BIAS_OFFSET + unit]));
  }
  workgroupBarrier();

  for (var label = lane; label < CLASSES; label += HIDDEN) {
    var value = 0.0;
    for (var channel = 0u; channel < HEAD; channel++) {
      value += modelWeights[OUTPUT_WEIGHT_OFFSET + label * HEAD + channel] * headHidden[channel];
    }
    logits[streamIndex * CLASSES + label] = value + modelWeights[OUTPUT_BIAS_OFFSET + label];
  }
}

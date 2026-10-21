export const STATES = new Set(['idle', 'working', 'complete', 'error']);
export const DIRECTIONS = Object.freeze({ front: 0, right: Math.PI / 2, back: Math.PI, left: -Math.PI / 2 });
export function resolveDirection(value) {
  const yaw = typeof value === 'string' ? DIRECTIONS[value] : value;
  if (!Number.isFinite(yaw)) throw new TypeError('Direction must be front, right, back, left, or a finite angle in radians.');
  return yaw;
}
export function damp(current, target, rate, dt) { return current + (target - current) * (1 - Math.exp(-rate * dt)); }
export function turn(current, target, rate, dt) {
  const delta = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  return current + delta * (1 - Math.exp(-rate * dt));
}
export function validateState(state) {
  if (!STATES.has(state)) throw new TypeError(`Unknown state: ${state}`);
  return state;
}

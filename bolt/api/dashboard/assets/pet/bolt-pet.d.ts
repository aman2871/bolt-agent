export type BoltState = 'idle' | 'working' | 'complete' | 'error';
export type BoltDirection = 'front' | 'right' | 'back' | 'left' | number;
export interface BoltOptions {
  state?: BoltState;
  direction?: BoltDirection;
  roam?: boolean;
  maxDpr?: number;
  turnSpeed?: number;
  reducedMotion?: boolean;
  autoStart?: boolean;
  fur?: 'auto' | 'off' | 'high';
  onStateChange?: (state: BoltState) => void;
  onContextLost?: () => void;
}
export interface BoltPet {
  /** Set the container size in CSS pixels. One argument creates a square. */
  setSize(width: number, height?: number): void;
  setState(state: BoltState): void;
  setDirection(direction: BoltDirection): void;
  setRoaming(enabled: boolean): void;
  /** Move to a clamped horizontal position in scene world units, not CSS pixels. */
  moveTo(x: number): void;
  pause(): void;
  resume(): void;
  /** Advance a paused instance, e.g. for deterministic recordings or host-controlled loops. */
  step(seconds: number): void;
  resize(): void;
  getState(): {state: BoltState; direction: number; yaw: number; x: number; roam: boolean; paused: boolean; reducedMotion: boolean};
  dispose(): void;
}
export function createBoltPet(container: HTMLElement, options?: BoltOptions): BoltPet;

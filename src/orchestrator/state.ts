import { randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { z } from 'zod';

export const PIPELINE_STAGES = [
  'queued',
  'ingesting',
  'memory_recall',
  'grounding',
  'hunting',
  'evaluating',
  'localizing',
  'fixing',
  'verifying',
  'drafting_issue',
  'drafting_pr',
  'completed',
  'rejected',
  'failed',
] as const;

export const PipelineStageSchema = z.enum(PIPELINE_STAGES);
export type PipelineStage = z.infer<typeof PipelineStageSchema>;

export const ISSUE_STATUSES = [
  'pending',
  'running',
  'paused',
  'completed',
  'rejected',
  'failed',
] as const;

export const IssueStatusSchema = z.enum(ISSUE_STATUSES);
export type IssueStatus = z.infer<typeof IssueStatusSchema>;

export const StageTransitionSchema = z.object({
  fromStage: PipelineStageSchema.nullable(),
  toStage: PipelineStageSchema,
  timestamp: z.string(),
  reason: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type StageTransition = z.infer<typeof StageTransitionSchema>;

export const IssueMetricsSchema = z.object({
  llmCalls: z.number().int().nonnegative(),
  inputTokens: z.number().int().nonnegative(),
  outputTokens: z.number().int().nonnegative(),
  durationMs: z.number().nonnegative(),
});
export type IssueMetrics = z.infer<typeof IssueMetricsSchema>;

export const IssueStateSchema = z.object({
  id: z.string().min(1),
  repo: z.string().min(1),
  stage: PipelineStageSchema,
  status: IssueStatusSchema,
  title: z.string().optional(),
  hypothesisId: z.string().optional(),
  sourceIssueNumber: z.number().int().positive().optional(),
  sourceIssueUrl: z.string().url().optional(),
  branch: z.string().optional(),
  attempts: z.number().int().nonnegative(),
  createdAt: z.string(),
  updatedAt: z.string(),
  error: z.string().optional(),
  history: z.array(StageTransitionSchema),
  metrics: IssueMetricsSchema,
});
export type IssueState = z.infer<typeof IssueStateSchema>;

export type StateEventType =
  | 'state:created'
  | 'state:updated'
  | 'state:stage_changed'
  | 'state:error'
  | 'state:removed';

export interface StateEvent {
  type: StateEventType;
  issueId: string;
  timestamp: string;
  state: IssueState;
  previousStage?: PipelineStage;
}

export interface CreateIssueStateInput {
  id?: string;
  repo: string;
  stage?: PipelineStage;
  status?: IssueStatus;
  title?: string;
  hypothesisId?: string;
  sourceIssueNumber?: number;
  sourceIssueUrl?: string;
  branch?: string;
  reason?: string;
}

export interface TransitionOptions {
  reason?: string;
  status?: IssueStatus;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface IssueListFilter {
  repo?: string;
  stage?: PipelineStage;
  status?: IssueStatus;
}

const TERMINAL_STAGE_TO_STATUS: Partial<Record<PipelineStage, IssueStatus>> = {
  completed: 'completed',
  rejected: 'rejected',
  failed: 'failed',
};

function cloneIssueState(state: IssueState): IssueState {
  return structuredClone(state);
}

export class StateManager {
  private readonly states = new Map<string, IssueState>();
  private readonly emitter = new EventEmitter();

  constructor(initialStates: IssueState[] = []) {
    for (const state of initialStates) {
      const parsed = IssueStateSchema.parse(state);
      this.states.set(parsed.id, cloneIssueState(parsed));
    }
  }

  create(input: CreateIssueStateInput): IssueState {
    const id = input.id ?? randomUUID();
    if (this.states.has(id)) {
      throw new Error(`IssueState with id "${id}" already exists`);
    }

    const now = new Date().toISOString();
    const initialStage: PipelineStage = input.stage ?? 'queued';
    const inferredStatus: IssueStatus =
      input.status ??
      TERMINAL_STAGE_TO_STATUS[initialStage] ??
      (initialStage === 'queued' ? 'pending' : 'running');

    const state: IssueState = {
      id,
      repo: input.repo,
      stage: initialStage,
      status: inferredStatus,
      title: input.title,
      hypothesisId: input.hypothesisId,
      sourceIssueNumber: input.sourceIssueNumber,
      sourceIssueUrl: input.sourceIssueUrl,
      branch: input.branch,
      attempts: 0,
      createdAt: now,
      updatedAt: now,
      history: [
        {
          fromStage: null,
          toStage: initialStage,
          timestamp: now,
          reason: input.reason ?? 'Initialized issue state',
        },
      ],
      metrics: {
        llmCalls: 0,
        inputTokens: 0,
        outputTokens: 0,
        durationMs: 0,
      },
    };

    this.states.set(id, state);
    const snapshot = cloneIssueState(state);
    this.emitEvent({
      type: 'state:created',
      issueId: id,
      timestamp: now,
      state: snapshot,
    });
    return snapshot;
  }

  get(id: string): IssueState | undefined {
    const found = this.states.get(id);
    return found ? cloneIssueState(found) : undefined;
  }

  require(id: string): IssueState {
    const found = this.states.get(id);
    if (!found) {
      throw new Error(`IssueState not found for id "${id}"`);
    }
    return cloneIssueState(found);
  }

  has(id: string): boolean {
    return this.states.has(id);
  }

  list(filter?: IssueListFilter): IssueState[] {
    const results: IssueState[] = [];
    for (const state of this.states.values()) {
      if (filter?.repo && state.repo !== filter.repo) continue;
      if (filter?.stage && state.stage !== filter.stage) continue;
      if (filter?.status && state.status !== filter.status) continue;
      results.push(cloneIssueState(state));
    }
    return results;
  }

  transition(
    id: string,
    toStage: PipelineStage,
    options: TransitionOptions = {},
  ): IssueState {
    const current = this.states.get(id);
    if (!current) {
      throw new Error(`Cannot transition unknown IssueState "${id}"`);
    }

    const now = new Date().toISOString();
    const previousStage = current.stage;
    const nextStatus =
      options.status ?? TERMINAL_STAGE_TO_STATUS[toStage] ?? 'running';

    const transitionEntry: StageTransition = {
      fromStage: previousStage,
      toStage,
      timestamp: now,
      reason: options.reason,
      metadata: options.metadata,
    };

    const nextState: IssueState = {
      ...current,
      stage: toStage,
      status: nextStatus,
      updatedAt: now,
      error: options.error ?? (toStage === 'failed' ? current.error : undefined),
      history: [...current.history, transitionEntry],
    };

    this.states.set(id, nextState);
    const snapshot = cloneIssueState(nextState);
    this.emitEvent({
      type: toStage === 'failed' ? 'state:error' : 'state:stage_changed',
      issueId: id,
      timestamp: now,
      state: snapshot,
      previousStage,
    });
    return snapshot;
  }

  update(
    id: string,
    patch: Partial<Omit<IssueState, 'id' | 'createdAt' | 'history'>>,
  ): IssueState {
    const current = this.states.get(id);
    if (!current) {
      throw new Error(`Cannot update unknown IssueState "${id}"`);
    }

    const now = new Date().toISOString();
    const nextState: IssueState = {
      ...current,
      ...patch,
      id: current.id,
      createdAt: current.createdAt,
      history: current.history,
      metrics: patch.metrics ? { ...current.metrics, ...patch.metrics } : current.metrics,
      updatedAt: now,
    };

    this.states.set(id, nextState);
    const snapshot = cloneIssueState(nextState);
    this.emitEvent({
      type: 'state:updated',
      issueId: id,
      timestamp: now,
      state: snapshot,
    });
    return snapshot;
  }

  recordMetrics(id: string, delta: Partial<IssueMetrics>): IssueState {
    const current = this.states.get(id);
    if (!current) {
      throw new Error(`Cannot record metrics for unknown IssueState "${id}"`);
    }

    return this.update(id, {
      metrics: {
        llmCalls: current.metrics.llmCalls + (delta.llmCalls ?? 0),
        inputTokens: current.metrics.inputTokens + (delta.inputTokens ?? 0),
        outputTokens: current.metrics.outputTokens + (delta.outputTokens ?? 0),
        durationMs: current.metrics.durationMs + (delta.durationMs ?? 0),
      },
    });
  }

  fail(
    id: string,
    error: string,
    metadata?: Record<string, unknown>,
  ): IssueState {
    return this.transition(id, 'failed', {
      status: 'failed',
      reason: error,
      error,
      metadata,
    });
  }

  remove(id: string): boolean {
    const existing = this.states.get(id);
    if (!existing) {
      return false;
    }

    this.states.delete(id);
    this.emitEvent({
      type: 'state:removed',
      issueId: id,
      timestamp: new Date().toISOString(),
      state: cloneIssueState(existing),
    });
    return true;
  }

  subscribe(listener: (event: StateEvent) => void): () => void {
    this.emitter.on('event', listener);
    return () => {
      this.emitter.off('event', listener);
    };
  }

  toJSON(): IssueState[] {
    return this.list();
  }

  private emitEvent(event: StateEvent): void {
    this.emitter.emit('event', event);
  }
}

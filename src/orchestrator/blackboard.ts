import { randomUUID } from 'node:crypto';
import { z } from 'zod';

export const DEFAULT_META_BASE_URL = 'https://api.meta.ai/v1';
export const DEFAULT_META_MODEL = 'muse-ember-1.0';

export const ModelConfigSchema = z.object({
  provider: z.literal('meta'),
  baseURL: z.string().url(),
  model: z.string().min(1),
  apiKeyEnv: z.string().min(1),
  temperature: z.number().min(0).max(2).optional(),
  maxTokens: z.number().int().positive().optional(),
});
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

export const RepoContextSchema = z.object({
  owner: z.string().min(1),
  name: z.string().min(1),
  fullName: z.string().min(1),
  url: z.string().min(1),
  defaultBranch: z.string().min(1),
  localPath: z.string().min(1),
  worktreePath: z.string().optional(),
  commitSha: z.string().optional(),
  language: z.string().optional(),
  languages: z.array(z.string()).optional(),
  packageManager: z.string().optional(),
  buildCommand: z.string().optional(),
  testCommand: z.string().optional(),
  lintCommand: z.string().optional(),
});
export type RepoContext = z.infer<typeof RepoContextSchema>;

export const MemoryLessonSchema = z.object({
  id: z.string(),
  category: z.enum(['pitfall', 'convention', 'pattern', 'false_positive']),
  summary: z.string(),
  repo: z.string().optional(),
  tags: z.array(z.string()),
  confidence: z.number().min(0).max(1),
  createdAt: z.string(),
});
export type MemoryLesson = z.infer<typeof MemoryLessonSchema>;

export const MemoryContextSchema = z.object({
  recalledAt: z.string(),
  pastRunIds: z.array(z.string()),
  lessons: z.array(MemoryLessonSchema),
  repoConventions: z.array(z.string()),
  knownPitfalls: z.array(z.string()),
  rejectedSignatures: z.array(z.string()),
});
export type MemoryContext = z.infer<typeof MemoryContextSchema>;

export const GroundedFileSchema = z.object({
  path: z.string(),
  role: z.string(),
  symbols: z.array(z.string()),
  summary: z.string().optional(),
});
export type GroundedFile = z.infer<typeof GroundedFileSchema>;

export const GroundingContextSchema = z.object({
  architectureSummary: z.string(),
  entrypoints: z.array(z.string()),
  testFiles: z.array(z.string()),
  relevantFiles: z.array(GroundedFileSchema),
  dependencies: z.record(z.string(), z.string()),
  baseline: z.object({
    buildCommand: z.string().optional(),
    buildPassed: z.boolean().optional(),
    testCommand: z.string().optional(),
    testsPassed: z.boolean().optional(),
    outputSnippet: z.string().optional(),
  }),
  completedAt: z.string(),
});
export type GroundingContext = z.infer<typeof GroundingContextSchema>;

export const BugSeveritySchema = z.enum(['critical', 'high', 'medium', 'low']);
export type BugSeverity = z.infer<typeof BugSeveritySchema>;

export const BugHypothesisSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string(),
  category: z.string(),
  severity: BugSeveritySchema,
  confidence: z.number().min(0).max(1),
  affectedFiles: z.array(z.string()),
  entryPoint: z.string().optional(),
  reproHypothesis: z.string(),
  evidence: z.array(z.string()),
  sourceIssueNumber: z.number().int().positive().optional(),
});
export type BugHypothesis = z.infer<typeof BugHypothesisSchema>;

export const HunterContextSchema = z.object({
  scanScope: z.array(z.string()),
  hypotheses: z.array(BugHypothesisSchema),
  notes: z.array(z.string()),
  completedAt: z.string(),
});
export type HunterContext = z.infer<typeof HunterContextSchema>;

export const ProofTypeSchema = z.enum([
  'repro_test',
  'runtime_trace',
  'static_proof',
  'unproven',
]);
export type ProofType = z.infer<typeof ProofTypeSchema>;

export const EvaluationVerdictSchema = z.object({
  hypothesisId: z.string(),
  accepted: z.boolean(),
  proofType: ProofTypeSchema,
  confidence: z.number().min(0).max(1),
  reproductionSteps: z.array(z.string()),
  reproCommand: z.string().optional(),
  reproOutput: z.string().optional(),
  failureSignature: z.string().optional(),
  rejectionReason: z.string().optional(),
});
export type EvaluationVerdict = z.infer<typeof EvaluationVerdictSchema>;

export const EvaluatorContextSchema = z.object({
  selectedHypothesisId: z.string().optional(),
  verdicts: z.array(EvaluationVerdictSchema),
  completedAt: z.string(),
});
export type EvaluatorContext = z.infer<typeof EvaluatorContextSchema>;

export const FaultLocationSchema = z.object({
  file: z.string(),
  startLine: z.number().int().positive(),
  endLine: z.number().int().positive(),
  symbol: z.string().optional(),
  role: z.enum(['root_cause', 'propagator', 'symptom', 'test']),
  snippet: z.string().optional(),
});
export type FaultLocation = z.infer<typeof FaultLocationSchema>;

export const LocalizationContextSchema = z.object({
  hypothesisId: z.string(),
  rootCause: z.string(),
  primaryFile: z.string(),
  symbols: z.array(z.string()),
  locations: z.array(FaultLocationSchema),
  callChain: z.array(z.string()),
  relatedTests: z.array(z.string()),
  completedAt: z.string(),
});
export type LocalizationContext = z.infer<typeof LocalizationContextSchema>;

export const FileEditSummarySchema = z.object({
  filePath: z.string(),
  summary: z.string(),
  hunkCount: z.number().int().nonnegative(),
});
export type FileEditSummary = z.infer<typeof FileEditSummarySchema>;

export const FixContextSchema = z.object({
  hypothesisId: z.string(),
  patchId: z.string().optional(),
  branchName: z.string(),
  baseCommitSha: z.string().optional(),
  strategy: z.string(),
  rationale: z.string(),
  diff: z.string(),
  edits: z.array(FileEditSummarySchema),
  iteration: z.number().int().positive(),
  maxIterations: z.number().int().positive(),
  completedAt: z.string(),
});
export type FixContext = z.infer<typeof FixContextSchema>;

export const VerificationCheckSchema = z.object({
  name: z.string(),
  command: z.string(),
  passed: z.boolean(),
  exitCode: z.number().int(),
  durationMs: z.number().nonnegative(),
  stdout: z.string(),
  stderr: z.string(),
});
export type VerificationCheck = z.infer<typeof VerificationCheckSchema>;

export const VerificationContextSchema = z.object({
  passed: z.boolean(),
  reproTestAdded: z.boolean(),
  reproTestPath: z.string().optional(),
  reproFailedBeforeFix: z.boolean(),
  reproPassedAfterFix: z.boolean(),
  suitePassed: z.boolean(),
  typecheckPassed: z.boolean(),
  lintPassed: z.boolean().optional(),
  checks: z.array(VerificationCheckSchema),
  regressionNotes: z.string().optional(),
  completedAt: z.string(),
});
export type VerificationContext = z.infer<typeof VerificationContextSchema>;

export const IssueDraftContextSchema = z.object({
  title: z.string(),
  body: z.string(),
  labels: z.array(z.string()),
  severity: BugSeveritySchema,
  reproductionSteps: z.array(z.string()),
  affectedFiles: z.array(z.string()),
  evidenceSnippet: z.string().optional(),
  issueNumber: z.number().int().positive().optional(),
  publishedUrl: z.string().url().optional(),
  createdAt: z.string(),
});
export type IssueDraftContext = z.infer<typeof IssueDraftContextSchema>;

export const PrDraftContextSchema = z.object({
  title: z.string(),
  body: z.string(),
  headBranch: z.string(),
  baseBranch: z.string(),
  commitMessage: z.string(),
  changedFiles: z.array(z.string()),
  checksSummary: z.string(),
  checklist: z.array(
    z.object({
      item: z.string(),
      checked: z.boolean(),
    }),
  ),
  readyToSubmit: z.boolean(),
  state: z.enum(['draft', 'open', 'merged', 'closed']).optional(),
  prNumber: z.number().int().positive().optional(),
  prUrl: z.string().url().optional(),
  createdAt: z.string(),
});
export type PrDraftContext = z.infer<typeof PrDraftContextSchema>;

export const BlackboardArtifactSchema = z.object({
  id: z.string(),
  kind: z.enum(['patch', 'repro', 'log', 'report', 'memory']),
  label: z.string(),
  path: z.string(),
  createdAt: z.string(),
});
export type BlackboardArtifact = z.infer<typeof BlackboardArtifactSchema>;

export interface TaskBlackboard {
  taskId: string;
  createdAt: string;
  updatedAt: string;
  model: ModelConfig;
  repo: RepoContext;
  memory?: MemoryContext;
  grounding?: GroundingContext;
  hunter?: HunterContext;
  evaluator?: EvaluatorContext;
  localization?: LocalizationContext;
  fix?: FixContext;
  verification?: VerificationContext;
  issueDraft?: IssueDraftContext;
  draft?: PrDraftContext;
  artifacts: BlackboardArtifact[];
  notes: string[];
}

export const TaskBlackboardSchema: z.ZodType<TaskBlackboard> = z.object({
  taskId: z.string().min(1),
  createdAt: z.string(),
  updatedAt: z.string(),
  model: ModelConfigSchema,
  repo: RepoContextSchema,
  memory: MemoryContextSchema.optional(),
  grounding: GroundingContextSchema.optional(),
  hunter: HunterContextSchema.optional(),
  evaluator: EvaluatorContextSchema.optional(),
  localization: LocalizationContextSchema.optional(),
  fix: FixContextSchema.optional(),
  verification: VerificationContextSchema.optional(),
  issueDraft: IssueDraftContextSchema.optional(),
  draft: PrDraftContextSchema.optional(),
  artifacts: z.array(BlackboardArtifactSchema),
  notes: z.array(z.string()),
});

export type BlackboardSectionKey =
  | 'repo'
  | 'memory'
  | 'grounding'
  | 'hunter'
  | 'evaluator'
  | 'localization'
  | 'fix'
  | 'verification'
  | 'issueDraft'
  | 'draft';

export interface CreateTaskBlackboardOptions {
  taskId?: string;
  model?: Partial<ModelConfig>;
  notes?: string[];
}

export function createTaskBlackboard(
  repo: RepoContext,
  options: CreateTaskBlackboardOptions = {},
): TaskBlackboard {
  const now = new Date().toISOString();
  return {
    taskId: options.taskId ?? randomUUID(),
    createdAt: now,
    updatedAt: now,
    model: {
      provider: 'meta',
      baseURL: options.model?.baseURL ?? DEFAULT_META_BASE_URL,
      model: options.model?.model ?? DEFAULT_META_MODEL,
      apiKeyEnv: options.model?.apiKeyEnv ?? 'META_API_KEY',
      temperature: options.model?.temperature,
      maxTokens: options.model?.maxTokens,
    },
    repo: RepoContextSchema.parse(repo),
    artifacts: [],
    notes: options.notes ? [...options.notes] : [],
  };
}

export function setBlackboardSection<K extends BlackboardSectionKey>(
  blackboard: TaskBlackboard,
  section: K,
  value: NonNullable<TaskBlackboard[K]>,
): TaskBlackboard {
  return {
    ...blackboard,
    [section]: value,
    updatedAt: new Date().toISOString(),
  };
}

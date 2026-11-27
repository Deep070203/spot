import { exec, execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import OpenAI from 'openai';
import type { ChatCompletionTool } from 'openai/resources/chat/completions';
import { z } from 'zod';
import {
  DEFAULT_META_BASE_URL,
  DEFAULT_META_MODEL,
  setBlackboardSection,
  type FileEditSummary,
  type FixContext,
  type TaskBlackboard,
} from './blackboard.js';
import type { StateManager } from './state.js';
import { resolveRepoEnv } from '../utils/env.js';
import type { BehavioralMemory } from '../utils/memory.js';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

export const ReadFileArgsSchema = z.object({
  path: z.string().min(1),
  startLine: z.number().int().positive().optional(),
  endLine: z.number().int().positive().optional(),
});

export const WriteFileArgsSchema = z.object({
  path: z.string().min(1),
  content: z.string(),
});

export const EditFileArgsSchema = z.object({
  path: z.string().min(1),
  target: z.string().min(1),
  replacement: z.string(),
});

export const SearchRepoArgsSchema = z.object({
  query: z.string().min(1),
  pathFilter: z.string().optional(),
});

export const RunCommandArgsSchema = z.object({
  command: z.string().min(1),
  timeoutMs: z.number().int().positive().max(180_000).optional(),
});

export const FinishFixArgsSchema = z.object({
  strategy: z.string().min(1),
  rationale: z.string().min(1),
  summary: z.string().min(1),
});

export type CodeFixToolName =
  | 'read_file'
  | 'write_file'
  | 'edit_file'
  | 'search_repo'
  | 'run_command'
  | 'finish_fix';

export const CODEFIX_TOOL_NAMES: readonly CodeFixToolName[] = [
  'read_file',
  'write_file',
  'edit_file',
  'search_repo',
  'run_command',
  'finish_fix',
] as const;

export interface ToolLoopExecutionContext {
  repoDir: string;
  blackboard: TaskBlackboard;
  executeTool: (name: CodeFixToolName, rawArgs: unknown) => Promise<string>;
}

export type AgyRunnerFn = (
  prompt: string,
  context: ToolLoopExecutionContext,
) => Promise<{ strategy: string; rationale: string; summary?: string }>;

export type ToolLoopAgentRunner = (
  context: ToolLoopExecutionContext,
  initialPrompt: string,
) => Promise<{ strategy: string; rationale: string }>;

export interface RunCodeFixOptions {
  branchName?: string;
  maxIterations?: number;
  maxToolTurns?: number;
  useAgy?: boolean;
  agyRunner?: AgyRunnerFn;
  toolLoopRunner?: ToolLoopAgentRunner;
  memory?: BehavioralMemory;
  stateManager?: StateManager;
  issueId?: string;
  apiKey?: string;
}

export interface RunCodeFixResult {
  blackboard: TaskBlackboard;
  fix: FixContext;
  probePreserved: boolean;
}

export interface ProbeSnapshot {
  probeFilePath: string;
  content: string;
}

function resolveInsideRepo(repoDir: string, relPath: string): string {
  const resolvedRepo = path.resolve(repoDir);
  const target = path.resolve(resolvedRepo, relPath);
  if (target !== resolvedRepo && !target.startsWith(`${resolvedRepo}${path.sep}`)) {
    throw new Error(`Path "${relPath}" escapes repository root "${resolvedRepo}"`);
  }
  return target;
}

/**
 * Guardrail ensuring the CodeFix agent never alters, weakens, or deletes
 * the proof-first reproduction probe file.
 */
export async function preserveProbe(
  repoDir: string,
  snapshot: ProbeSnapshot | null,
): Promise<{ preserved: boolean; restored: boolean }> {
  if (!snapshot) {
    return { preserved: true, restored: false };
  }

  const absProbe = resolveInsideRepo(
    repoDir,
    path.relative(repoDir, path.resolve(repoDir, snapshot.probeFilePath)),
  );

  try {
    const current = await fs.readFile(absProbe, 'utf8');
    if (current === snapshot.content) {
      return { preserved: true, restored: false };
    }
  } catch {
    // Probe was deleted; restore below.
  }

  await fs.mkdir(path.dirname(absProbe), { recursive: true });
  await fs.writeFile(absProbe, snapshot.content, 'utf8');
  return { preserved: false, restored: true };
}

async function captureProbeSnapshot(
  repoDir: string,
  blackboard: TaskBlackboard,
): Promise<ProbeSnapshot | null> {
  const selectedId =
    blackboard.localization?.hypothesisId ??
    blackboard.evaluator?.selectedHypothesisId ??
    blackboard.hunter?.hypotheses[0]?.id;
  const hyp = blackboard.hunter?.hypotheses.find((h) => h.id === selectedId);
  const probeFilePath = hyp?.probeFilePath;
  if (!probeFilePath) {
    return null;
  }

  const absProbe = path.resolve(repoDir, probeFilePath);
  if (hyp?.probeContent) {
    return { probeFilePath: absProbe, content: hyp.probeContent };
  }
  try {
    const content = await fs.readFile(absProbe, 'utf8');
    return { probeFilePath: absProbe, content };
  } catch {
    return null;
  }
}

const OPENAI_TOOLS: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read a file or line range inside the repository.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          startLine: { type: 'number' },
          endLine: { type: 'number' },
        },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write or create a file inside the repository (e.g. regression test).',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          content: { type: 'string' },
        },
        required: ['path', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'edit_file',
      description: 'Replace an exact target snippet inside a repository file.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          target: { type: 'string' },
          replacement: { type: 'string' },
        },
        required: ['path', 'target', 'replacement'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_repo',
      description: 'Search the repository with git grep.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          pathFilter: { type: 'string' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_command',
      description: 'Run a verification or test command in the isolated repo environment.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string' },
          timeoutMs: { type: 'number' },
        },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'finish_fix',
      description: 'Signal that the surgical fix and regression test are complete.',
      parameters: {
        type: 'object',
        properties: {
          strategy: { type: 'string' },
          rationale: { type: 'string' },
          summary: { type: 'string' },
        },
        required: ['strategy', 'rationale', 'summary'],
      },
    },
  },
];

export class ToolLoopAgent {
  private readonly repoDir: string;
  private readonly probeAbsPath: string | null;
  private readonly resolvedEnv: Record<string, string>;
  private finishedPayload: {
    strategy: string;
    rationale: string;
    summary: string;
  } | null = null;
  readonly touchedFiles = new Map<string, string>();

  constructor(
    repoDir: string,
    resolvedEnv: Record<string, string>,
    probeAbsPath: string | null,
  ) {
    this.repoDir = path.resolve(repoDir);
    this.resolvedEnv = resolvedEnv;
    this.probeAbsPath = probeAbsPath;
  }

  get finished(): boolean {
    return this.finishedPayload !== null;
  }

  get finishInfo() {
    return this.finishedPayload;
  }

  async executeTool(name: CodeFixToolName, rawArgs: unknown): Promise<string> {
    switch (name) {
      case 'read_file': {
        const args = ReadFileArgsSchema.parse(rawArgs);
        const abs = resolveInsideRepo(this.repoDir, args.path);
        const raw = await fs.readFile(abs, 'utf8');
        const lines = raw.split(/\r?\n/);
        const start = Math.max(1, args.startLine ?? 1);
        const end = Math.min(lines.length, args.endLine ?? lines.length);
        return lines
          .slice(start - 1, end)
          .map((line, i) => `${start + i}: ${line}`)
          .join('\n');
      }
      case 'write_file': {
        const args = WriteFileArgsSchema.parse(rawArgs);
        const abs = resolveInsideRepo(this.repoDir, args.path);
        if (this.probeAbsPath && abs === this.probeAbsPath) {
          throw new Error('Modifying the reproduction probe file is forbidden by preserveProbe()');
        }
        await fs.mkdir(path.dirname(abs), { recursive: true });
        await fs.writeFile(abs, args.content, 'utf8');
        this.touchedFiles.set(args.path, 'Wrote file');
        return `Wrote ${args.path}`;
      }
      case 'edit_file': {
        const args = EditFileArgsSchema.parse(rawArgs);
        const abs = resolveInsideRepo(this.repoDir, args.path);
        if (this.probeAbsPath && abs === this.probeAbsPath) {
          throw new Error('Modifying the reproduction probe file is forbidden by preserveProbe()');
        }
        const current = await fs.readFile(abs, 'utf8');
        if (!current.includes(args.target)) {
          throw new Error(`Target snippet not found in ${args.path}`);
        }
        const updated = current.replace(args.target, args.replacement);
        await fs.writeFile(abs, updated, 'utf8');
        this.touchedFiles.set(args.path, 'Edited target block');
        return `Updated ${args.path}`;
      }
      case 'search_repo': {
        const args = SearchRepoArgsSchema.parse(rawArgs);
        const grepArgs = ['grep', '-n', '-F', args.query];
        if (args.pathFilter) {
          grepArgs.push('--', args.pathFilter);
        }
        try {
          const { stdout } = await execFileAsync('git', grepArgs, {
            cwd: this.repoDir,
            maxBuffer: 2 * 1024 * 1024,
          });
          return stdout.trim() || '(no matches)';
        } catch {
          return '(no matches)';
        }
      }
      case 'run_command': {
        const args = RunCommandArgsSchema.parse(rawArgs);
        try {
          const { stdout, stderr } = await execAsync(args.command, {
            cwd: this.repoDir,
            env: this.resolvedEnv,
            timeout: args.timeoutMs ?? 60_000,
            maxBuffer: 5 * 1024 * 1024,
          });
          return `exit=0\n${stdout}\n${stderr}`.trim();
        } catch (err) {
          const e = err as { code?: number; stdout?: string; stderr?: string; message?: string };
          return `exit=${e.code ?? 1}\n${e.stdout ?? ''}\n${e.stderr ?? e.message ?? ''}`.trim();
        }
      }
      case 'finish_fix': {
        const args = FinishFixArgsSchema.parse(rawArgs);
        this.finishedPayload = args;
        return 'Fix marked complete.';
      }
    }
  }

  async runWithMuse(
    blackboard: TaskBlackboard,
    initialPrompt: string,
    options: {
      apiKey: string;
      maxToolTurns: number;
      stateManager?: StateManager;
      issueId?: string;
    },
  ): Promise<{ strategy: string; rationale: string }> {
    const client = new OpenAI({
      baseURL: blackboard.model.baseURL || DEFAULT_META_BASE_URL,
      apiKey: options.apiKey,
    });

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      {
        role: 'system',
        content:
          'You are Spot CodeFixAgent. Apply a minimal, surgical fix to the localized root cause, add/update a regression test, verify with run_command, and call finish_fix. Never alter the reproduction probe.',
      },
      { role: 'user', content: initialPrompt },
    ];

    for (let turn = 0; turn < options.maxToolTurns; turn += 1) {
      const start = Date.now();
      const response = await client.chat.completions.create({
        model: blackboard.model.model || DEFAULT_META_MODEL,
        temperature: 0.1,
        messages,
        tools: OPENAI_TOOLS,
      });

      if (options.stateManager && options.issueId && options.stateManager.has(options.issueId)) {
        options.stateManager.recordMetrics(options.issueId, {
          llmCalls: 1,
          inputTokens: response.usage?.prompt_tokens ?? 0,
          outputTokens: response.usage?.completion_tokens ?? 0,
          durationMs: Date.now() - start,
        });
      }

      const msg = response.choices[0]?.message;
      if (!msg) break;
      messages.push(msg);

      const toolCalls = msg.tool_calls ?? [];
      if (toolCalls.length === 0) {
        break;
      }

      for (const tc of toolCalls) {
        if (tc.type !== 'function') continue;
        const toolName = tc.function.name as CodeFixToolName;
        let resultText: string;
        try {
          const parsedArgs = JSON.parse(tc.function.arguments || '{}');
          resultText = await this.executeTool(toolName, parsedArgs);
        } catch (err) {
          resultText = `Tool error: ${(err as Error).message}`;
        }

        messages.push({
          role: 'tool',
          tool_call_id: tc.id,
          content: resultText.slice(0, 8000),
        });
      }

      if (this.finished) {
        break;
      }
    }

    return (
      this.finishedPayload ?? {
        strategy: 'Surgical fix via ToolLoopAgent',
        rationale: 'Completed tool loop edits.',
      }
    );
  }
}

async function defaultAgyRunner(
  prompt: string,
  context: ToolLoopExecutionContext,
): Promise<{ strategy: string; rationale: string }> {
  const { stdout } = await execFileAsync('agy', ['run', '--prompt', prompt], {
    cwd: context.repoDir,
    maxBuffer: 10 * 1024 * 1024,
  });
  return {
    strategy: 'AGY runner fix',
    rationale: stdout.trim().slice(0, 1000) || 'Executed via USE_AGY runner',
  };
}

async function computeGitDiff(repoDir: string): Promise<{
  diff: string;
  changedFiles: string[];
  baseCommitSha?: string;
}> {
  let diff = '';
  let changedFiles: string[] = [];
  let baseCommitSha: string | undefined;

  try {
    const rev = await execFileAsync('git', ['rev-parse', 'HEAD'], { cwd: repoDir });
    baseCommitSha = rev.stdout.trim();
  } catch {
    baseCommitSha = undefined;
  }

  try {
    const { stdout } = await execFileAsync('git', ['diff'], {
      cwd: repoDir,
      maxBuffer: 10 * 1024 * 1024,
    });
    diff = stdout;
    const names = await execFileAsync('git', ['diff', '--name-only'], {
      cwd: repoDir,
    });
    changedFiles = names.stdout
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
  } catch {
    // Non-git fallback.
  }

  return { diff, changedFiles, baseCommitSha };
}

export async function runCodeFixAgent(
  blackboard: TaskBlackboard,
  options: RunCodeFixOptions = {},
): Promise<RunCodeFixResult> {
  const repoDir = path.resolve(
    blackboard.repo.worktreePath ?? blackboard.repo.localPath,
  );
  const localization = blackboard.localization;
  const hypothesisId =
    localization?.hypothesisId ??
    blackboard.evaluator?.selectedHypothesisId ??
    blackboard.hunter?.hypotheses[0]?.id ??
    `hyp-${randomUUID().slice(0, 6)}`;
  const hypothesis = blackboard.hunter?.hypotheses.find(
    (h) => h.id === hypothesisId,
  );

  if (options.stateManager && options.issueId && options.stateManager.has(options.issueId)) {
    options.stateManager.transition(options.issueId, 'fixing', {
      reason: `Applying surgical fix for ${hypothesis?.title ?? hypothesisId}`,
    });
  }

  const resolvedEnv = await resolveRepoEnv(repoDir);
  const probeSnapshot = await captureProbeSnapshot(repoDir, blackboard);
  const loopAgent = new ToolLoopAgent(
    repoDir,
    resolvedEnv.env,
    probeSnapshot?.probeFilePath ?? null,
  );

  const initialPrompt = [
    `Fix verified bug: ${hypothesis?.title ?? hypothesisId}`,
    `Root Cause: ${localization?.rootCause ?? hypothesis?.description ?? ''}`,
    `Primary File: ${localization?.primaryFile ?? hypothesis?.affectedFiles[0] ?? ''}`,
    `Symbols: ${localization?.symbols.join(', ') ?? hypothesis?.targetSymbol ?? ''}`,
    hypothesis?.probeCommand
      ? `Reproduction Command (must pass after fix): ${hypothesis.probeCommand}`
      : '',
    'Use read_file, search_repo, edit_file, write_file, run_command, and finish_fix.',
  ]
    .filter(Boolean)
    .join('\n');

  const execCtx: ToolLoopExecutionContext = {
    repoDir,
    blackboard,
    executeTool: (name, args) => loopAgent.executeTool(name, args),
  };

  let outcome: { strategy: string; rationale: string };
  const useAgy = options.useAgy ?? process.env.USE_AGY === '1';

  if (useAgy) {
    const runner = options.agyRunner ?? defaultAgyRunner;
    outcome = await runner(initialPrompt, execCtx);
  } else if (options.toolLoopRunner) {
    outcome = await options.toolLoopRunner(execCtx, initialPrompt);
  } else {
    const apiKey =
      options.apiKey ??
      process.env[blackboard.model.apiKeyEnv] ??
      process.env.META_API_KEY ??
      process.env.MODEL_API_KEY;
    if (!apiKey || apiKey.trim().length === 0) {
      throw new Error(
        `Missing Meta API key (${blackboard.model.apiKeyEnv}) for runCodeFixAgent`,
      );
    }
    outcome = await loopAgent.runWithMuse(blackboard, initialPrompt, {
      apiKey,
      maxToolTurns: options.maxToolTurns ?? 12,
      stateManager: options.stateManager,
      issueId: options.issueId,
    });
  }

  // Enforce preserveProbe() guardrail after code-fix loop finishes.
  const guard = await preserveProbe(repoDir, probeSnapshot);
  const { diff, changedFiles, baseCommitSha } = await computeGitDiff(repoDir);

  const mergedFiles = [...new Set([...changedFiles, ...loopAgent.touchedFiles.keys()])];
  const edits: FileEditSummary[] = mergedFiles.map((filePath) => ({
    filePath,
    summary: loopAgent.touchedFiles.get(filePath) ?? 'Modified during fix loop',
    hunkCount: (diff.match(new RegExp(`\\+\\+\\+ b/${filePath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'g')) ?? []).length || 1,
  }));

  const patchId = `patch-${randomUUID().slice(0, 8)}`;
  const branchName =
    options.branchName ?? `spot/fix-${hypothesisId.replace(/[^a-zA-Z0-9_-]/g, '-')}`;

  const fix: FixContext = {
    hypothesisId,
    patchId,
    branchName,
    baseCommitSha,
    strategy: outcome.strategy,
    rationale: outcome.rationale,
    diff,
    edits,
    iteration: 1,
    maxIterations: options.maxIterations ?? 3,
    completedAt: new Date().toISOString(),
  };

  if (options.memory) {
    await options.memory.saveProposal({
      id: patchId,
      repo: blackboard.repo.fullName,
      title: hypothesis?.title ?? `Fix ${hypothesisId}`,
      severity: hypothesis?.severity ?? 'medium',
      status: 'draft',
      hypothesisSummary: localization?.rootCause ?? hypothesis?.description ?? '',
      diff,
      changedFiles: mergedFiles,
    });
  }

  const nextBlackboard = setBlackboardSection(blackboard, 'fix', fix);

  return {
    blackboard: nextBlackboard,
    fix,
    probePreserved: guard.preserved,
  };
}

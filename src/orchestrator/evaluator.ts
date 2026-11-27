import { exec, execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import OpenAI from 'openai';
import { z } from 'zod';
import {
  DEFAULT_META_BASE_URL,
  DEFAULT_META_MODEL,
  setBlackboardSection,
  type BugHypothesis,
  type EvaluationVerdict,
  type EvaluatorContext,
  type TaskBlackboard,
} from './blackboard.js';
import type { StateManager } from './state.js';
import { resolveRepoEnv } from '../utils/env.js';
import { isHuntSource } from '../utils/hunt-scope.js';
import type { BehavioralMemory } from '../utils/memory.js';
import {
  assessProbeFailure,
  assessReachability,
  buildProbeCommand,
  commandTimeoutMs,
  detectProject,
  resolveProbeFile,
} from '../utils/toolchain.js';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

export const MIN_EVALUATOR_CONFIDENCE = 0.7;

const ArchitectReviewSchema = z.object({
  accepted: z.boolean(),
  confidence: z.number().min(0).max(1),
  proofType: z
    .enum(['repro_test', 'runtime_trace', 'static_proof', 'unproven'])
    .default('repro_test'),
  rationale: z.string().default(''),
  reproductionSteps: z.array(z.string()).default([]),
  rejectionReason: z.string().optional(),
});

export interface GitGrepCallerMatch {
  file: string;
  line: number;
  snippet: string;
}

export type EvaluatorCompletionFn = (
  prompt: string,
  meta: {
    hypothesisId: string;
    role: 'principal_architect';
  },
) => Promise<string>;

export interface BugEvaluatorAgentOptions {
  memory?: BehavioralMemory;
  stateManager?: StateManager;
  issueId?: string;
  completionFn?: EvaluatorCompletionFn;
  apiKey?: string;
  minConfidence?: number;
  cleanupRejectedProbes?: boolean;
}

export interface BugEvaluatorRunResult {
  blackboard: TaskBlackboard;
  evaluator: EvaluatorContext;
  acceptedCount: number;
  rejectedCount: number;
}

function extractJsonObject(raw: string): string {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return fenced[1].trim();
  }
  const first = raw.indexOf('{');
  const last = raw.lastIndexOf('}');
  if (first !== -1 && last > first) {
    return raw.slice(first, last + 1);
  }
  return raw.trim();
}

async function runCommandCapture(
  command: string,
  cwd: string,
  env: Record<string, string>,
  timeoutMs: number,
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  try {
    const { stdout, stderr } = await execAsync(command, {
      cwd,
      env,
      timeout: timeoutMs,
      maxBuffer: 10 * 1024 * 1024,
    });
    return { exitCode: 0, stdout, stderr };
  } catch (error) {
    const err = error as {
      code?: number | string;
      stdout?: string;
      stderr?: string;
      message?: string;
    };
    return {
      exitCode: typeof err.code === 'number' ? err.code : 1,
      stdout: err.stdout ?? '',
      stderr: err.stderr ?? err.message ?? '',
    };
  }
}

async function gitGrepUpstreamCallers(
  repoDir: string,
  symbol: string,
  targetFile: string,
): Promise<GitGrepCallerMatch[]> {
  if (!symbol.trim()) {
    return [];
  }

  try {
    const { stdout } = await execFileAsync(
      'git',
      ['grep', '-n', '-F', symbol],
      { cwd: repoDir, maxBuffer: 5 * 1024 * 1024 },
    );

    const normalizedTarget = targetFile.replace(/\\/g, '/').replace(/^\.\//, '');
    const matches: GitGrepCallerMatch[] = [];

    for (const line of stdout.split(/\r?\n/)) {
      if (!line.trim()) continue;
      const firstColon = line.indexOf(':');
      const secondColon = line.indexOf(':', firstColon + 1);
      if (firstColon === -1 || secondColon === -1) continue;

      const file = line.slice(0, firstColon).replace(/\\/g, '/');
      const lineNum = Number.parseInt(line.slice(firstColon + 1, secondColon), 10);
      const snippet = line.slice(secondColon + 1).trim();

      if (!isHuntSource(file)) continue;
      if (file === normalizedTarget && /\b(?:function|def|func|fn|class)\s+/.test(snippet)) {
        continue;
      }
      matches.push({
        file,
        line: Number.isNaN(lineNum) ? 1 : lineNum,
        snippet,
      });
    }

    return matches;
  } catch {
    return [];
  }
}

export class BugEvaluatorAgent {
  private readonly memory?: BehavioralMemory;
  private readonly stateManager?: StateManager;
  private readonly issueId?: string;
  private readonly completionFn?: EvaluatorCompletionFn;
  private readonly apiKey?: string;
  private readonly minConfidence: number;
  private readonly cleanupRejectedProbes: boolean;

  constructor(options: BugEvaluatorAgentOptions = {}) {
    this.memory = options.memory;
    this.stateManager = options.stateManager;
    this.issueId = options.issueId;
    this.completionFn = options.completionFn;
    this.apiKey = options.apiKey;
    this.minConfidence = options.minConfidence ?? MIN_EVALUATOR_CONFIDENCE;
    this.cleanupRejectedProbes = options.cleanupRejectedProbes ?? true;
  }

  private async callPrincipalArchitect(
    blackboard: TaskBlackboard,
    prompt: string,
    hypothesisId: string,
  ): Promise<z.infer<typeof ArchitectReviewSchema>> {
    let raw: string;

    if (this.completionFn) {
      raw = await this.completionFn(prompt, {
        hypothesisId,
        role: 'principal_architect',
      });
    } else {
      const apiKey =
        this.apiKey ??
        process.env[blackboard.model.apiKeyEnv] ??
        process.env.META_API_KEY ??
        process.env.MODEL_API_KEY;
      if (!apiKey || apiKey.trim().length === 0) {
        throw new Error(
          `Missing Meta API key (${blackboard.model.apiKeyEnv}) for BugEvaluatorAgent`,
        );
      }

      const client = new OpenAI({
        baseURL: blackboard.model.baseURL || DEFAULT_META_BASE_URL,
        apiKey,
      });

      const start = Date.now();
      const response = await client.chat.completions.create({
        model: blackboard.model.model || DEFAULT_META_MODEL,
        temperature: 0.1,
        messages: [
          {
            role: 'system',
            content:
              'You are a skeptical Principal Architect reviewing bug claims. Reject hallucinations, mock stand-ins, unreachable code, and unguarded guesses. Return strict JSON only.',
          },
          { role: 'user', content: prompt },
        ],
      });

      if (this.stateManager && this.issueId && this.stateManager.has(this.issueId)) {
        this.stateManager.recordMetrics(this.issueId, {
          llmCalls: 1,
          inputTokens: response.usage?.prompt_tokens ?? 0,
          outputTokens: response.usage?.completion_tokens ?? 0,
          durationMs: Date.now() - start,
        });
      }

      raw = response.choices[0]?.message?.content ?? '{}';
    }

    return ArchitectReviewSchema.parse(JSON.parse(extractJsonObject(raw)));
  }

  private async evaluateHypothesis(
    blackboard: TaskBlackboard,
    repoDir: string,
    hypothesis: BugHypothesis,
    filesToCleanup: Set<string>,
  ): Promise<EvaluationVerdict> {
    const primaryFile = hypothesis.affectedFiles[0] ?? '';
    const project = await detectProject(repoDir);
    const resolvedEnv = await resolveRepoEnv(repoDir);

    let probeFilePath = hypothesis.probeFilePath;
    let probeCommand = hypothesis.probeCommand;
    let probeContent = hypothesis.probeContent;
    let createdTempProbe = false;

    if (!probeFilePath && probeContent) {
      probeFilePath = await resolveProbeFile(repoDir, project, {
        probeName: `spot_eval_${hypothesis.id}`,
        targetFile: primaryFile,
      });
      await fs.mkdir(path.dirname(probeFilePath), { recursive: true });
      await fs.writeFile(probeFilePath, probeContent, 'utf8');
      createdTempProbe = true;
    } else if (probeFilePath && !probeContent) {
      try {
        probeContent = await fs.readFile(probeFilePath, 'utf8');
      } catch {
        probeContent = undefined;
      }
    }

    if (!probeCommand && probeFilePath) {
      const spec = await buildProbeCommand(repoDir, project, probeFilePath);
      probeCommand = spec.command;
    }

    if (!probeCommand) {
      return {
        hypothesisId: hypothesis.id,
        accepted: false,
        proofType: 'unproven',
        confidence: 0,
        reproductionSteps: [hypothesis.reproHypothesis],
        rejectionReason: 'Missing probeCommand for independent dynamic re-execution.',
      };
    }

    // 1. Independent dynamic re-execution of probeCommand
    const timeoutMs = commandTimeoutMs('probe', project.primaryLanguage);
    const execRes = await runCommandCapture(
      probeCommand,
      repoDir,
      resolvedEnv.env,
      timeoutMs,
    );
    const combinedOutput = `${execRes.stdout}\n${execRes.stderr}`.trim();
    const probeAssessment = assessProbeFailure({
      language: project.primaryLanguage,
      exitCode: execRes.exitCode,
      stdout: execRes.stdout,
      stderr: execRes.stderr,
      probeFilePath,
      probeContent,
      targetFile: primaryFile,
      targetSymbol: hypothesis.targetSymbol,
    });

    if (!probeAssessment.valid) {
      if ((createdTempProbe || this.cleanupRejectedProbes) && probeFilePath) {
        filesToCleanup.add(probeFilePath);
      }
      return {
        hypothesisId: hypothesis.id,
        accepted: false,
        proofType: 'unproven',
        confidence: 0.1,
        reproductionSteps: [hypothesis.reproHypothesis],
        reproCommand: probeCommand,
        reproOutput: combinedOutput,
        rejectionReason: `Dynamic re-execution rejected (${probeAssessment.category}): ${probeAssessment.reason}`,
      };
    }

    // 2. git grep upstream caller + static reachability analysis
    const targetSymbol = hypothesis.targetSymbol ?? '';
    const [grepCallers, reachability] = await Promise.all([
      targetSymbol
        ? gitGrepUpstreamCallers(repoDir, targetSymbol, primaryFile)
        : Promise.resolve([]),
      targetSymbol && primaryFile
        ? assessReachability(repoDir, primaryFile, targetSymbol)
        : Promise.resolve(null),
    ]);

    if (reachability && !reachability.reachable && grepCallers.length === 0) {
      if ((createdTempProbe || this.cleanupRejectedProbes) && probeFilePath) {
        filesToCleanup.add(probeFilePath);
      }
      return {
        hypothesisId: hypothesis.id,
        accepted: false,
        proofType: 'unproven',
        confidence: 0.2,
        reproductionSteps: [hypothesis.reproHypothesis],
        reproCommand: probeCommand,
        reproOutput: combinedOutput,
        rejectionReason: `Static reachability & git grep check failed: ${reachability.reason}`,
      };
    }

    // 3. Skeptical Principal Architect LLM review (requires confidence >= 0.7)
    let targetSourceSnippet = '';
    if (primaryFile) {
      try {
        const rawSource = await fs.readFile(path.join(repoDir, primaryFile), 'utf8');
        targetSourceSnippet = rawSource.split(/\r?\n/).slice(0, 160).join('\n');
      } catch {
        targetSourceSnippet = '';
      }
    }

    const reviewPrompt = [
      'Skeptical Principal Architect Bug Adjudication',
      `Hypothesis ID: ${hypothesis.id}`,
      `Title: ${hypothesis.title}`,
      `Category: ${hypothesis.category} | Claimed Severity: ${hypothesis.severity}`,
      `Description: ${hypothesis.description}`,
      `Target File: ${primaryFile} | Symbol: ${targetSymbol || 'N/A'}`,
      '',
      `Dynamic Re-Execution Command: ${probeCommand}`,
      `Exit Code: ${execRes.exitCode}`,
      `Dynamic Output:\n${combinedOutput.slice(0, 2000)}`,
      '',
      probeContent ? `Probe Source:\n${probeContent.slice(0, 2000)}` : '',
      targetSourceSnippet ? `Target Source:\n${targetSourceSnippet}` : '',
      '',
      `git grep Upstream Callers (${grepCallers.length}):`,
      grepCallers.length > 0
        ? grepCallers
            .slice(0, 12)
            .map((c) => `- ${c.file}:${c.line}: ${c.snippet}`)
            .join('\n')
        : '- (none found via git grep; relies on exported API surface)',
      reachability ? `Reachability Assessment: ${reachability.reason}` : '',
      '',
      `Evaluate strictly: Is this a real, reachable, non-mocked bug? Reject if speculative, unreachable, already guarded by callers, or environmental. Acceptance requires confidence >= ${this.minConfidence}.`,
      'Return strict JSON: {"accepted":true,"confidence":0.85,"proofType":"repro_test","rationale":"...","reproductionSteps":["..."],"rejectionReason":"..."}',
    ]
      .filter(Boolean)
      .join('\n');

    let review: z.infer<typeof ArchitectReviewSchema>;
    try {
      review = await this.callPrincipalArchitect(
        blackboard,
        reviewPrompt,
        hypothesis.id,
      );
    } catch (err) {
      return {
        hypothesisId: hypothesis.id,
        accepted: false,
        proofType: 'unproven',
        confidence: 0,
        reproductionSteps: [hypothesis.reproHypothesis],
        reproCommand: probeCommand,
        reproOutput: combinedOutput,
        rejectionReason: `Principal Architect review failed to parse: ${(err as Error).message}`,
      };
    }

    const meetsThreshold =
      review.accepted && review.confidence >= this.minConfidence;

    if (!meetsThreshold) {
      if ((createdTempProbe || this.cleanupRejectedProbes) && probeFilePath) {
        filesToCleanup.add(probeFilePath);
      }
      const rejectionReason =
        review.rejectionReason ??
        (!review.accepted
          ? review.rationale || 'Rejected by skeptical Principal Architect review.'
          : `Confidence ${review.confidence.toFixed(2)} is below threshold ${this.minConfidence.toFixed(2)}.`);

      return {
        hypothesisId: hypothesis.id,
        accepted: false,
        proofType: review.proofType,
        confidence: review.confidence,
        reproductionSteps:
          review.reproductionSteps.length > 0
            ? review.reproductionSteps
            : [hypothesis.reproHypothesis],
        reproCommand: probeCommand,
        reproOutput: combinedOutput,
        rejectionReason,
      };
    }

    return {
      hypothesisId: hypothesis.id,
      accepted: true,
      proofType: review.proofType === 'unproven' ? 'repro_test' : review.proofType,
      confidence: review.confidence,
      reproductionSteps:
        review.reproductionSteps.length > 0
          ? review.reproductionSteps
          : [hypothesis.reproHypothesis],
      reproCommand: probeCommand,
      reproOutput: combinedOutput,
      failureSignature: combinedOutput.split(/\r?\n/)[0]?.slice(0, 200),
    };
  }

  async evaluate(blackboard: TaskBlackboard): Promise<BugEvaluatorRunResult> {
    const repoDir = path.resolve(
      blackboard.repo.worktreePath ?? blackboard.repo.localPath,
    );

    if (this.stateManager && this.issueId && this.stateManager.has(this.issueId)) {
      this.stateManager.transition(this.issueId, 'evaluating', {
        reason: 'Running independent probe re-execution, git grep caller check, and Principal Architect review',
      });
    }

    const hypotheses = blackboard.hunter?.hypotheses ?? [];
    const verdicts: EvaluationVerdict[] = [];
    const filesToCleanup = new Set<string>();
    const acceptedProbeFiles = new Set<string>();

    for (const hypothesis of hypotheses) {
      const verdict = await this.evaluateHypothesis(
        blackboard,
        repoDir,
        hypothesis,
        filesToCleanup,
      );
      verdicts.push(verdict);
      if (verdict.accepted && hypothesis.probeFilePath) {
        acceptedProbeFiles.add(hypothesis.probeFilePath);
      }

      if (this.memory) {
        const primaryFile = hypothesis.affectedFiles[0] ?? '';
        await this.memory.recordProbe(blackboard.repo.fullName, {
          hypothesisId: hypothesis.id,
          title: hypothesis.title,
          signature: `${primaryFile}::${hypothesis.targetSymbol ?? ''}::${hypothesis.category}`,
          verdict: verdict.accepted ? 'accepted' : 'rejected',
          proofType: verdict.proofType,
          reason:
            verdict.rejectionReason ??
            'Accepted after independent re-execution, reachability check, and Principal Architect review',
          files: hypothesis.affectedFiles,
        });
      }
    }

    for (const filePath of filesToCleanup) {
      if (!acceptedProbeFiles.has(filePath)) {
        await fs.rm(filePath, { force: true });
      }
    }

    const acceptedVerdicts = verdicts
      .filter((v) => v.accepted)
      .sort((a, b) => b.confidence - a.confidence);
    const selectedHypothesisId = acceptedVerdicts[0]?.hypothesisId;

    const evaluator: EvaluatorContext = {
      selectedHypothesisId,
      verdicts,
      completedAt: new Date().toISOString(),
    };

    const nextBlackboard = setBlackboardSection(
      blackboard,
      'evaluator',
      evaluator,
    );

    return {
      blackboard: nextBlackboard,
      evaluator,
      acceptedCount: acceptedVerdicts.length,
      rejectedCount: verdicts.length - acceptedVerdicts.length,
    };
  }
}

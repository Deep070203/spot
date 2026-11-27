import { exec, execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
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
  type GroundedFile,
  type GroundingContext,
  type HunterContext,
  type TaskBlackboard,
} from './blackboard.js';
import type { StateManager } from './state.js';
import {
  mineFilePatterns,
  type FileMiningResult,
} from '../utils/ast-miner.js';
import { normalizeCommandWithVenv, resolveRepoEnv } from '../utils/env.js';
import { selectHuntScope } from '../utils/hunt-scope.js';
import type { BehavioralMemory } from '../utils/memory.js';
import {
  assessProbeFailure,
  auditNote,
  buildProbeCommand,
  commandTimeoutMs,
  detectProject,
  huntExtensions,
  prepareToolchainWorkspace,
  probeInstructions,
  resolveProbeFile,
  type DetectedProject,
  type ProbeFailureAssessment,
  type SourceHeuristics,
} from '../utils/toolchain.js';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

export const BUG_ARCHETYPES = [
  // Round 1: State, Boundaries & Coercion
  'ZERO_STATE_CLAMP',
  'NULL_UNDEFINED_DEREF',
  'NUMERIC_PRECISION_OVERFLOW',
  'IMMUTABLE_DEFAULT_SHARING',
  // Round 2: Data Flow, Bucketing & Serialization
  'DEFAULT_BUCKET_PARTITION',
  'PARSER_ENCODING_ROUNDTRIP',
  'HOT_LOOP_COMPLEXITY',
  // Round 3: Concurrency, Async & Resource Liveness
  'CONCURRENCY_LIVENESS',
  'TOCTOU_RACE_CONDITION',
  'RESOURCE_HANDLE_LEAK',
  // Round 4: I/O Boundaries, Security & Error Contracts
  'FRAMEWORK_IO_ABSTRACTION',
  'SILENT_ERROR_SWALLOWING',
  'API_CONTRACT_MISMATCH',
] as const;

export type BugArchetype = (typeof BUG_ARCHETYPES)[number];

export interface AuditRoundSpec {
  round: number;
  name: string;
  archetypes: readonly BugArchetype[];
}

export const AUDIT_ROUNDS: readonly AuditRoundSpec[] = [
  {
    round: 1,
    name: 'State, Boundaries & Coercion',
    archetypes: [
      'ZERO_STATE_CLAMP',
      'NULL_UNDEFINED_DEREF',
      'NUMERIC_PRECISION_OVERFLOW',
      'IMMUTABLE_DEFAULT_SHARING',
    ],
  },
  {
    round: 2,
    name: 'Data Flow, Bucketing & Serialization',
    archetypes: [
      'DEFAULT_BUCKET_PARTITION',
      'PARSER_ENCODING_ROUNDTRIP',
      'HOT_LOOP_COMPLEXITY',
    ],
  },
  {
    round: 3,
    name: 'Concurrency, Async & Resource Liveness',
    archetypes: [
      'CONCURRENCY_LIVENESS',
      'TOCTOU_RACE_CONDITION',
      'RESOURCE_HANDLE_LEAK',
    ],
  },
  {
    round: 4,
    name: 'I/O Boundaries, Security & Error Contracts',
    archetypes: [
      'FRAMEWORK_IO_ABSTRACTION',
      'SILENT_ERROR_SWALLOWING',
      'API_CONTRACT_MISMATCH',
    ],
  },
];

const HIGH_SIGNAL_KEYWORDS = [
  'TODO',
  'FIXME',
  'HACK',
  'XXX',
  'race',
  'clamp',
  'fallback',
  'default',
  'timeout',
  'lock',
  'mutex',
  'parse',
  'bucket',
  'partition',
  'exec',
  'shell',
  'async',
  'await',
  'retry',
];

const RawCandidateSchema = z.object({
  title: z.string().min(1),
  description: z.string().min(1),
  category: z.string().min(1),
  severity: z.enum(['critical', 'high', 'medium', 'low']).default('medium'),
  confidence: z.number().min(0).max(1).default(0.75),
  targetFile: z.string().min(1),
  targetSymbol: z.string().optional(),
  reproHypothesis: z.string().min(1),
  evidence: z.array(z.string()).default([]),
  probeCode: z.string().min(1),
});

const RoundResponseSchema = z.object({
  candidates: z.array(RawCandidateSchema).default([]),
});

const ProbeRetryResponseSchema = z.object({
  probeCode: z.string().min(1),
  notes: z.string().optional(),
});

export interface RankedHuntCandidate {
  filePath: string;
  churnScore: number;
  astScore: number;
  keywordScore: number;
  totalScore: number;
  mining: FileMiningResult;
}

export type HunterCompletionFn = (
  prompt: string,
  meta: {
    phase: 'audit_round' | 'probe_retry';
    round?: number;
    archetypes?: readonly BugArchetype[];
    attempt?: number;
  },
) => Promise<string>;

export interface BugHunterAgentOptions {
  memory?: BehavioralMemory;
  stateManager?: StateManager;
  issueId?: string;
  completionFn?: HunterCompletionFn;
  apiKey?: string;
  maxFiles?: number;
  maxCandidatesPerRound?: number;
  runBaseline?: boolean;
  keepVerifiedProbes?: boolean;
}

export interface BugHunterRunResult {
  blackboard: TaskBlackboard;
  grounding: GroundingContext;
  hunter: HunterContext;
  rankedFiles: RankedHuntCandidate[];
  roundsExecuted: number;
  probeRetriesUsed: number;
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

function computeKeywordScore(content: string): number {
  let score = 0;
  for (const kw of HIGH_SIGNAL_KEYWORDS) {
    const re = new RegExp(`\\b${kw}\\b`, 'gi');
    const matches = content.match(re);
    if (matches) {
      score += Math.min(matches.length, 3);
    }
  }
  return score;
}

function extractSymbolsSnippet(content: string): string[] {
  const symbols = new Set<string>();
  const patterns = [
    /(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/g,
    /(?:export\s+)?class\s+([A-Za-z_$][\w$]*)/g,
    /(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=/g,
    /def\s+([A-Za-z_]\w*)\s*\(/g,
    /func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(/g,
    /(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)/g,
  ];
  for (const re of patterns) {
    for (const m of content.matchAll(re)) {
      if (m[1]) symbols.add(m[1]);
    }
  }
  return [...symbols].slice(0, 20);
}

async function runShellCommand(
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
    const exitCode = typeof err.code === 'number' ? err.code : 1;
    return {
      exitCode,
      stdout: err.stdout ?? '',
      stderr: err.stderr ?? err.message ?? '',
    };
  }
}

async function collectGitFileChurn(repoDir: string): Promise<Map<string, number>> {
  const churn = new Map<string, number>();
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['log', '--name-only', '--format=', '-n', '100'],
      { cwd: repoDir, maxBuffer: 10 * 1024 * 1024 },
    );
    for (const line of stdout.split(/\r?\n/)) {
      const clean = line.trim().replace(/\\/g, '/');
      if (clean) {
        churn.set(clean, (churn.get(clean) ?? 0) + 1);
      }
    }
  } catch {
    // Repo may have no commit history yet.
  }
  return churn;
}

export class BugHunterAgent {
  private readonly memory?: BehavioralMemory;
  private readonly stateManager?: StateManager;
  private readonly issueId?: string;
  private readonly completionFn?: HunterCompletionFn;
  private readonly apiKey?: string;
  private readonly maxFiles: number;
  private readonly maxCandidatesPerRound: number;
  private readonly runBaseline: boolean;
  private readonly keepVerifiedProbes: boolean;

  constructor(options: BugHunterAgentOptions = {}) {
    this.memory = options.memory;
    this.stateManager = options.stateManager;
    this.issueId = options.issueId;
    this.completionFn = options.completionFn;
    this.apiKey = options.apiKey;
    this.maxFiles = options.maxFiles ?? 25;
    this.maxCandidatesPerRound = options.maxCandidatesPerRound ?? 3;
    this.runBaseline = options.runBaseline ?? true;
    this.keepVerifiedProbes = options.keepVerifiedProbes ?? true;
  }

  private async callLlm(
    blackboard: TaskBlackboard,
    prompt: string,
    meta: Parameters<HunterCompletionFn>[1],
  ): Promise<string> {
    if (this.completionFn) {
      return this.completionFn(prompt, meta);
    }

    const apiKey =
      this.apiKey ??
      process.env[blackboard.model.apiKeyEnv] ??
      process.env.META_API_KEY ??
      process.env.MODEL_API_KEY;
    if (!apiKey || apiKey.trim().length === 0) {
      throw new Error(
        `Missing Meta API key (${blackboard.model.apiKeyEnv}) for BugHunterAgent`,
      );
    }

    const client = new OpenAI({
      baseURL: blackboard.model.baseURL || DEFAULT_META_BASE_URL,
      apiKey,
    });

    const start = Date.now();
    const response = await client.chat.completions.create({
      model: blackboard.model.model || DEFAULT_META_MODEL,
      temperature: blackboard.model.temperature ?? 0.2,
      messages: [
        {
          role: 'system',
          content:
            'You are Spot BugHunter, a proof-first autonomous bug discovery agent. Return strict JSON only.',
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

    return response.choices[0]?.message?.content ?? '{"candidates":[]}';
  }

  private async runBaselineChecks(
    repoDir: string,
    project: DetectedProject,
    env: Record<string, string>,
  ): Promise<GroundingContext['baseline']> {
    if (!this.runBaseline) {
      return {
        buildCommand: project.typecheckCommand ?? project.buildCommand,
        testCommand: project.testCommand,
      };
    }

    const buildCmd = project.typecheckCommand ?? project.buildCommand;
    let buildPassed: boolean | undefined;
    let outputParts: string[] = [];

    if (buildCmd) {
      const normalizedBuild = normalizeCommandWithVenv(buildCmd, {
        venvBinDir: null,
        pythonBinary: null,
      });
      const res = await runShellCommand(
        normalizedBuild,
        repoDir,
        env,
        commandTimeoutMs('build', project.primaryLanguage),
      );
      buildPassed = res.exitCode === 0;
      outputParts.push(`[build] exit=${res.exitCode}\n${res.stdout}\n${res.stderr}`.trim());
    }

    let testsPassed: boolean | undefined;
    if (project.testCommand) {
      const resolvedEnv = await resolveRepoEnv(repoDir);
      const normalizedTest = normalizeCommandWithVenv(project.testCommand, resolvedEnv);
      const res = await runShellCommand(
        normalizedTest,
        repoDir,
        resolvedEnv.env,
        commandTimeoutMs('test', project.primaryLanguage),
      );
      testsPassed = res.exitCode === 0;
      outputParts.push(`[test] exit=${res.exitCode}\n${res.stdout}\n${res.stderr}`.trim());
    }

    return {
      buildCommand: buildCmd,
      buildPassed,
      testCommand: project.testCommand,
      testsPassed,
      outputSnippet: outputParts.join('\n\n').slice(0, 2000) || undefined,
    };
  }

  private async executeProbeWithOneRetry(params: {
    blackboard: TaskBlackboard;
    repoDir: string;
    project: DetectedProject;
    heuristics: SourceHeuristics;
    env: Record<string, string>;
    candidate: z.infer<typeof RawCandidateSchema>;
    round: number;
    archetypes: readonly BugArchetype[];
  }): Promise<{
    valid: boolean;
    probeFilePath: string;
    probeCommand: string;
    probeCode: string;
    probeOutput: string;
    assessment: ProbeFailureAssessment;
    retried: boolean;
  }> {
    const {
      blackboard,
      repoDir,
      project,
      heuristics,
      env,
      candidate,
      round,
      archetypes,
    } = params;

    const probeName = `spot_r${round}_${candidate.targetSymbol ?? path.basename(candidate.targetFile, path.extname(candidate.targetFile))}_${randomUUID().slice(0, 6)}`;
    const probeFilePath = await resolveProbeFile(repoDir, project, {
      probeName,
      targetFile: candidate.targetFile,
    });
    const probeSpec = await buildProbeCommand(repoDir, project, probeFilePath);

    let currentProbeCode = candidate.probeCode;
    let retried = false;

    for (let attempt = 1; attempt <= 2; attempt += 1) {
      await fs.mkdir(path.dirname(probeFilePath), { recursive: true });
      await fs.writeFile(probeFilePath, currentProbeCode, 'utf8');

      const execRes = await runShellCommand(
        probeSpec.command,
        probeSpec.cwd,
        env,
        probeSpec.timeoutMs,
      );
      const combinedOutput = `${execRes.stdout}\n${execRes.stderr}`.trim();
      const assessment = assessProbeFailure({
        language: probeSpec.language,
        exitCode: execRes.exitCode,
        stdout: execRes.stdout,
        stderr: execRes.stderr,
        probeFilePath,
        probeContent: currentProbeCode,
        targetFile: candidate.targetFile,
        targetSymbol: candidate.targetSymbol,
      });

      if (assessment.valid) {
        if (!this.keepVerifiedProbes) {
          await fs.rm(probeFilePath, { force: true });
        }
        return {
          valid: true,
          probeFilePath,
          probeCommand: probeSpec.command,
          probeCode: currentProbeCode,
          probeOutput: combinedOutput,
          assessment,
          retried,
        };
      }

      const retryable =
        attempt === 1 &&
        !assessment.environmental &&
        (assessment.category === 'probe_syntax_or_compile_error' ||
          assessment.category === 'passed_unexpectedly' ||
          assessment.category === 'mock_standin' ||
          assessment.category === 'insufficient_evidence');

      if (!retryable) {
        await fs.rm(probeFilePath, { force: true });
        return {
          valid: false,
          probeFilePath,
          probeCommand: probeSpec.command,
          probeCode: currentProbeCode,
          probeOutput: combinedOutput,
          assessment,
          retried,
        };
      }

      retried = true;
      const retryPrompt = [
        `Your reproduction probe for "${candidate.title}" (${candidate.targetFile}) failed validation.`,
        `Failure Category: ${assessment.category}`,
        `Reason: ${assessment.reason}`,
        `Command: ${probeSpec.command}`,
        `Exit Code: ${execRes.exitCode}`,
        `Output:\n${combinedOutput.slice(0, 2000)}`,
        '',
        probeInstructions(project, candidate.targetFile, heuristics),
        '',
        `Previous Probe Code:\n${currentProbeCode}`,
        '',
        'Fix the probe so it directly imports real repository code and fails (exits non-zero) ONLY when the bug triggers.',
        'Return strict JSON: {"probeCode":"...","notes":"..."}',
      ].join('\n');

      try {
        const retryRaw = await this.callLlm(blackboard, retryPrompt, {
          phase: 'probe_retry',
          round,
          archetypes,
          attempt: 2,
        });
        const parsedRetry = ProbeRetryResponseSchema.parse(
          JSON.parse(extractJsonObject(retryRaw)),
        );
        currentProbeCode = parsedRetry.probeCode;
      } catch {
        await fs.rm(probeFilePath, { force: true });
        return {
          valid: false,
          probeFilePath,
          probeCommand: probeSpec.command,
          probeCode: currentProbeCode,
          probeOutput: combinedOutput,
          assessment,
          retried,
        };
      }
    }

    await fs.rm(probeFilePath, { force: true });
    return {
      valid: false,
      probeFilePath,
      probeCommand: probeSpec.command,
      probeCode: currentProbeCode,
      probeOutput: '',
      assessment: {
        valid: false,
        category: 'insufficient_evidence',
        reason: 'Probe failed after retry.',
        environmental: false,
        mockStandin: false,
      },
      retried,
    };
  }

  async run(blackboard: TaskBlackboard): Promise<BugHunterRunResult> {
    const repoDir = path.resolve(blackboard.repo.worktreePath ?? blackboard.repo.localPath);

    if (this.stateManager && this.issueId && this.stateManager.has(this.issueId)) {
      this.stateManager.transition(this.issueId, 'grounding', {
        reason: 'Running baseline check, git churn grounding, and AST ranking',
      });
    }

    const [project, workspace, repoMemory, probeHistory] = await Promise.all([
      detectProject(repoDir),
      prepareToolchainWorkspace(repoDir),
      this.memory ? this.memory.loadRepoMemory(blackboard.repo.fullName) : Promise.resolve(null),
      this.memory ? this.memory.loadProbesHistory(blackboard.repo.fullName) : Promise.resolve([]),
    ]);

    const [baseline, scope, churnMap] = await Promise.all([
      this.runBaselineChecks(repoDir, project, workspace.env.env),
      selectHuntScope(repoDir, {
        extensions: huntExtensions(project),
        maxFiles: this.maxFiles,
      }),
      collectGitFileChurn(repoDir),
    ]);

    const rankedFiles: RankedHuntCandidate[] = [];
    const groundedFiles: GroundedFile[] = [];
    const fileSnippets = new Map<string, string>();

    for (const relFile of scope.selectedFiles) {
      const absFile = path.join(repoDir, relFile);
      let content = '';
      try {
        content = await fs.readFile(absFile, 'utf8');
      } catch {
        continue;
      }
      fileSnippets.set(relFile, content);

      const mining = await mineFilePatterns(absFile, content);
      const churnScore = churnMap.get(relFile) ?? 0;
      const keywordScore = computeKeywordScore(content);
      const totalScore = mining.score * 3 + churnScore * 2 + keywordScore;

      rankedFiles.push({
        filePath: relFile,
        churnScore,
        astScore: mining.score,
        keywordScore,
        totalScore,
        mining,
      });

      groundedFiles.push({
        path: relFile,
        role: 'hunt_candidate',
        symbols: extractSymbolsSnippet(content),
      });
    }

    rankedFiles.sort(
      (a, b) =>
        b.totalScore - a.totalScore ||
        b.astScore - a.astScore ||
        a.filePath.localeCompare(b.filePath),
    );

    const grounding: GroundingContext = {
      architectureSummary: `Detected ${project.primaryLanguage} (${project.packageManager}, runner=${project.testRunner}) across ${scope.scopeDirs.join(', ') || '.'}`,
      entrypoints: scope.selectedFiles.filter((f) =>
        /(?:index|main|app|server|cli)\.[^.]+$/.test(path.basename(f)),
      ),
      testFiles: [],
      relevantFiles: groundedFiles,
      dependencies: {},
      baseline,
      completedAt: new Date().toISOString(),
    };

    let nextBlackboard = setBlackboardSection(blackboard, 'grounding', grounding);

    if (this.stateManager && this.issueId && this.stateManager.has(this.issueId)) {
      this.stateManager.transition(this.issueId, 'hunting', {
        reason: 'Running 4-round batch audit across 13 bug archetypes',
      });
    }

    const rejectedSignatures = new Set(
      probeHistory
        .filter((p) => p.verdict === 'rejected')
        .map((p) => p.signature),
    );

    const topFilesForAudit = rankedFiles.slice(0, 8);
    const contextDigest = topFilesForAudit
      .map((candidate) => {
        const raw = fileSnippets.get(candidate.filePath) ?? '';
        const numbered = raw
          .split(/\r?\n/)
          .slice(0, 120)
          .map((line, idx) => `${idx + 1}: ${line}`)
          .join('\n');
        const astSummary = candidate.mining.findings
          .slice(0, 5)
          .map((f) => `  - L${f.line} [${f.kind}] ${f.message}`)
          .join('\n');
        return `### ${candidate.filePath} (score=${candidate.totalScore}, churn=${candidate.churnScore}, ast=${candidate.astScore}, kw=${candidate.keywordScore})\n${astSummary ? `AST Findings:\n${astSummary}\n` : ''}\`\`\`\n${numbered}\n\`\`\``;
      })
      .join('\n\n');

    const hypotheses: BugHypothesis[] = [];
    const notes: string[] = [];
    let probeRetriesUsed = 0;

    for (const roundSpec of AUDIT_ROUNDS) {
      const prompt = [
        `Spot BugHunter Round ${roundSpec.round}/4: ${roundSpec.name}`,
        `Focus Archetypes: ${roundSpec.archetypes.join(', ')}`,
        '',
        auditNote(project, workspace.heuristics),
        '',
        probeInstructions(project, undefined, workspace.heuristics),
        '',
        repoMemory && repoMemory.pitfalls.length > 0
          ? `Known Repo Pitfalls:\n- ${repoMemory.pitfalls.join('\n- ')}`
          : '',
        rejectedSignatures.size > 0
          ? `Skip previously rejected signatures: ${[...rejectedSignatures].join(', ')}`
          : '',
        '',
        `Top Ranked Source Files:\n${contextDigest}`,
        '',
        `Propose up to ${this.maxCandidatesPerRound} concrete, proof-first bug candidates for this round's archetypes.`,
        'Return strict JSON: {"candidates":[{"title":"...","description":"...","category":"...","severity":"high","confidence":0.85,"targetFile":"...","targetSymbol":"...","reproHypothesis":"...","evidence":["..."],"probeCode":"..."}]}',
      ]
        .filter(Boolean)
        .join('\n');

      let parsedCandidates: z.infer<typeof RawCandidateSchema>[] = [];
      try {
        const rawResponse = await this.callLlm(nextBlackboard, prompt, {
          phase: 'audit_round',
          round: roundSpec.round,
          archetypes: roundSpec.archetypes,
        });
        const parsed = RoundResponseSchema.parse(
          JSON.parse(extractJsonObject(rawResponse)),
        );
        parsedCandidates = parsed.candidates.slice(0, this.maxCandidatesPerRound);
      } catch (err) {
        notes.push(
          `Round ${roundSpec.round} (${roundSpec.name}) failed to parse LLM output: ${(err as Error).message}`,
        );
        continue;
      }

      for (const candidate of parsedCandidates) {
        const sig = `${candidate.targetFile}::${candidate.targetSymbol ?? ''}::${candidate.category}`;
        if (rejectedSignatures.has(sig)) {
          notes.push(`Skipped previously rejected signature ${sig}`);
          continue;
        }
        if (
          hypotheses.some(
            (h) =>
              h.affectedFiles.includes(candidate.targetFile) &&
              h.targetSymbol === candidate.targetSymbol &&
              h.category === candidate.category,
          )
        ) {
          continue;
        }

        const probeResult = await this.executeProbeWithOneRetry({
          blackboard: nextBlackboard,
          repoDir,
          project,
          heuristics: workspace.heuristics,
          env: workspace.env.env,
          candidate,
          round: roundSpec.round,
          archetypes: roundSpec.archetypes,
        });

        if (probeResult.retried) {
          probeRetriesUsed += 1;
        }

        const hypothesisId = `hyp-${randomUUID().slice(0, 8)}`;

        if (probeResult.valid) {
          hypotheses.push({
            id: hypothesisId,
            title: candidate.title,
            description: candidate.description,
            category: candidate.category,
            severity: candidate.severity,
            confidence: candidate.confidence,
            affectedFiles: [candidate.targetFile],
            targetSymbol: candidate.targetSymbol,
            reproHypothesis: candidate.reproHypothesis,
            evidence: [
              ...candidate.evidence,
              `Probe command: ${probeResult.probeCommand}`,
            ],
            probeFilePath: probeResult.probeFilePath,
            probeCommand: probeResult.probeCommand,
            probeContent: probeResult.probeCode,
            probeOutput: probeResult.probeOutput,
          });
        } else {
          notes.push(
            `Probe rejected for "${candidate.title}": [${probeResult.assessment.category}] ${probeResult.assessment.reason}`,
          );
          if (this.memory) {
            await this.memory.recordProbe(blackboard.repo.fullName, {
              hypothesisId,
              title: candidate.title,
              signature: sig,
              verdict: 'rejected',
              proofType: 'unproven',
              reason: probeResult.assessment.reason,
              files: [candidate.targetFile],
            });
          }
        }
      }
    }

    const hunter: HunterContext = {
      scanScope: scope.scopeDirs,
      hypotheses,
      notes,
      completedAt: new Date().toISOString(),
    };

    nextBlackboard = setBlackboardSection(nextBlackboard, 'hunter', hunter);

    return {
      blackboard: nextBlackboard,
      grounding,
      hunter,
      rankedFiles,
      roundsExecuted: AUDIT_ROUNDS.length,
      probeRetriesUsed,
    };
  }
}

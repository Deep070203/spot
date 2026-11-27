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
  type EvaluationVerdict,
  type TaskBlackboard,
} from './blackboard.js';
import type { StateManager } from './state.js';
import { resolveRepoEnv } from '../utils/env.js';
import type { GitHubIssueSummary } from '../utils/github.js';
import { isHuntSource, selectHuntScope } from '../utils/hunt-scope.js';
import {
  assessProbeFailure,
  buildProbeCommand,
  commandTimeoutMs,
  detectProject,
  detectSourceHeuristics,
  huntExtensions,
  probeInstructions,
  resolveProbeFile,
} from '../utils/toolchain.js';

const execAsync = promisify(exec);
const execFileAsync = promisify(execFile);

export type ReproduceCompletionFn = (prompt: string) => Promise<string>;

export interface ReproduceOptions {
  completionFn?: ReproduceCompletionFn;
  apiKey?: string;
  stateManager?: StateManager;
  issueId?: string;
  maxRetries?: number;
}

export interface LocalIssueAnalysis {
  issueNumber: number;
  title: string;
  candidateFiles: string[];
  candidateSymbols: string[];
  hypothesis: BugHypothesis;
}

export interface ReproductionAttemptResult {
  reproduced: boolean;
  hypothesis: BugHypothesis;
  verdict: EvaluationVerdict;
}

export interface ProofAttemptResult {
  proven: boolean;
  blackboard: TaskBlackboard;
  hypothesis: BugHypothesis;
  verdict: EvaluationVerdict;
}

const ProbeGenerationSchema = z.object({
  targetFile: z.string().min(1),
  targetSymbol: z.string().optional(),
  category: z.string().default('ISSUE_REPRO'),
  severity: z.enum(['critical', 'high', 'medium', 'low']).default('high'),
  reproHypothesis: z.string().min(1),
  probeCode: z.string().min(1),
});

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

function extractMentionedTokens(text: string): {
  fileTokens: string[];
  symbolTokens: string[];
} {
  const fileTokens = new Set<string>();
  const symbolTokens = new Set<string>();

  for (const m of text.matchAll(/\b([\w./-]+\.(?:ts|tsx|mts|js|mjs|cjs|py|go|rs))\b/g)) {
    if (m[1]) fileTokens.add(m[1]);
  }
  for (const m of text.matchAll(/`([A-Za-z_$][\w$.]{2,60})`/g)) {
    if (m[1]) {
      const clean = m[1].split('.').pop()?.replace(/\(\)$/, '');
      if (clean) symbolTokens.add(clean);
    }
  }

  return {
    fileTokens: [...fileTokens],
    symbolTokens: [...symbolTokens],
  };
}

async function runCommand(
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

async function callMuse(
  blackboard: TaskBlackboard,
  prompt: string,
  options: ReproduceOptions,
): Promise<string> {
  if (options.completionFn) {
    return options.completionFn(prompt);
  }

  const apiKey =
    options.apiKey ??
    process.env[blackboard.model.apiKeyEnv] ??
    process.env.META_API_KEY ??
    process.env.MODEL_API_KEY;
  if (!apiKey || apiKey.trim().length === 0) {
    throw new Error(
      `Missing Meta API key (${blackboard.model.apiKeyEnv}) for reproduction probe generation`,
    );
  }

  const client = new OpenAI({
    baseURL: blackboard.model.baseURL || DEFAULT_META_BASE_URL,
    apiKey,
  });

  const start = Date.now();
  const response = await client.chat.completions.create({
    model: blackboard.model.model || DEFAULT_META_MODEL,
    temperature: 0.2,
    messages: [
      {
        role: 'system',
        content:
          'You are Spot Reproducer. Generate a minimal, real-code reproduction probe. Return strict JSON only.',
      },
      { role: 'user', content: prompt },
    ],
  });

  if (options.stateManager && options.issueId && options.stateManager.has(options.issueId)) {
    options.stateManager.recordMetrics(options.issueId, {
      llmCalls: 1,
      inputTokens: response.usage?.prompt_tokens ?? 0,
      outputTokens: response.usage?.completion_tokens ?? 0,
      durationMs: Date.now() - start,
    });
  }

  return response.choices[0]?.message?.content ?? '{}';
}

export async function analyzeIssueLocally(
  repoDir: string,
  issue: Pick<GitHubIssueSummary, 'number' | 'title' | 'body' | 'labels'>,
): Promise<LocalIssueAnalysis> {
  const resolvedRepo = path.resolve(repoDir);
  const project = await detectProject(resolvedRepo);
  const scope = await selectHuntScope(resolvedRepo, {
    extensions: huntExtensions(project),
    maxFiles: 60,
  });

  const fullText = `${issue.title}\n${issue.body}`;
  const { fileTokens, symbolTokens } = extractMentionedTokens(fullText);

  const candidateFiles = new Set<string>();
  for (const fileToken of fileTokens) {
    const normalized = fileToken.replace(/^\.\//, '');
    for (const tracked of scope.selectedFiles) {
      if (tracked === normalized || tracked.endsWith(`/${normalized}`)) {
        candidateFiles.add(tracked);
      }
    }
  }

  for (const symbol of symbolTokens.slice(0, 6)) {
    try {
      const { stdout } = await execFileAsync(
        'git',
        ['grep', '-l', '-F', symbol],
        { cwd: resolvedRepo, maxBuffer: 2 * 1024 * 1024 },
      );
      for (const line of stdout.split(/\r?\n/)) {
        const clean = line.trim();
        if (clean && isHuntSource(clean)) {
          candidateFiles.add(clean);
        }
      }
    } catch {
      // Symbol not found via git grep.
    }
  }

  if (candidateFiles.size === 0 && scope.selectedFiles.length > 0) {
    for (const fallbackFile of scope.selectedFiles.slice(0, 3)) {
      candidateFiles.add(fallbackFile);
    }
  }

  const affectedFiles = [...candidateFiles].slice(0, 5);
  const lowerLabels = issue.labels.map((l) => l.toLowerCase());
  const severity: BugHypothesis['severity'] = lowerLabels.some((l) =>
    /critical|security|crash|p0/.test(l),
  )
    ? 'critical'
    : lowerLabels.some((l) => /high|regression|p1/.test(l))
      ? 'high'
      : 'medium';

  const hypothesis: BugHypothesis = {
    id: `issue-${issue.number}-${randomUUID().slice(0, 6)}`,
    title: issue.title,
    description: issue.body.slice(0, 2000) || issue.title,
    category: 'ISSUE_REPRO',
    severity,
    confidence: 0.75,
    affectedFiles,
    targetSymbol: symbolTokens[0],
    reproHypothesis: `Reproduce GitHub issue #${issue.number}: ${issue.title}`,
    evidence: [
      `GitHub issue #${issue.number}`,
      ...affectedFiles.map((f) => `Matched candidate file: ${f}`),
    ],
    sourceIssueNumber: issue.number,
  };

  return {
    issueNumber: issue.number,
    title: issue.title,
    candidateFiles: affectedFiles,
    candidateSymbols: symbolTokens,
    hypothesis,
  };
}

export async function reproduceCandidateLocally(
  blackboard: TaskBlackboard,
  hypothesis: BugHypothesis,
  options: ReproduceOptions = {},
): Promise<ReproductionAttemptResult> {
  const repoDir = path.resolve(
    blackboard.repo.worktreePath ?? blackboard.repo.localPath,
  );
  const project = await detectProject(repoDir);
  const heuristics = await detectSourceHeuristics(repoDir, project);
  const resolvedEnv = await resolveRepoEnv(repoDir);
  const maxRetries = options.maxRetries ?? 1;

  let currentHypothesis: BugHypothesis = { ...hypothesis };
  let probeFilePath = currentHypothesis.probeFilePath;
  let probeCommand = currentHypothesis.probeCommand;
  let probeContent = currentHypothesis.probeContent;

  const primaryFile = currentHypothesis.affectedFiles[0] ?? '';

  if (!probeContent && !probeCommand) {
    const snippets: string[] = [];
    for (const relFile of currentHypothesis.affectedFiles.slice(0, 3)) {
      try {
        const content = await fs.readFile(path.join(repoDir, relFile), 'utf8');
        snippets.push(
          `### ${relFile}\n\`\`\`\n${content.split(/\r?\n/).slice(0, 120).join('\n')}\n\`\`\``,
        );
      } catch {
        // Ignore unreadable file.
      }
    }

    const prompt = [
      `Generate a reproduction probe for hypothesis "${currentHypothesis.title}"`,
      `Description: ${currentHypothesis.description}`,
      `Repro Hypothesis: ${currentHypothesis.reproHypothesis}`,
      probeInstructions(project, primaryFile, heuristics),
      snippets.join('\n\n'),
      'Return strict JSON: {"targetFile":"...","targetSymbol":"...","category":"...","severity":"high","reproHypothesis":"...","probeCode":"..."}',
    ].join('\n\n');

    const raw = await callMuse(blackboard, prompt, options);
    const parsed = ProbeGenerationSchema.parse(JSON.parse(extractJsonObject(raw)));
    probeContent = parsed.probeCode;
    currentHypothesis = {
      ...currentHypothesis,
      affectedFiles:
        currentHypothesis.affectedFiles.length > 0
          ? currentHypothesis.affectedFiles
          : [parsed.targetFile],
      targetSymbol: currentHypothesis.targetSymbol ?? parsed.targetSymbol,
      probeContent,
    };
  }

  if (!probeFilePath && probeContent) {
    probeFilePath = await resolveProbeFile(repoDir, project, {
      probeName: `spot_repro_${currentHypothesis.id}`,
      targetFile: currentHypothesis.affectedFiles[0] ?? primaryFile,
    });
  }

  if (!probeCommand && probeFilePath) {
    const spec = await buildProbeCommand(repoDir, project, probeFilePath);
    probeCommand = spec.command;
  }

  if (!probeCommand) {
    return {
      reproduced: false,
      hypothesis: currentHypothesis,
      verdict: {
        hypothesisId: currentHypothesis.id,
        accepted: false,
        proofType: 'unproven',
        confidence: 0,
        reproductionSteps: [currentHypothesis.reproHypothesis],
        rejectionReason: 'Unable to determine probeCommand for local reproduction.',
      },
    };
  }

  const timeoutMs = commandTimeoutMs('probe', project.primaryLanguage);

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    if (probeFilePath && probeContent) {
      await fs.mkdir(path.dirname(probeFilePath), { recursive: true });
      await fs.writeFile(probeFilePath, probeContent, 'utf8');
    }

    const execRes = await runCommand(
      probeCommand,
      repoDir,
      resolvedEnv.env,
      timeoutMs,
    );
    const combinedOutput = `${execRes.stdout}\n${execRes.stderr}`.trim();
    const assessment = assessProbeFailure({
      language: project.primaryLanguage,
      exitCode: execRes.exitCode,
      stdout: execRes.stdout,
      stderr: execRes.stderr,
      probeFilePath,
      probeContent,
      targetFile: currentHypothesis.affectedFiles[0],
      targetSymbol: currentHypothesis.targetSymbol,
    });

    if (assessment.valid) {
      currentHypothesis = {
        ...currentHypothesis,
        probeFilePath,
        probeCommand,
        probeContent,
        probeOutput: combinedOutput,
      };
      return {
        reproduced: true,
        hypothesis: currentHypothesis,
        verdict: {
          hypothesisId: currentHypothesis.id,
          accepted: true,
          proofType: 'repro_test',
          confidence: Math.max(0.75, currentHypothesis.confidence),
          reproductionSteps: [currentHypothesis.reproHypothesis],
          reproCommand: probeCommand,
          reproOutput: combinedOutput,
          failureSignature: combinedOutput.split(/\r?\n/)[0]?.slice(0, 200),
        },
      };
    }

    const canRetry =
      attempt < maxRetries &&
      !assessment.environmental &&
      Boolean(options.completionFn || options.apiKey || process.env[blackboard.model.apiKeyEnv]);

    if (!canRetry || !probeContent) {
      if (probeFilePath) {
        await fs.rm(probeFilePath, { force: true });
      }
      return {
        reproduced: false,
        hypothesis: currentHypothesis,
        verdict: {
          hypothesisId: currentHypothesis.id,
          accepted: false,
          proofType: 'unproven',
          confidence: 0.1,
          reproductionSteps: [currentHypothesis.reproHypothesis],
          reproCommand: probeCommand,
          reproOutput: combinedOutput,
          rejectionReason: assessment.reason,
        },
      };
    }

    const retryPrompt = [
      `Reproduction probe attempt failed (${assessment.category}): ${assessment.reason}`,
      `Output:\n${combinedOutput.slice(0, 1500)}`,
      `Previous probe:\n${probeContent}`,
      'Return revised JSON: {"targetFile":"...","reproHypothesis":"...","probeCode":"..."}',
    ].join('\n\n');
    const retryRaw = await callMuse(blackboard, retryPrompt, options);
    const parsedRetry = ProbeGenerationSchema.parse(
      JSON.parse(extractJsonObject(retryRaw)),
    );
    probeContent = parsedRetry.probeCode;
  }

  return {
    reproduced: false,
    hypothesis: currentHypothesis,
    verdict: {
      hypothesisId: currentHypothesis.id,
      accepted: false,
      proofType: 'unproven',
      confidence: 0,
      reproductionSteps: [currentHypothesis.reproHypothesis],
      rejectionReason: 'Reproduction failed after retries.',
    },
  };
}

export async function attemptReproductionProof(
  blackboard: TaskBlackboard,
  issue: Pick<GitHubIssueSummary, 'number' | 'title' | 'body' | 'labels'>,
  options: ReproduceOptions = {},
): Promise<ProofAttemptResult> {
  const repoDir = path.resolve(
    blackboard.repo.worktreePath ?? blackboard.repo.localPath,
  );
  const analysis = await analyzeIssueLocally(repoDir, issue);
  const result = await reproduceCandidateLocally(
    blackboard,
    analysis.hypothesis,
    options,
  );

  let nextBlackboard = setBlackboardSection(blackboard, 'hunter', {
    scanScope: analysis.candidateFiles,
    hypotheses: [result.hypothesis],
    notes: [`Targeted reproduction for issue #${issue.number}`],
    completedAt: new Date().toISOString(),
  });

  nextBlackboard = setBlackboardSection(nextBlackboard, 'evaluator', {
    selectedHypothesisId: result.reproduced ? result.hypothesis.id : undefined,
    verdicts: [result.verdict],
    completedAt: new Date().toISOString(),
  });

  return {
    proven: result.reproduced,
    blackboard: nextBlackboard,
    hypothesis: result.hypothesis,
    verdict: result.verdict,
  };
}

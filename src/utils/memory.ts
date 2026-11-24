import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import OpenAI from 'openai';
import { z } from 'zod';
import {
  DEFAULT_META_BASE_URL,
  DEFAULT_META_MODEL,
} from '../orchestrator/blackboard.js';

export const DEFAULT_AUTOHARNESS_DIR = path.join(os.homedir(), '.autoharness');
export const MAX_PROPOSALS = 30;

export const DEFAULT_GLOBAL_RULES: readonly string[] = [
  'Proof-first invariant: never propose or submit a fix without a reproducible failure or concrete static/runtime proof.',
  'Minimal diff discipline: change only the lines necessary to fix the verified root cause and avoid unrelated refactors.',
  'Regression test requirement: add or update an automated test that fails before the fix and passes after the fix.',
  'Verification gate: run the repository typecheck, build, and relevant test suite after the final edit before drafting a PR.',
  'Convention adherence: follow the repository CONTRIBUTING.md, issue/PR templates, and existing code style.',
  'Anti-hallucination guard: never invent file paths, symbol names, test results, or runtime outputs not directly observed.',
];

export const BehavioralRuleSchema = z.object({
  id: z.string(),
  category: z.enum(['pitfall', 'convention', 'pattern', 'false_positive', 'review']),
  rule: z.string(),
  rationale: z.string().optional(),
  tags: z.array(z.string()),
  confidence: z.number().min(0).max(1),
  source: z.string().optional(),
  createdAt: z.string(),
});
export type BehavioralRule = z.infer<typeof BehavioralRuleSchema>;

export const RepoMemoryFileSchema = z.object({
  repo: z.string(),
  globalRules: z.array(z.string()),
  rules: z.array(BehavioralRuleSchema),
  conventions: z.array(z.string()),
  pitfalls: z.array(z.string()),
  updatedAt: z.string(),
});
export type RepoMemoryFile = z.infer<typeof RepoMemoryFileSchema>;

export const ProbeHistoryEntrySchema = z.object({
  id: z.string(),
  hypothesisId: z.string(),
  title: z.string(),
  signature: z.string(),
  verdict: z.enum(['accepted', 'rejected', 'fixed', 'failed']),
  proofType: z.enum(['repro_test', 'runtime_trace', 'static_proof', 'unproven']),
  reason: z.string(),
  files: z.array(z.string()),
  timestamp: z.string(),
});
export type ProbeHistoryEntry = z.infer<typeof ProbeHistoryEntrySchema>;

export const RepoEnvironmentFileSchema = z.object({
  repo: z.string(),
  language: z.string().optional(),
  packageManager: z.string().optional(),
  nodeVersion: z.string().optional(),
  buildCommand: z.string().optional(),
  testCommand: z.string().optional(),
  lintCommand: z.string().optional(),
  notes: z.array(z.string()),
  detectedAt: z.string(),
  lastVerifiedAt: z.string().optional(),
});
export type RepoEnvironmentFile = z.infer<typeof RepoEnvironmentFileSchema>;

export const ProposalVaultEntrySchema = z.object({
  id: z.string(),
  repo: z.string(),
  title: z.string(),
  severity: z.enum(['critical', 'high', 'medium', 'low']),
  status: z.enum(['draft', 'verified', 'submitted', 'rejected']),
  hypothesisSummary: z.string(),
  issueTitle: z.string().optional(),
  issueBody: z.string().optional(),
  prTitle: z.string().optional(),
  prBody: z.string().optional(),
  diff: z.string().optional(),
  reproOutput: z.string().optional(),
  changedFiles: z.array(z.string()),
  pinned: z.boolean().optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
});
export type ProposalVaultEntry = z.infer<typeof ProposalVaultEntrySchema>;

export interface SaveProposalInput {
  id?: string;
  repo: string;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'draft' | 'verified' | 'submitted' | 'rejected';
  hypothesisSummary: string;
  issueTitle?: string;
  issueBody?: string;
  prTitle?: string;
  prBody?: string;
  diff?: string;
  reproOutput?: string;
  changedFiles?: string[];
  pinned?: boolean;
}

export interface DistillFeedbackInput {
  repo: string;
  feedback: string;
  prNumber?: number;
  diffSnippet?: string;
  context?: string;
}

export interface DistillFeedbackResult {
  addedRules: BehavioralRule[];
  memory: RepoMemoryFile;
  usedFallback: boolean;
}

export type TextCompletionFn = (prompt: string) => Promise<string>;

export interface BehavioralMemoryOptions {
  baseDir?: string;
  maxProposals?: number;
  apiKey?: string;
  baseURL?: string;
  model?: string;
  completionFn?: TextCompletionFn;
}

const DistilledPayloadSchema = z.object({
  lessons: z.array(
    z.object({
      category: z.enum([
        'pitfall',
        'convention',
        'pattern',
        'false_positive',
        'review',
      ]),
      rule: z.string().min(1),
      rationale: z.string().optional(),
      tags: z.array(z.string()).default([]),
      confidence: z.number().min(0).max(1).default(0.8),
    }),
  ),
});

function sanitizeRepoSlug(repo: string): string {
  return repo
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '__')
    .replace(/^_+|_+$/g, '');
}

async function readJsonFile<T>(
  filePath: string,
  schema: z.ZodType<T>,
): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, 'utf8');
    return schema.parse(JSON.parse(raw));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function writeJsonFile(filePath: string, data: unknown): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function extractJsonObject(raw: string): string {
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return fenced[1].trim();
  }
  const firstBrace = raw.indexOf('{');
  const lastBrace = raw.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    return raw.slice(firstBrace, lastBrace + 1);
  }
  return raw.trim();
}

export class BehavioralMemory {
  readonly baseDir: string;
  readonly maxProposals: number;
  private readonly apiKey?: string;
  private readonly baseURL: string;
  private readonly model: string;
  private readonly completionFn?: TextCompletionFn;

  constructor(options: BehavioralMemoryOptions = {}) {
    this.baseDir = path.resolve(options.baseDir ?? DEFAULT_AUTOHARNESS_DIR);
    this.maxProposals = options.maxProposals ?? MAX_PROPOSALS;
    this.apiKey =
      options.apiKey ?? process.env.META_API_KEY ?? process.env.MODEL_API_KEY;
    this.baseURL =
      options.baseURL ?? process.env.META_BASE_URL ?? DEFAULT_META_BASE_URL;
    this.model = options.model ?? process.env.META_MODEL ?? DEFAULT_META_MODEL;
    this.completionFn = options.completionFn;
  }

  getRepoDir(repo: string): string {
    return path.join(this.baseDir, 'repos', sanitizeRepoSlug(repo));
  }

  getMemoryPath(repo: string): string {
    return path.join(this.getRepoDir(repo), 'memory.json');
  }

  getProbesHistoryPath(repo: string): string {
    return path.join(this.getRepoDir(repo), 'probes_history.json');
  }

  getEnvironmentPath(repo: string): string {
    return path.join(this.getRepoDir(repo), 'environment.json');
  }

  getProposalsIndexPath(): string {
    return path.join(this.baseDir, 'proposals.json');
  }

  getProposalDir(id: string): string {
    return path.join(this.baseDir, 'proposals', id);
  }

  async loadRepoMemory(repo: string): Promise<RepoMemoryFile> {
    const filePath = this.getMemoryPath(repo);
    const existing = await readJsonFile(filePath, RepoMemoryFileSchema);
    if (existing) {
      return existing;
    }

    const initialized: RepoMemoryFile = {
      repo,
      globalRules: [...DEFAULT_GLOBAL_RULES],
      rules: [],
      conventions: [],
      pitfalls: [],
      updatedAt: new Date().toISOString(),
    };
    await writeJsonFile(filePath, initialized);
    return initialized;
  }

  async saveRepoMemory(repo: string, memory: RepoMemoryFile): Promise<RepoMemoryFile> {
    const updated: RepoMemoryFile = {
      ...memory,
      repo,
      globalRules:
        memory.globalRules.length > 0
          ? memory.globalRules
          : [...DEFAULT_GLOBAL_RULES],
      updatedAt: new Date().toISOString(),
    };
    await writeJsonFile(this.getMemoryPath(repo), updated);
    return updated;
  }

  async addRule(
    repo: string,
    input: Omit<BehavioralRule, 'id' | 'createdAt'> & { id?: string },
  ): Promise<RepoMemoryFile> {
    const memory = await this.loadRepoMemory(repo);
    const normalizedRule = input.rule.trim();
    const duplicate = memory.rules.some(
      (existing) => existing.rule.toLowerCase() === normalizedRule.toLowerCase(),
    );

    if (!duplicate) {
      memory.rules.push({
        id: input.id ?? randomUUID(),
        category: input.category,
        rule: normalizedRule,
        rationale: input.rationale,
        tags: input.tags,
        confidence: input.confidence,
        source: input.source,
        createdAt: new Date().toISOString(),
      });
      if (input.category === 'convention' && !memory.conventions.includes(normalizedRule)) {
        memory.conventions.push(normalizedRule);
      }
      if (input.category === 'pitfall' && !memory.pitfalls.includes(normalizedRule)) {
        memory.pitfalls.push(normalizedRule);
      }
    }

    return this.saveRepoMemory(repo, memory);
  }

  async loadProbesHistory(repo: string): Promise<ProbeHistoryEntry[]> {
    const existing = await readJsonFile(
      this.getProbesHistoryPath(repo),
      z.array(ProbeHistoryEntrySchema),
    );
    return existing ?? [];
  }

  async recordProbe(
    repo: string,
    entry: Omit<ProbeHistoryEntry, 'id' | 'timestamp'> & {
      id?: string;
      timestamp?: string;
    },
  ): Promise<ProbeHistoryEntry[]> {
    const history = await this.loadProbesHistory(repo);
    const nextEntry: ProbeHistoryEntry = {
      ...entry,
      id: entry.id ?? randomUUID(),
      timestamp: entry.timestamp ?? new Date().toISOString(),
    };
    history.push(nextEntry);
    await writeJsonFile(this.getProbesHistoryPath(repo), history);
    return history;
  }

  async hasSeenProbe(repo: string, signature: string): Promise<boolean> {
    const history = await this.loadProbesHistory(repo);
    return history.some((entry) => entry.signature === signature);
  }

  async loadEnvironment(repo: string): Promise<RepoEnvironmentFile | null> {
    return readJsonFile(this.getEnvironmentPath(repo), RepoEnvironmentFileSchema);
  }

  async saveEnvironment(
    repo: string,
    env: Omit<RepoEnvironmentFile, 'repo' | 'detectedAt'> & {
      detectedAt?: string;
    },
  ): Promise<RepoEnvironmentFile> {
    const existing = await this.loadEnvironment(repo);
    const record: RepoEnvironmentFile = {
      ...env,
      repo,
      notes: env.notes ?? existing?.notes ?? [],
      detectedAt:
        env.detectedAt ?? existing?.detectedAt ?? new Date().toISOString(),
      lastVerifiedAt: env.lastVerifiedAt ?? new Date().toISOString(),
    };
    await writeJsonFile(this.getEnvironmentPath(repo), record);
    return record;
  }

  async listProposals(repo?: string): Promise<ProposalVaultEntry[]> {
    const all =
      (await readJsonFile(
        this.getProposalsIndexPath(),
        z.array(ProposalVaultEntrySchema),
      )) ?? [];
    if (!repo) {
      return all;
    }
    return all.filter((item) => item.repo === repo);
  }

  async getProposal(id: string): Promise<ProposalVaultEntry | null> {
    const proposalPath = path.join(this.getProposalDir(id), 'proposal.json');
    const detailed = await readJsonFile(proposalPath, ProposalVaultEntrySchema);
    if (detailed) {
      return detailed;
    }
    const index = await this.listProposals();
    return index.find((item) => item.id === id) ?? null;
  }

  async saveProposal(input: SaveProposalInput): Promise<ProposalVaultEntry> {
    const now = new Date().toISOString();
    const id = input.id ?? randomUUID();
    const existing = await this.getProposal(id);

    const entry: ProposalVaultEntry = {
      id,
      repo: input.repo,
      title: input.title,
      severity: input.severity,
      status: input.status,
      hypothesisSummary: input.hypothesisSummary,
      issueTitle: input.issueTitle,
      issueBody: input.issueBody,
      prTitle: input.prTitle,
      prBody: input.prBody,
      diff: input.diff,
      reproOutput: input.reproOutput,
      changedFiles: input.changedFiles ?? [],
      pinned: input.pinned ?? existing?.pinned ?? false,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };

    const proposalDir = this.getProposalDir(id);
    await fs.mkdir(proposalDir, { recursive: true });
    await writeJsonFile(path.join(proposalDir, 'proposal.json'), entry);
    if (entry.diff !== undefined) {
      await fs.writeFile(path.join(proposalDir, 'patch.diff'), entry.diff, 'utf8');
    }
    if (entry.reproOutput !== undefined) {
      await fs.writeFile(
        path.join(proposalDir, 'repro.txt'),
        entry.reproOutput,
        'utf8',
      );
    }

    const index = await this.listProposals();
    const withoutCurrent = index.filter((item) => item.id !== id);
    const updatedIndex = [entry, ...withoutCurrent];

    const evicted: ProposalVaultEntry[] = [];
    while (updatedIndex.length > this.maxProposals) {
      let evictIdx = -1;
      for (let i = updatedIndex.length - 1; i >= 0; i -= 1) {
        if (!updatedIndex[i]?.pinned) {
          evictIdx = i;
          break;
        }
      }
      if (evictIdx === -1) {
        evictIdx = updatedIndex.length - 1;
      }
      const [removed] = updatedIndex.splice(evictIdx, 1);
      if (removed) {
        evicted.push(removed);
      }
    }

    await writeJsonFile(this.getProposalsIndexPath(), updatedIndex);

    for (const removed of evicted) {
      await fs.rm(this.getProposalDir(removed.id), {
        recursive: true,
        force: true,
      });
    }

    return entry;
  }

  async deleteProposal(id: string): Promise<boolean> {
    const index = await this.listProposals();
    const filtered = index.filter((item) => item.id !== id);
    if (filtered.length === index.length) {
      return false;
    }
    await writeJsonFile(this.getProposalsIndexPath(), filtered);
    await fs.rm(this.getProposalDir(id), { recursive: true, force: true });
    return true;
  }

  async distillReviewFeedback(
    input: DistillFeedbackInput,
  ): Promise<DistillFeedbackResult> {
    const prompt = [
      'You are distilling maintainer PR/code-review feedback into durable engineering rules.',
      `Repository: ${input.repo}`,
      input.prNumber ? `PR #${input.prNumber}` : '',
      input.context ? `Context:\n${input.context}` : '',
      input.diffSnippet ? `Diff:\n${input.diffSnippet}` : '',
      `Review Feedback:\n${input.feedback}`,
      '',
      'Return strict JSON matching:',
      '{"lessons":[{"category":"review|pitfall|convention|pattern|false_positive","rule":"...","rationale":"...","tags":["..."],"confidence":0.85}]}',
    ]
      .filter(Boolean)
      .join('\n\n');

    let rawText: string | null = null;
    let usedFallback = false;

    if (this.completionFn) {
      rawText = await this.completionFn(prompt);
    } else if (this.apiKey && this.apiKey.trim().length > 0) {
      const client = new OpenAI({
        baseURL: this.baseURL,
        apiKey: this.apiKey,
      });
      const response = await client.chat.completions.create({
        model: this.model,
        temperature: 0.2,
        messages: [
          {
            role: 'system',
            content:
              'Distill code review feedback into concise, actionable JSON rules. Respond with JSON only.',
          },
          { role: 'user', content: prompt },
        ],
      });
      rawText = response.choices[0]?.message?.content ?? null;
    } else {
      usedFallback = true;
    }

    const addedRules: BehavioralRule[] = [];
    let memory = await this.loadRepoMemory(input.repo);

    if (rawText) {
      try {
        const parsed = DistilledPayloadSchema.parse(
          JSON.parse(extractJsonObject(rawText)),
        );
        for (const lesson of parsed.lessons) {
          const ruleEntry: BehavioralRule = {
            id: randomUUID(),
            category: lesson.category,
            rule: lesson.rule.trim(),
            rationale: lesson.rationale,
            tags: lesson.tags,
            confidence: lesson.confidence,
            source: input.prNumber ? `PR #${input.prNumber}` : 'review-feedback',
            createdAt: new Date().toISOString(),
          };
          const beforeCount = memory.rules.length;
          memory = await this.addRule(input.repo, ruleEntry);
          if (memory.rules.length > beforeCount) {
            const persisted = memory.rules[memory.rules.length - 1];
            if (persisted) addedRules.push(persisted);
          }
        }
        return { addedRules, memory, usedFallback };
      } catch {
        usedFallback = true;
      }
    }

    const fallbackRuleText = input.feedback.trim().split('\n')[0]?.slice(0, 240);
    if (fallbackRuleText) {
      const beforeCount = memory.rules.length;
      memory = await this.addRule(input.repo, {
        category: 'review',
        rule: fallbackRuleText,
        rationale: 'Fallback distillation from raw review feedback',
        tags: ['review-feedback'],
        confidence: 0.6,
        source: input.prNumber ? `PR #${input.prNumber}` : 'review-feedback',
      });
      if (memory.rules.length > beforeCount) {
        const persisted = memory.rules[memory.rules.length - 1];
        if (persisted) addedRules.push(persisted);
      }
    }

    return { addedRules, memory, usedFallback: true };
  }
}

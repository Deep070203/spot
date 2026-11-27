import fs from 'node:fs/promises';
import path from 'node:path';
import OpenAI from 'openai';
import { z } from 'zod';
import {
  DEFAULT_META_BASE_URL,
  DEFAULT_META_MODEL,
  setBlackboardSection,
  type BugHypothesis,
  type FaultLocation,
  type LocalizationContext,
  type TaskBlackboard,
} from './blackboard.js';
import type { StateManager } from './state.js';
import { assessReachability } from '../utils/toolchain.js';

export type LocalizationCompletionFn = (prompt: string) => Promise<string>;

export interface RunLocalizationOptions {
  hypothesisId?: string;
  completionFn?: LocalizationCompletionFn;
  apiKey?: string;
  stateManager?: StateManager;
  issueId?: string;
}

export interface RunLocalizationResult {
  blackboard: TaskBlackboard;
  localization: LocalizationContext;
}

const LocalizationLlmSchema = z.object({
  rootCause: z.string().min(1),
  primaryFile: z.string().min(1),
  symbols: z.array(z.string()).default([]),
  locations: z
    .array(
      z.object({
        file: z.string().min(1),
        startLine: z.number().int().positive(),
        endLine: z.number().int().positive(),
        symbol: z.string().optional(),
        role: z.enum(['root_cause', 'propagator', 'symptom', 'test']),
        snippet: z.string().optional(),
      }),
    )
    .default([]),
  callChain: z.array(z.string()).default([]),
  relatedTests: z.array(z.string()).default([]),
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

function selectTargetHypothesis(
  blackboard: TaskBlackboard,
  explicitId?: string,
): BugHypothesis {
  const hypotheses = blackboard.hunter?.hypotheses ?? [];
  const targetId =
    explicitId ?? blackboard.evaluator?.selectedHypothesisId ?? hypotheses[0]?.id;
  const found = hypotheses.find((h) => h.id === targetId) ?? hypotheses[0];
  if (!found) {
    throw new Error('Cannot run localization: no BugHypothesis present on blackboard');
  }
  return found;
}

async function findRelatedTestFiles(
  repoDir: string,
  primaryFile: string,
): Promise<string[]> {
  const stem = path
    .basename(primaryFile, path.extname(primaryFile))
    .replace(/[^a-zA-Z0-9_-]/g, '');
  const matches: string[] = [];

  async function walk(dir: string): Promise<void> {
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (
        entry.name.startsWith('.') ||
        ['node_modules', 'dist', 'build', 'target', '.venv', 'venv'].includes(
          entry.name,
        )
      ) {
        continue;
      }
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (
        entry.isFile() &&
        /(?:test|spec)/i.test(full) &&
        (!stem || entry.name.includes(stem) || full.includes('tests'))
      ) {
        matches.push(path.relative(repoDir, full).replace(/\\/g, '/'));
      }
    }
  }

  await walk(repoDir);
  return matches.slice(0, 15);
}

export async function runLocalization(
  blackboard: TaskBlackboard,
  options: RunLocalizationOptions = {},
): Promise<RunLocalizationResult> {
  const repoDir = path.resolve(
    blackboard.repo.worktreePath ?? blackboard.repo.localPath,
  );
  const hypothesis = selectTargetHypothesis(blackboard, options.hypothesisId);
  const primaryFile = hypothesis.affectedFiles[0] ?? '';

  if (options.stateManager && options.issueId && options.stateManager.has(options.issueId)) {
    options.stateManager.transition(options.issueId, 'localizing', {
      reason: `Localizing root cause for ${hypothesis.title}`,
    });
  }

  const [sourceContent, reachability, relatedTests] = await Promise.all([
    primaryFile
      ? fs.readFile(path.join(repoDir, primaryFile), 'utf8').catch(() => '')
      : Promise.resolve(''),
    primaryFile && hypothesis.targetSymbol
      ? assessReachability(repoDir, primaryFile, hypothesis.targetSymbol)
      : Promise.resolve(null),
    primaryFile ? findRelatedTestFiles(repoDir, primaryFile) : Promise.resolve([]),
  ]);

  const verdict = blackboard.evaluator?.verdicts.find(
    (v) => v.hypothesisId === hypothesis.id,
  );

  const numberedSource = sourceContent
    .split(/\r?\n/)
    .slice(0, 200)
    .map((line, i) => `${i + 1}: ${line}`)
    .join('\n');

  const apiKey =
    options.apiKey ??
    process.env[blackboard.model.apiKeyEnv] ??
    process.env.META_API_KEY ??
    process.env.MODEL_API_KEY;

  let localizationData: z.infer<typeof LocalizationLlmSchema> | null = null;

  if (options.completionFn || (apiKey && apiKey.trim().length > 0)) {
    const prompt = [
      'Spot Fault Localization',
      `Hypothesis: ${hypothesis.title}`,
      `Description: ${hypothesis.description}`,
      `Primary File: ${primaryFile}`,
      `Target Symbol: ${hypothesis.targetSymbol ?? 'unknown'}`,
      verdict?.reproOutput ? `Probe Output:\n${verdict.reproOutput.slice(0, 1500)}` : '',
      reachability
        ? `Static Callers: ${reachability.callers.map((c) => `${c.file}:${c.line}`).join(', ') || 'none'}`
        : '',
      numberedSource ? `Source:\n${numberedSource}` : '',
      'Return strict JSON: {"rootCause":"...","primaryFile":"...","symbols":["..."],"locations":[{"file":"...","startLine":1,"endLine":5,"symbol":"...","role":"root_cause","snippet":"..."}],"callChain":["..."],"relatedTests":["..."]}',
    ]
      .filter(Boolean)
      .join('\n\n');

    let raw: string;
    if (options.completionFn) {
      raw = await options.completionFn(prompt);
    } else {
      const client = new OpenAI({
        baseURL: blackboard.model.baseURL || DEFAULT_META_BASE_URL,
        apiKey: apiKey!,
      });
      const start = Date.now();
      const response = await client.chat.completions.create({
        model: blackboard.model.model || DEFAULT_META_MODEL,
        temperature: 0.1,
        messages: [
          {
            role: 'system',
            content:
              'You are Spot Fault Localizer. Pinpoint the exact root cause file, lines, symbols, and call chain. Return strict JSON only.',
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
      raw = response.choices[0]?.message?.content ?? '{}';
    }

    try {
      localizationData = LocalizationLlmSchema.parse(
        JSON.parse(extractJsonObject(raw)),
      );
    } catch {
      localizationData = null;
    }
  }

  const fallbackLocations: FaultLocation[] = [];
  if (primaryFile) {
    const lines = sourceContent.split(/\r?\n/);
    const sym = hypothesis.targetSymbol;
    const matchIdx = sym
      ? lines.findIndex((l) => l.includes(sym))
      : 0;
    const startLine = Math.max(1, (matchIdx >= 0 ? matchIdx : 0) + 1);
    fallbackLocations.push({
      file: primaryFile,
      startLine,
      endLine: Math.min(lines.length || startLine, startLine + 5),
      symbol: sym,
      role: 'root_cause',
      snippet: lines.slice(startLine - 1, startLine + 4).join('\n') || undefined,
    });
  }

  const localization: LocalizationContext = {
    hypothesisId: hypothesis.id,
    rootCause: localizationData?.rootCause ?? hypothesis.description,
    primaryFile: localizationData?.primaryFile ?? primaryFile,
    symbols:
      localizationData && localizationData.symbols.length > 0
        ? localizationData.symbols
        : hypothesis.targetSymbol
          ? [hypothesis.targetSymbol]
          : [],
    locations:
      localizationData && localizationData.locations.length > 0
        ? localizationData.locations
        : fallbackLocations,
    callChain:
      localizationData && localizationData.callChain.length > 0
        ? localizationData.callChain
        : [
            ...(reachability?.callers.map((c) => `${c.file}:${c.line}`) ?? []),
            `${primaryFile}${hypothesis.targetSymbol ? `#${hypothesis.targetSymbol}` : ''}`,
          ],
    relatedTests:
      localizationData && localizationData.relatedTests.length > 0
        ? localizationData.relatedTests
        : relatedTests,
    completedAt: new Date().toISOString(),
  };

  const nextBlackboard = setBlackboardSection(
    blackboard,
    'localization',
    localization,
  );

  return {
    blackboard: nextBlackboard,
    localization,
  };
}

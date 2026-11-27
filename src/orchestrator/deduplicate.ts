import { execFile } from 'node:child_process';
import path from 'node:path';
import { promisify } from 'node:util';
import type { BugHypothesis } from './blackboard.js';
import type { GitHubService } from '../utils/github.js';
import type { BehavioralMemory } from '../utils/memory.js';

const execFileAsync = promisify(execFile);

export interface DeduplicateCandidateOptions {
  repoFullName: string;
  repoDir: string;
  memory?: BehavioralMemory;
  github?: GitHubService;
  owner?: string;
  repo?: string;
}

export interface DeduplicationResult {
  duplicate: boolean;
  reason?: string;
  matchedSource?: 'probe_history' | 'proposal_vault' | 'git_log' | 'github_prs';
  matchedReference?: string;
}

function normalizeTitleTokens(title: string): Set<string> {
  return new Set(
    title
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length >= 3),
  );
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let intersection = 0;
  for (const item of a) {
    if (b.has(item)) intersection += 1;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

export async function deduplicateCandidate(
  candidate: Pick<
    BugHypothesis,
    'id' | 'title' | 'category' | 'affectedFiles' | 'targetSymbol' | 'sourceIssueNumber'
  >,
  options: DeduplicateCandidateOptions,
): Promise<DeduplicationResult> {
  const primaryFile = candidate.affectedFiles[0] ?? '';
  const signature = `${primaryFile}::${candidate.targetSymbol ?? ''}::${candidate.category}`;
  const candidateTokens = normalizeTitleTokens(candidate.title);

  // 1. Check BehavioralMemory probes history (previously rejected or fixed signatures)
  if (options.memory) {
    const history = await options.memory.loadProbesHistory(options.repoFullName);
    for (const entry of history) {
      if (
        entry.signature === signature &&
        (entry.verdict === 'rejected' || entry.verdict === 'fixed')
      ) {
        return {
          duplicate: true,
          matchedSource: 'probe_history',
          matchedReference: entry.id,
          reason: `Signature "${signature}" already recorded as ${entry.verdict} in probes_history.json (${entry.reason}).`,
        };
      }
    }

    // 2. Check Proposal & Patch Vault for already verified/submitted proposals on same files/title
    const proposals = await options.memory.listProposals(options.repoFullName);
    for (const prop of proposals) {
      if (prop.status === 'rejected') continue;
      const sameFiles =
        primaryFile.length > 0 && prop.changedFiles.includes(primaryFile);
      const sim = jaccardSimilarity(
        candidateTokens,
        normalizeTitleTokens(prop.title),
      );
      if ((sameFiles && sim >= 0.5) || sim >= 0.75) {
        return {
          duplicate: true,
          matchedSource: 'proposal_vault',
          matchedReference: prop.id,
          reason: `Matches existing proposal "${prop.title}" (${prop.id}, status=${prop.status}) in vault.`,
        };
      }
    }
  }

  // 3. Check recent git commits on disk to avoid re-reporting already-fixed issues
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['log', '-n', '30', '--oneline'],
      { cwd: path.resolve(options.repoDir), maxBuffer: 2 * 1024 * 1024 },
    );
    for (const line of stdout.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (
        candidate.sourceIssueNumber &&
        new RegExp(`#${candidate.sourceIssueNumber}\\b`).test(trimmed)
      ) {
        return {
          duplicate: true,
          matchedSource: 'git_log',
          matchedReference: trimmed.split(' ')[0],
          reason: `Recent commit already references issue #${candidate.sourceIssueNumber}: ${trimmed}`,
        };
      }
      const msg = trimmed.replace(/^[0-9a-f]+\s+/i, '');
      if (jaccardSimilarity(candidateTokens, normalizeTitleTokens(msg)) >= 0.75) {
        return {
          duplicate: true,
          matchedSource: 'git_log',
          matchedReference: trimmed.split(' ')[0],
          reason: `Recent commit closely matches candidate: ${trimmed}`,
        };
      }
    }
  } catch {
    // Repo might not have git log yet.
  }

  // 4. Optional GitHub open PR check
  if (options.github && options.owner && options.repo) {
    try {
      const openPRs = await options.github.withAuthFallback(async (client) => {
        const { data } = await client.rest.pulls.list({
          owner: options.owner!,
          repo: options.repo!,
          state: 'open',
          per_page: 25,
        });
        return data;
      });

      for (const pr of openPRs) {
        const prText = `${pr.title}\n${pr.body ?? ''}`;
        if (
          candidate.sourceIssueNumber &&
          new RegExp(`#${candidate.sourceIssueNumber}\\b`).test(prText)
        ) {
          return {
            duplicate: true,
            matchedSource: 'github_prs',
            matchedReference: `#${pr.number}`,
            reason: `Open PR #${pr.number} (${pr.title}) already addresses issue #${candidate.sourceIssueNumber}.`,
          };
        }
        if (
          jaccardSimilarity(candidateTokens, normalizeTitleTokens(pr.title)) >= 0.7
        ) {
          return {
            duplicate: true,
            matchedSource: 'github_prs',
            matchedReference: `#${pr.number}`,
            reason: `Open PR #${pr.number} closely matches candidate title (${pr.title}).`,
          };
        }
      }
    } catch {
      // Non-fatal if offline or rate-limited.
    }
  }

  return { duplicate: false };
}

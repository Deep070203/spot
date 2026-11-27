import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import type { GitHubService } from '../utils/github.js';

const execFileAsync = promisify(execFile);

export interface RepoPolicy {
  hasContributing: boolean;
  contributingPath?: string;
  contributingSummary?: string;
  prTemplate?: string;
  prTemplatePath?: string;
  issueTemplates: Array<{ name: string; path: string; content: string }>;
  requiresCLA: boolean;
  requiresDCO: boolean;
  requiresConventionalCommits: boolean;
  preferredTestCommand?: string;
  notes: string[];
}

export interface MergePatterns {
  commitStyle: 'conventional' | 'imperative' | 'mixed';
  recentCommitPrefixes: string[];
  typicalBranchPrefixes: string[];
  includesTestsInFixes: boolean;
  sampleCommitMessages: string[];
  prConventions: string[];
}

async function readIfExists(filePath: string): Promise<string | null> {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {
    return null;
  }
}

export async function extractRepoPolicy(
  repoDir: string,
  options: { github?: GitHubService; owner?: string; repo?: string } = {},
): Promise<RepoPolicy> {
  const resolved = path.resolve(repoDir);

  let contributingPath: string | undefined;
  let contributingContent: string | null = null;
  for (const rel of [
    'CONTRIBUTING.md',
    '.github/CONTRIBUTING.md',
    'docs/CONTRIBUTING.md',
    'contributing.md',
  ]) {
    const content = await readIfExists(path.join(resolved, rel));
    if (content !== null) {
      contributingPath = rel;
      contributingContent = content;
      break;
    }
  }

  if (!contributingContent && options.github && options.owner && options.repo) {
    try {
      const remoteContrib = await options.github.getContributing(
        options.owner,
        options.repo,
      );
      if (remoteContrib) {
        contributingPath = remoteContrib.path;
        contributingContent = remoteContrib.content;
      }
    } catch {
      // Non-fatal fallback.
    }
  }

  let prTemplatePath: string | undefined;
  let prTemplate: string | undefined;
  for (const rel of [
    '.github/PULL_REQUEST_TEMPLATE.md',
    '.github/pull_request_template.md',
    'PULL_REQUEST_TEMPLATE.md',
    'pull_request_template.md',
  ]) {
    const content = await readIfExists(path.join(resolved, rel));
    if (content !== null) {
      prTemplatePath = rel;
      prTemplate = content;
      break;
    }
  }

  const issueTemplates: RepoPolicy['issueTemplates'] = [];
  const issueTemplateDir = path.join(resolved, '.github', 'ISSUE_TEMPLATE');
  try {
    const entries = await fs.readdir(issueTemplateDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const content = await readIfExists(path.join(issueTemplateDir, entry.name));
      if (content !== null) {
        issueTemplates.push({
          name: entry.name,
          path: `.github/ISSUE_TEMPLATE/${entry.name}`,
          content,
        });
      }
    }
  } catch {
    // No issue template dir.
  }

  const singleIssueTpl = await readIfExists(
    path.join(resolved, '.github', 'ISSUE_TEMPLATE.md'),
  );
  if (singleIssueTpl !== null) {
    issueTemplates.push({
      name: 'ISSUE_TEMPLATE.md',
      path: '.github/ISSUE_TEMPLATE.md',
      content: singleIssueTpl,
    });
  }

  if (
    (!prTemplate || issueTemplates.length === 0) &&
    options.github &&
    options.owner &&
    options.repo
  ) {
    try {
      const remoteTemplates = await options.github.getTemplates(
        options.owner,
        options.repo,
      );
      if (!prTemplate && remoteTemplates.prTemplates[0]) {
        prTemplate = remoteTemplates.prTemplates[0].content;
        prTemplatePath = remoteTemplates.prTemplates[0].path;
      }
      if (issueTemplates.length === 0) {
        for (const tpl of remoteTemplates.issueTemplates) {
          issueTemplates.push({
            name: tpl.name,
            path: tpl.path,
            content: tpl.content,
          });
        }
      }
    } catch {
      // Non-fatal fallback.
    }
  }

  const combinedPolicyText = [
    contributingContent ?? '',
    prTemplate ?? '',
    ...issueTemplates.map((t) => t.content),
  ].join('\n');

  const requiresCLA =
    /\b(?:Contributor License Agreement|CLA)\b/i.test(combinedPolicyText);
  const requiresDCO =
    /\b(?:Developer Certificate of Origin|DCO|Signed-off-by|git commit -s)\b/i.test(
      combinedPolicyText,
    );

  const hasCommitlint = Boolean(
    (await readIfExists(path.join(resolved, '.commitlintrc.json'))) ??
      (await readIfExists(path.join(resolved, 'commitlint.config.js'))) ??
      (await readIfExists(path.join(resolved, 'commitlint.config.cjs'))),
  );
  const requiresConventionalCommits =
    hasCommitlint ||
    /\bconventional commits?\b/i.test(combinedPolicyText);

  const notes: string[] = [];
  if (requiresCLA) notes.push('Repository requires signing a CLA before merging.');
  if (requiresDCO) notes.push('Repository requires DCO Signed-off-by trailer (`git commit -s`).');
  if (requiresConventionalCommits) {
    notes.push('Use Conventional Commits format (e.g. `fix(scope): description`).');
  }
  if (prTemplatePath) {
    notes.push(`Follow PR template at ${prTemplatePath}.`);
  }

  return {
    hasContributing: Boolean(contributingContent),
    contributingPath,
    contributingSummary: contributingContent?.slice(0, 2000),
    prTemplate,
    prTemplatePath,
    issueTemplates,
    requiresCLA,
    requiresDCO,
    requiresConventionalCommits,
    notes,
  };
}

export async function extractMergePatterns(
  repoDir: string,
  options: { github?: GitHubService; owner?: string; repo?: string } = {},
): Promise<MergePatterns> {
  const resolved = path.resolve(repoDir);
  const sampleCommitMessages: string[] = [];
  const prefixCounts = new Map<string, number>();
  let conventionalCount = 0;

  try {
    const { stdout } = await execFileAsync(
      'git',
      ['log', '-n', '25', '--format=%s'],
      { cwd: resolved, maxBuffer: 2 * 1024 * 1024 },
    );
    for (const line of stdout.split(/\r?\n/)) {
      const msg = line.trim();
      if (!msg) continue;
      sampleCommitMessages.push(msg);
      const convMatch = msg.match(
        /^(feat|fix|chore|docs|refactor|test|perf|ci|build)(?:\([^)]+\))?!?:/i,
      );
      if (convMatch?.[1]) {
        conventionalCount += 1;
        const prefix = convMatch[1].toLowerCase();
        prefixCounts.set(prefix, (prefixCounts.get(prefix) ?? 0) + 1);
      }
    }
  } catch {
    // Repo may not have git history.
  }

  let includesTestsInFixes = false;
  try {
    const { stdout } = await execFileAsync(
      'git',
      ['log', '-n', '15', '--name-only', '--grep=fix', '-i', '--format='],
      { cwd: resolved, maxBuffer: 2 * 1024 * 1024 },
    );
    if (/(?:test|spec)/i.test(stdout)) {
      includesTestsInFixes = true;
    }
  } catch {
    includesTestsInFixes = true;
  }

  const prConventions: string[] = [];
  if (options.github && options.owner && options.repo) {
    try {
      const mergedPRs = await options.github.withAuthFallback(async (client) => {
        const { data } = await client.rest.pulls.list({
          owner: options.owner!,
          repo: options.repo!,
          state: 'closed',
          sort: 'updated',
          direction: 'desc',
          per_page: 10,
        });
        return data.filter((p) => p.merged_at !== null);
      });

      for (const pr of mergedPRs.slice(0, 5)) {
        prConventions.push(`Merged PR #${pr.number}: ${pr.title}`);
      }
    } catch {
      // Non-fatal if offline/rate-limited.
    }
  }

  const ratio =
    sampleCommitMessages.length > 0
      ? conventionalCount / sampleCommitMessages.length
      : 0;
  const commitStyle: MergePatterns['commitStyle'] =
    ratio >= 0.6 ? 'conventional' : ratio === 0 ? 'imperative' : 'mixed';

  const recentCommitPrefixes = [...prefixCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([k]) => k);

  return {
    commitStyle,
    recentCommitPrefixes,
    typicalBranchPrefixes:
      commitStyle === 'conventional' ? ['fix/', 'bugfix/'] : ['spot/fix-', 'fix/'],
    includesTestsInFixes,
    sampleCommitMessages: sampleCommitMessages.slice(0, 10),
    prConventions,
  };
}

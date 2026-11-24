import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import type { Octokit } from 'octokit';
import {
  createTaskBlackboard,
  setBlackboardSection,
} from '../src/orchestrator/blackboard.js';
import { StateManager } from '../src/orchestrator/state.js';
import { GitHubService, parseRepoSlug } from '../src/utils/github.js';
import {
  BehavioralMemory,
  DEFAULT_GLOBAL_RULES,
  MAX_PROPOSALS,
} from '../src/utils/memory.js';

const tempDirs: string[] = [];

async function makeTempDir(prefix: string): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

after(async () => {
  await Promise.all(
    tempDirs.map((dir) => fs.rm(dir, { recursive: true, force: true })),
  );
});

describe('Phase 1 contracts: StateManager & TaskBlackboard', () => {
  it('tracks lifecycle transitions, metrics, and subscribers', () => {
    const manager = new StateManager();
    const events: string[] = [];
    const unsubscribe = manager.subscribe((ev) => {
      events.push(`${ev.type}:${ev.state.stage}`);
    });

    const created = manager.create({
      repo: 'octocat/hello-world',
      title: 'Fix null pointer in parser',
    });
    assert.equal(created.stage, 'queued');
    assert.equal(created.status, 'pending');

    const hunting = manager.transition(created.id, 'hunting', {
      reason: 'Starting bug hunt',
    });
    assert.equal(hunting.stage, 'hunting');
    assert.equal(hunting.status, 'running');
    assert.equal(hunting.history.length, 2);

    const updated = manager.recordMetrics(created.id, {
      llmCalls: 2,
      inputTokens: 500,
      outputTokens: 120,
      durationMs: 340,
    });
    assert.equal(updated.metrics.llmCalls, 2);
    assert.equal(updated.metrics.inputTokens, 500);

    const completed = manager.transition(created.id, 'completed');
    assert.equal(completed.status, 'completed');
    assert.deepEqual(events, [
      'state:created:queued',
      'state:stage_changed:hunting',
      'state:updated:hunting',
      'state:stage_changed:completed',
    ]);

    unsubscribe();
  });

  it('creates and updates TaskBlackboard sections with Muse Ember defaults', () => {
    const bb = createTaskBlackboard({
      owner: 'octocat',
      name: 'hello-world',
      fullName: 'octocat/hello-world',
      url: 'https://github.com/octocat/hello-world',
      defaultBranch: 'main',
      localPath: '/tmp/hello-world',
    });

    assert.equal(bb.model.model, 'muse-ember-1.0');
    assert.equal(bb.model.baseURL, 'https://api.meta.ai/v1');

    const withHunter = setBlackboardSection(bb, 'hunter', {
      scanScope: ['src/'],
      hypotheses: [
        {
          id: 'hyp-1',
          title: 'Unhandled empty input',
          description: 'Empty string causes out-of-bounds access',
          category: 'bounds',
          severity: 'high',
          confidence: 0.9,
          affectedFiles: ['src/parser.ts'],
          reproHypothesis: 'Call parse("")',
          evidence: ['src/parser.ts:42'],
        },
      ],
      notes: [],
      completedAt: new Date().toISOString(),
    });

    assert.equal(withHunter.hunter?.hypotheses.length, 1);
    assert.equal(withHunter.hunter?.hypotheses[0]?.id, 'hyp-1');
  });
});

describe('Phase 2: GitHubService', () => {
  it('parses owner/repo slugs and GitHub URLs', () => {
    assert.deepEqual(parseRepoSlug('expressjs/express'), {
      owner: 'expressjs',
      repo: 'express',
      fullName: 'expressjs/express',
    });
    assert.deepEqual(
      parseRepoSlug('https://github.com/expressjs/express.git'),
      {
        owner: 'expressjs',
        repo: 'express',
        fullName: 'expressjs/express',
      },
    );
  });

  it('falls back to unauthenticated Octokit on 401 and ranks labeled bug issues', async () => {
    const calls: string[] = [];

    const mockFactory = (token?: string): Octokit => {
      const client = {
        rest: {
          search: {
            issuesAndPullRequests: async () => {
              calls.push(token ? 'auth' : 'unauth');
              if (token) {
                const err = new Error('Bad credentials') as Error & {
                  status: number;
                };
                err.status = 401;
                throw err;
              }
              return {
                data: {
                  items: [
                    {
                      number: 10,
                      title: 'Unlabeled issue',
                      body: 'Something is broken in runtime',
                      state: 'open',
                      labels: [],
                      url: 'https://api.github.com/repos/org/app/issues/10',
                      html_url: 'https://github.com/org/app/issues/10',
                      user: { login: 'alice' },
                      created_at: '2026-01-01T00:00:00Z',
                      updated_at: '2026-01-03T00:00:00Z',
                      comments: 1,
                    },
                    {
                      number: 11,
                      title: 'Confirmed crash bug',
                      body: 'Detailed reproduction steps '.repeat(10),
                      state: 'open',
                      labels: [{ name: 'bug' }],
                      url: 'https://api.github.com/repos/org/app/issues/11',
                      html_url: 'https://github.com/org/app/issues/11',
                      user: { login: 'bob' },
                      created_at: '2026-01-01T00:00:00Z',
                      updated_at: '2026-01-02T00:00:00Z',
                      comments: 2,
                    },
                    {
                      number: 12,
                      title: 'Duplicate issue',
                      body: 'dup',
                      state: 'open',
                      labels: [{ name: 'duplicate' }],
                      url: 'https://api.github.com/repos/org/app/issues/12',
                      html_url: 'https://github.com/org/app/issues/12',
                      user: { login: 'carol' },
                      created_at: '2026-01-01T00:00:00Z',
                      updated_at: '2026-01-04T00:00:00Z',
                      comments: 0,
                    },
                    {
                      number: 13,
                      title: 'Pull request should be skipped',
                      body: 'pr',
                      state: 'open',
                      labels: [{ name: 'bug' }],
                      pull_request: { url: 'https://api.github.com/pr/13' },
                      url: 'https://api.github.com/repos/org/app/issues/13',
                      html_url: 'https://github.com/org/app/pull/13',
                      user: { login: 'dan' },
                      created_at: '2026-01-01T00:00:00Z',
                      updated_at: '2026-01-05T00:00:00Z',
                      comments: 0,
                    },
                  ],
                },
              };
            },
          },
          pulls: {
            get: async () => ({ data: 'diff --git a/a.ts b/a.ts\n+fix' }),
            listFiles: async () => ({
              data: [
                {
                  filename: 'a.ts',
                  status: 'modified',
                  additions: 1,
                  deletions: 0,
                  changes: 1,
                  patch: '+fix',
                  blob_url: 'https://github.com/org/app/blob/main/a.ts',
                },
              ],
            }),
          },
          repos: {
            getContent: async ({ path: reqPath }: { path: string }) => {
              if (reqPath === 'CONTRIBUTING.md') {
                return {
                  data: {
                    type: 'file',
                    path: 'CONTRIBUTING.md',
                    name: 'CONTRIBUTING.md',
                    content: Buffer.from('# Contributing\nRun tests').toString(
                      'base64',
                    ),
                  },
                };
              }
              if (reqPath === '.github/PULL_REQUEST_TEMPLATE.md') {
                return {
                  data: {
                    type: 'file',
                    path: '.github/PULL_REQUEST_TEMPLATE.md',
                    name: 'PULL_REQUEST_TEMPLATE.md',
                    content: Buffer.from('## Summary').toString('base64'),
                  },
                };
              }
              const err = new Error('Not Found') as Error & { status: number };
              err.status = 404;
              throw err;
            },
          },
        },
      };
      return client as unknown as Octokit;
    };

    const gh = new GitHubService({
      token: 'bad-token',
      octokitFactory: mockFactory,
    });

    assert.equal(gh.isAuthenticated, true);
    const issues = await gh.openBugIssueQuery('org', 'app');
    assert.equal(gh.isAuthenticated, false);
    assert.deepEqual(calls, ['auth', 'unauth']);
    assert.equal(issues.length, 2);
    assert.equal(issues[0]?.number, 11);
    assert.equal(issues[1]?.number, 10);

    const diff = await gh.getPRDiff('org', 'app', 13);
    assert.match(diff, /diff --git/);

    const files = await gh.getPRFiles('org', 'app', 13);
    assert.equal(files[0]?.filename, 'a.ts');

    const contrib = await gh.getContributing('org', 'app');
    assert.equal(contrib?.path, 'CONTRIBUTING.md');
    assert.match(contrib?.content ?? '', /Run tests/);

    const templates = await gh.getTemplates('org', 'app');
    assert.equal(templates.prTemplates.length, 1);
    assert.equal(templates.prTemplates[0]?.content, '## Summary');
  });
});

describe('Phase 2: BehavioralMemory & Proposal Vault', () => {
  it('isolates memory.json, probes_history.json, and environment.json per repo', async () => {
    const baseDir = await makeTempDir('spot-mem-');
    const memory = new BehavioralMemory({ baseDir });

    assert.ok(DEFAULT_GLOBAL_RULES.length >= 5);

    await memory.addRule('org/repo-a', {
      category: 'pitfall',
      rule: 'Never mutate shared config object in place',
      tags: ['config'],
      confidence: 0.9,
    });

    await memory.recordProbe('org/repo-a', {
      hypothesisId: 'h1',
      title: 'Null config crash',
      signature: 'sig-null-config',
      verdict: 'rejected',
      proofType: 'unproven',
      reason: 'Guarded by caller',
      files: ['src/config.ts'],
    });

    await memory.saveEnvironment('org/repo-a', {
      language: 'TypeScript',
      packageManager: 'npm',
      testCommand: 'npm test',
      notes: ['Node 22'],
    });

    const memA = await memory.loadRepoMemory('org/repo-a');
    const memB = await memory.loadRepoMemory('org/repo-b');
    assert.equal(memA.rules.length, 1);
    assert.equal(memB.rules.length, 0);
    assert.equal(await memory.hasSeenProbe('org/repo-a', 'sig-null-config'), true);
    assert.equal(await memory.hasSeenProbe('org/repo-b', 'sig-null-config'), false);

    const envA = await memory.loadEnvironment('org/repo-a');
    const envB = await memory.loadEnvironment('org/repo-b');
    assert.equal(envA?.testCommand, 'npm test');
    assert.equal(envB, null);
  });

  it('maintains a 30-item Proposal & Patch Vault in proposals.json and proposals/<id>/', async () => {
    const baseDir = await makeTempDir('spot-vault-');
    const memory = new BehavioralMemory({ baseDir });

    for (let i = 1; i <= 35; i += 1) {
      await memory.saveProposal({
        id: `prop-${i}`,
        repo: 'org/repo-a',
        title: `Fix #${i}`,
        severity: 'medium',
        status: 'verified',
        hypothesisSummary: `Summary ${i}`,
        diff: `--- a/f.ts\n+++ b/f.ts\n// fix ${i}\n`,
        changedFiles: ['f.ts'],
      });
    }

    const list = await memory.listProposals();
    assert.equal(list.length, MAX_PROPOSALS);
    assert.equal(list[0]?.id, 'prop-35');
    assert.equal(await memory.getProposal('prop-1'), null);
    assert.equal(await memory.getProposal('prop-5'), null);
    assert.ok(await memory.getProposal('prop-6'));

    const patchFile = await fs.readFile(
      path.join(baseDir, 'proposals', 'prop-35', 'patch.diff'),
      'utf8',
    );
    assert.match(patchFile, /fix 35/);
  });

  it('distills review feedback into repo memory via LLM completion', async () => {
    const baseDir = await makeTempDir('spot-distill-');
    const memory = new BehavioralMemory({
      baseDir,
      completionFn: async () =>
        JSON.stringify({
          lessons: [
            {
              category: 'review',
              rule: 'Always preserve error.cause when wrapping errors',
              rationale: 'Maintainers inspect error.cause in tests',
              tags: ['errors'],
              confidence: 0.95,
            },
          ],
        }),
    });

    const result = await memory.distillReviewFeedback({
      repo: 'org/repo-a',
      prNumber: 42,
      feedback: 'Please preserve error.cause when rethrowing.',
    });

    assert.equal(result.usedFallback, false);
    assert.equal(result.addedRules.length, 1);
    assert.equal(
      result.memory.rules[0]?.rule,
      'Always preserve error.cause when wrapping errors',
    );
  });
});

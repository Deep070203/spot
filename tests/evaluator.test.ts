import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { promisify } from 'node:util';
import { createTaskBlackboard } from '../src/orchestrator/blackboard.js';
import {
  BugEvaluatorAgent,
  MIN_EVALUATOR_CONFIDENCE,
} from '../src/orchestrator/evaluator.js';
import {
  AUDIT_ROUNDS,
  BUG_ARCHETYPES,
  BugHunterAgent,
} from '../src/orchestrator/hunter.js';
import { StateManager } from '../src/orchestrator/state.js';
import { BehavioralMemory } from '../src/utils/memory.js';

const execFileAsync = promisify(execFile);
const tempDirs: string[] = [];

async function makeTempDir(prefix: string): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

async function createSampleBuggyRepo(): Promise<string> {
  const repoDir = await makeTempDir('spot-eval-repo-');
  await fs.mkdir(path.join(repoDir, 'src'), { recursive: true });
  await fs.mkdir(path.join(repoDir, 'tests'), { recursive: true });

  await fs.writeFile(
    path.join(repoDir, 'package.json'),
    JSON.stringify({
      name: 'sample-buggy',
      type: 'module',
      scripts: {
        test: 'node --test tests/smoke.test.mjs',
      },
    }),
  );

  await fs.writeFile(
    path.join(repoDir, 'src', 'config.mjs'),
    `export function resolvePort(opts = {}) {
  // ZERO_STATE_CLAMP bug: port=0 is valid ephemeral port, falsy || coerces 0 to 3000
  return opts.port || 3000;
}

function deadHelper() {
  return 42;
}
`,
  );

  await fs.writeFile(
    path.join(repoDir, 'src', 'index.mjs'),
    `import { resolvePort } from './config.mjs';

export function startServer(opts) {
  return resolvePort(opts);
}
`,
  );

  await fs.writeFile(
    path.join(repoDir, 'tests', 'smoke.test.mjs'),
    `import assert from 'node:assert/strict';
import test from 'node:test';
import { startServer } from '../src/index.mjs';

test('default port works', () => {
  assert.equal(startServer({}), 3000);
});
`,
  );

  await execFileAsync('git', ['init'], { cwd: repoDir });
  await execFileAsync('git', ['config', 'user.email', 'spot@example.com'], {
    cwd: repoDir,
  });
  await execFileAsync('git', ['config', 'user.name', 'Spot'], { cwd: repoDir });
  await execFileAsync('git', ['add', '.'], { cwd: repoDir });
  await execFileAsync('git', ['commit', '-m', 'init sample repo'], {
    cwd: repoDir,
  });

  return repoDir;
}

after(async () => {
  await Promise.all(
    tempDirs.map((dir) => fs.rm(dir, { recursive: true, force: true })),
  );
});

describe('Phase 5: BugHunterAgent & BugEvaluatorAgent', () => {
  it('runs baseline check, churn+AST+keyword ranking, 4-round 13-archetype audit, and 1-retry probe loop', async () => {
    const repoDir = await createSampleBuggyRepo();
    const memoryDir = await makeTempDir('spot-eval-mem-');
    const memory = new BehavioralMemory({ baseDir: memoryDir });
    const stateManager = new StateManager();
    const issue = stateManager.create({ repo: 'acme/sample-buggy' });

    assert.equal(AUDIT_ROUNDS.length, 4);
    assert.equal(BUG_ARCHETYPES.length, 13);

    const roundsSeen: number[] = [];
    let retryCalled = false;

    const hunter = new BugHunterAgent({
      memory,
      stateManager,
      issueId: issue.id,
      completionFn: async (_prompt, meta) => {
        if (meta.phase === 'audit_round' && meta.round) {
          roundsSeen.push(meta.round);
          if (meta.round === 1) {
            // Deliberately emit a probe that passes (exit 0) on first try to trigger 1-retry feedback loop.
            return JSON.stringify({
              candidates: [
                {
                  title: 'resolvePort clamps port 0 to 3000',
                  description: 'opts.port || 3000 discards valid port 0.',
                  category: 'ZERO_STATE_CLAMP',
                  severity: 'high',
                  confidence: 0.9,
                  targetFile: 'src/config.mjs',
                  targetSymbol: 'resolvePort',
                  reproHypothesis: 'Calling resolvePort({ port: 0 }) returns 3000 instead of 0.',
                  evidence: ['src/config.mjs:3'],
                  probeCode: `import assert from 'node:assert/strict';
import { resolvePort } from '../src/config.mjs';
// Bad initial assertion: does not fail on bug
assert.equal(resolvePort({}), 3000);
`,
                },
              ],
            });
          }
          return JSON.stringify({ candidates: [] });
        }

        if (meta.phase === 'probe_retry') {
          retryCalled = true;
          return JSON.stringify({
            probeCode: `import assert from 'node:assert/strict';
import { resolvePort } from '../src/config.mjs';
assert.equal(resolvePort({ port: 0 }), 0);
`,
            notes: 'Asserts ephemeral port 0 is preserved.',
          });
        }

        return JSON.stringify({ candidates: [] });
      },
    });

    const bb = createTaskBlackboard({
      owner: 'acme',
      name: 'sample-buggy',
      fullName: 'acme/sample-buggy',
      url: 'https://github.com/acme/sample-buggy',
      defaultBranch: 'main',
      localPath: repoDir,
    });

    const result = await hunter.run(bb);

    assert.deepEqual(roundsSeen, [1, 2, 3, 4]);
    assert.equal(retryCalled, true);
    assert.equal(result.probeRetriesUsed, 1);
    assert.equal(result.grounding.baseline.testsPassed, true);
    assert.equal(result.hunter.hypotheses.length, 1);
    assert.equal(
      result.hunter.hypotheses[0]?.targetSymbol,
      'resolvePort',
    );
    assert.ok(result.hunter.hypotheses[0]?.probeCommand);
    assert.equal(stateManager.get(issue.id)?.stage, 'hunting');
  });

  it('dynamically re-executes probe, verifies git grep upstream callers, and accepts confidence >= 0.7', async () => {
    const repoDir = await createSampleBuggyRepo();
    const memoryDir = await makeTempDir('spot-eval-mem2-');
    const memory = new BehavioralMemory({ baseDir: memoryDir });
    const stateManager = new StateManager();
    const issue = stateManager.create({ repo: 'acme/sample-buggy' });

    const probeFile = path.join(repoDir, 'tests', 'repro_port.probe.mjs');
    const probeCode = `import assert from 'node:assert/strict';
import { resolvePort } from '../src/config.mjs';
assert.equal(resolvePort({ port: 0 }), 0);
`;
    await fs.writeFile(probeFile, probeCode, 'utf8');

    const bb = createTaskBlackboard({
      owner: 'acme',
      name: 'sample-buggy',
      fullName: 'acme/sample-buggy',
      url: 'https://github.com/acme/sample-buggy',
      defaultBranch: 'main',
      localPath: repoDir,
    });

    bb.hunter = {
      scanScope: ['src'],
      hypotheses: [
        {
          id: 'hyp-valid',
          title: 'resolvePort clamps port 0 to 3000',
          description: 'opts.port || 3000 treats 0 as falsy',
          category: 'ZERO_STATE_CLAMP',
          severity: 'high',
          confidence: 0.9,
          affectedFiles: ['src/config.mjs'],
          targetSymbol: 'resolvePort',
          reproHypothesis: 'resolvePort({ port: 0 }) returns 3000',
          evidence: ['src/config.mjs:3'],
          probeFilePath: probeFile,
          probeCommand: `node ${probeFile}`,
          probeContent: probeCode,
        },
      ],
      notes: [],
      completedAt: new Date().toISOString(),
    };

    let sawUpstreamCallerInPrompt = false;
    const evaluator = new BugEvaluatorAgent({
      memory,
      stateManager,
      issueId: issue.id,
      completionFn: async (prompt) => {
        if (prompt.includes('src/index.mjs')) {
          sawUpstreamCallerInPrompt = true;
        }
        return JSON.stringify({
          accepted: true,
          confidence: 0.92,
          proofType: 'repro_test',
          rationale: 'Real upstream caller in src/index.mjs and failing assertion reproduced.',
          reproductionSteps: ['Call startServer({ port: 0 })'],
        });
      },
    });

    const evalResult = await evaluator.evaluate(bb);
    assert.equal(sawUpstreamCallerInPrompt, true);
    assert.equal(evalResult.acceptedCount, 1);
    assert.equal(evalResult.rejectedCount, 0);
    assert.equal(evalResult.evaluator.selectedHypothesisId, 'hyp-valid');
    assert.equal(evalResult.evaluator.verdicts[0]?.accepted, true);
    assert.ok(
      (evalResult.evaluator.verdicts[0]?.confidence ?? 0) >=
        MIN_EVALUATOR_CONFIDENCE,
    );
    assert.equal(stateManager.get(issue.id)?.stage, 'evaluating');
  });

  it('filters out low-confidence (< 0.7), unreachable dead code, and mock stand-in hallucinations', async () => {
    const repoDir = await createSampleBuggyRepo();
    const memoryDir = await makeTempDir('spot-eval-mem3-');
    const memory = new BehavioralMemory({ baseDir: memoryDir });

    const realProbeFile = path.join(repoDir, 'tests', 'repro_real.probe.mjs');
    const realProbeCode = `import assert from 'node:assert/strict';
import { resolvePort } from '../src/config.mjs';
assert.equal(resolvePort({ port: 0 }), 0);
`;
    await fs.writeFile(realProbeFile, realProbeCode, 'utf8');

    const mockProbeFile = path.join(repoDir, 'tests', 'repro_mock.probe.mjs');
    const mockProbeCode = `import assert from 'node:assert/strict';
function resolvePort(opts = {}) { return opts.port || 3000; }
assert.equal(resolvePort({ port: 0 }), 0);
`;
    await fs.writeFile(mockProbeFile, mockProbeCode, 'utf8');

    const deadProbeFile = path.join(repoDir, 'tests', 'repro_dead.probe.mjs');
    await fs.writeFile(deadProbeFile, realProbeCode, 'utf8');

    const bb = createTaskBlackboard({
      owner: 'acme',
      name: 'sample-buggy',
      fullName: 'acme/sample-buggy',
      url: 'https://github.com/acme/sample-buggy',
      defaultBranch: 'main',
      localPath: repoDir,
    });

    bb.hunter = {
      scanScope: ['src'],
      hypotheses: [
        {
          id: 'hyp-low-conf',
          title: 'Low confidence candidate',
          description: 'Skeptical architect only 0.55 confident',
          category: 'ZERO_STATE_CLAMP',
          severity: 'medium',
          confidence: 0.6,
          affectedFiles: ['src/config.mjs'],
          targetSymbol: 'resolvePort',
          reproHypothesis: 'Call resolvePort({ port: 0 })',
          evidence: [],
          probeFilePath: realProbeFile,
          probeCommand: `node ${realProbeFile}`,
          probeContent: realProbeCode,
        },
        {
          id: 'hyp-mock',
          title: 'Mock stand-in hallucination',
          description: 'Redefines function inline',
          category: 'ZERO_STATE_CLAMP',
          severity: 'high',
          confidence: 0.9,
          affectedFiles: ['src/config.mjs'],
          targetSymbol: 'resolvePort',
          reproHypothesis: 'Fake repro',
          evidence: [],
          probeFilePath: mockProbeFile,
          probeCommand: `node ${mockProbeFile}`,
          probeContent: mockProbeCode,
        },
        {
          id: 'hyp-unreachable',
          title: 'Unreachable dead helper',
          description: 'deadHelper has no callers and is not exported',
          category: 'ZERO_STATE_CLAMP',
          severity: 'low',
          confidence: 0.8,
          affectedFiles: ['src/config.mjs'],
          targetSymbol: 'deadHelper',
          reproHypothesis: 'Dead code hypothesis',
          evidence: [],
          probeFilePath: deadProbeFile,
          probeCommand: `node ${deadProbeFile}`,
          probeContent: realProbeCode,
        },
      ],
      notes: [],
      completedAt: new Date().toISOString(),
    };

    const evaluator = new BugEvaluatorAgent({
      memory,
      completionFn: async () =>
        JSON.stringify({
          accepted: true,
          confidence: 0.55,
          proofType: 'repro_test',
          rationale: 'Uncertain edge case.',
          reproductionSteps: [],
        }),
    });

    const evalResult = await evaluator.evaluate(bb);
    assert.equal(evalResult.acceptedCount, 0);
    assert.equal(evalResult.rejectedCount, 3);
    assert.equal(evalResult.evaluator.selectedHypothesisId, undefined);
    assert.match(
      evalResult.evaluator.verdicts[0]?.rejectionReason ?? '',
      /0\.70|threshold/i,
    );
    assert.match(
      evalResult.evaluator.verdicts[1]?.rejectionReason ?? '',
      /mock_standin/i,
    );
    assert.match(
      evalResult.evaluator.verdicts[2]?.rejectionReason ?? '',
      /reachability|callers/i,
    );
  });
});

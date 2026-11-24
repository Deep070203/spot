import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import { promisify } from 'node:util';
import {
  discoverPythonPackageRoots,
  normalizeCommandWithVenv,
  resolveRepoEnv,
} from '../src/utils/env.js';
import {
  changesOnDisk,
  isHuntSource,
  scopeContainingExtensions,
  selectHuntScope,
} from '../src/utils/hunt-scope.js';

const execFileAsync = promisify(execFile);
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

describe('Phase 3: env.ts (Python stdlib shadowing & venv isolation)', () => {
  it('discovers src/ and PEP 420 namespace package roots without adding package interiors', async () => {
    const repoDir = await makeTempDir('spot-py-roots-');

    // src/ layout with PEP 420 namespace package `acme` and interior shadowing module `json.py`
    await fs.mkdir(path.join(repoDir, 'src', 'acme', 'billing'), {
      recursive: true,
    });
    await fs.writeFile(
      path.join(repoDir, 'src', 'acme', 'billing', '__init__.py'),
      '',
    );
    await fs.writeFile(
      path.join(repoDir, 'src', 'acme', 'billing', 'json.py'),
      'x = 1\n',
    );

    // Monorepo subproject with top-level regular package
    await fs.mkdir(path.join(repoDir, 'packages', 'worker', 'worker_pkg'), {
      recursive: true,
    });
    await fs.writeFile(
      path.join(repoDir, 'packages', 'worker', 'pyproject.toml'),
      '[project]\nname = "worker"\n',
    );
    await fs.writeFile(
      path.join(repoDir, 'packages', 'worker', 'worker_pkg', '__init__.py'),
      '',
    );
    await fs.writeFile(
      path.join(repoDir, 'packages', 'worker', 'worker_pkg', 'logging.py'),
      '',
    );

    const roots = await discoverPythonPackageRoots(repoDir);
    assert.ok(roots.includes(path.join(repoDir, 'src')));
    assert.ok(roots.includes(path.join(repoDir, 'packages', 'worker')));
    // Package interiors MUST NOT be on PYTHONPATH
    assert.ok(!roots.includes(path.join(repoDir, 'src', 'acme')));
    assert.ok(!roots.includes(path.join(repoDir, 'src', 'acme', 'billing')));
    assert.ok(
      !roots.includes(path.join(repoDir, 'packages', 'worker', 'worker_pkg')),
    );
  });

  it('sets PYTHONSAFEPATH=1 and normalizes python/pytest commands with -P and venv', async () => {
    const repoDir = await makeTempDir('spot-venv-');
    const venvBin = path.join(repoDir, '.venv', 'bin');
    await fs.mkdir(venvBin, { recursive: true });
    await fs.writeFile(path.join(venvBin, 'python'), '#!/bin/sh\n', {
      mode: 0o755,
    });

    const resolved = await resolveRepoEnv(repoDir, {
      baseEnv: { PATH: '/usr/bin:/bin' },
    });

    assert.equal(resolved.env.PYTHONSAFEPATH, '1');
    assert.equal(resolved.venvDir, path.join(repoDir, '.venv'));
    assert.ok(resolved.env.PATH?.startsWith(venvBin));

    const pyCmd = normalizeCommandWithVenv('python -m unittest', resolved);
    assert.equal(pyCmd, `${path.join(venvBin, 'python')} -P -m unittest`);

    const alreadySafe = normalizeCommandWithVenv(
      'python3 -P script.py',
      resolved,
    );
    assert.equal(alreadySafe, `${path.join(venvBin, 'python')} -P script.py`);

    const pytestCmd = normalizeCommandWithVenv('pytest -q tests/', resolved);
    assert.equal(
      pytestCmd,
      `${path.join(venvBin, 'python')} -P -m pytest -q tests/`,
    );
  });
});

describe('Phase 3: hunt-scope.ts (monorepo scoping, churn & disk changes)', () => {
  it('filters source files accurately with isHuntSource and scopeContainingExtensions', () => {
    assert.equal(isHuntSource('src/core/parser.ts'), true);
    assert.equal(isHuntSource('services/api/handler.py'), true);
    assert.equal(isHuntSource('src/core/parser.test.ts'), false);
    assert.equal(isHuntSource('tests/test_handler.py'), false);
    assert.equal(isHuntSource('node_modules/pkg/index.js'), false);
    assert.equal(isHuntSource('dist/index.d.ts'), false);
    assert.equal(isHuntSource('README.md'), false);

    const scopes = scopeContainingExtensions(
      [
        'packages/auth/src/token.ts',
        'packages/billing/models.py',
        'docs/guide.md',
      ],
      ['.ts'],
    );
    assert.deepEqual(scopes, ['packages/auth']);
  });

  it('uses git ls-files -z, churn clustering, and changesOnDisk in selectHuntScope', async () => {
    const repoDir = await makeTempDir('spot-scope-');
    await execFileAsync('git', ['init'], { cwd: repoDir });
    await execFileAsync('git', ['config', 'user.email', 'spot@example.com'], {
      cwd: repoDir,
    });
    await execFileAsync('git', ['config', 'user.name', 'Spot'], {
      cwd: repoDir,
    });

    await fs.mkdir(path.join(repoDir, 'packages', 'hot'), { recursive: true });
    await fs.mkdir(path.join(repoDir, 'packages', 'cold'), { recursive: true });
    await fs.mkdir(path.join(repoDir, 'tests'), { recursive: true });

    await fs.writeFile(
      path.join(repoDir, 'packages', 'hot', 'engine.ts'),
      'export const v = 1;\n',
    );
    await fs.writeFile(
      path.join(repoDir, 'packages', 'cold', 'util.ts'),
      'export const u = 1;\n',
    );
    await fs.writeFile(
      path.join(repoDir, 'tests', 'engine.test.ts'),
      'export {};\n',
    );

    await execFileAsync('git', ['add', '.'], { cwd: repoDir });
    await execFileAsync('git', ['commit', '-m', 'init'], { cwd: repoDir });

    // Create extra churn on packages/hot/engine.ts
    await fs.writeFile(
      path.join(repoDir, 'packages', 'hot', 'engine.ts'),
      'export const v = 2;\n',
    );
    await execFileAsync('git', ['add', 'packages/hot/engine.ts'], {
      cwd: repoDir,
    });
    await execFileAsync('git', ['commit', '-m', 'churn hot'], { cwd: repoDir });

    // Create dirty uncommitted file
    await fs.writeFile(
      path.join(repoDir, 'packages', 'hot', 'dirty.ts'),
      'export const d = 1;\n',
    );

    const disk = await changesOnDisk(repoDir);
    assert.equal(disk.hasChanges, true);
    assert.deepEqual(disk.untracked, ['packages/hot/dirty.ts']);

    const scope = await selectHuntScope(repoDir);
    assert.equal(scope.sourceFilesCount, 2);
    assert.equal(scope.selectedFiles[0], 'packages/hot/engine.ts');
    assert.equal(scope.churnClusters[0]?.scope, 'packages/hot');
  });
});

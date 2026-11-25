import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { after, describe, it } from 'node:test';
import {
  assessProbeFailure,
  assessReachability,
  auditNote,
  buildProbeCommand,
  commandTimeoutMs,
  detectProject,
  detectSourceHeuristics,
  goProbeFile,
  huntExtensions,
  isEnvironmentalProbeFailure,
  prepareToolchainWorkspace,
  probeInstructions,
  resolveProbeFile,
  rustPackage,
  typescriptProbeFile,
} from '../src/utils/toolchain.js';

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

describe('Phase 4: Multi-language project detection and probe command building', () => {
  it('detects TypeScript ESM projects and builds tsx probe commands', async () => {
    const repoDir = await makeTempDir('spot-ts-');
    await fs.mkdir(path.join(repoDir, 'tests'), { recursive: true });
    await fs.mkdir(path.join(repoDir, 'node_modules', '.bin'), {
      recursive: true,
    });
    await fs.writeFile(
      path.join(repoDir, 'node_modules', '.bin', 'tsx'),
      '#!/bin/sh\n',
      { mode: 0o755 },
    );
    await fs.writeFile(
      path.join(repoDir, 'package.json'),
      JSON.stringify({
        name: 'demo-ts',
        type: 'module',
        scripts: { test: 'tsx --test', typecheck: 'tsc --noEmit' },
        devDependencies: { typescript: '^5.8.0', tsx: '^4.19.0' },
      }),
    );
    await fs.writeFile(
      path.join(repoDir, 'tsconfig.json'),
      JSON.stringify({ compilerOptions: { moduleResolution: 'NodeNext' } }),
    );

    const project = await detectProject(repoDir);
    assert.equal(project.primaryLanguage, 'typescript');
    assert.equal(project.isEsm, true);
    assert.deepEqual(huntExtensions(project), ['.ts', '.tsx', '.mts', '.cts']);
    assert.ok(commandTimeoutMs('probe', 'typescript') > 0);

    const probePath = await typescriptProbeFile(repoDir, 'null-clamp');
    assert.ok(probePath.endsWith(path.join('tests', 'null_clamp.probe.mts')));

    const cmd = await buildProbeCommand(repoDir, project, probePath);
    assert.match(cmd.command, /tsx/);
    assert.match(cmd.command, /null_clamp\.probe\.mts/);

    const workspace = await prepareToolchainWorkspace(repoDir, project);
    assert.equal(workspace.ready, true);
    assert.equal(workspace.heuristics.requiresJsExtensionsInImports, true);
    assert.match(auditNote(project, workspace.heuristics), /NodeNext/);
    assert.match(
      probeInstructions(project, 'src/parser.ts', workspace.heuristics),
      /\.js/,
    );
  });

  it('detects Go modules, resolves package probe files, and builds go test commands', async () => {
    const repoDir = await makeTempDir('spot-go-');
    await fs.mkdir(path.join(repoDir, 'pkg', 'router'), { recursive: true });
    await fs.writeFile(
      path.join(repoDir, 'go.mod'),
      'module github.com/acme/gateway\n\ngo 1.23\n',
    );
    await fs.writeFile(
      path.join(repoDir, 'pkg', 'router', 'router.go'),
      'package router\n\nfunc Match(p string) bool { return false }\n',
    );

    const project = await detectProject(repoDir);
    assert.equal(project.primaryLanguage, 'go');
    assert.deepEqual(huntExtensions(project), ['.go']);

    const goProbe = await goProbeFile(repoDir, 'pkg/router/router.go', 'match_bug');
    assert.equal(goProbe.packageName, 'router');
    assert.ok(goProbe.filePath.endsWith(path.join('pkg', 'router', 'match_bug_test.go')));

    const cmd = await buildProbeCommand(repoDir, project, goProbe.filePath, {
      testName: 'TestSpotProbeMatchBug',
    });
    assert.equal(
      cmd.command,
      'go test -v -run TestSpotProbeMatchBug ./pkg/router',
    );
  });

  it('detects Cargo workspaces, resolves crate package names, and builds cargo test commands', async () => {
    const repoDir = await makeTempDir('spot-rust-');
    await fs.mkdir(path.join(repoDir, 'crates', 'core', 'src'), {
      recursive: true,
    });
    await fs.writeFile(
      path.join(repoDir, 'Cargo.toml'),
      '[workspace]\nmembers = ["crates/core"]\n',
    );
    await fs.writeFile(
      path.join(repoDir, 'crates', 'core', 'Cargo.toml'),
      '[package]\nname = "acme-core"\nversion = "0.1.0"\n',
    );
    await fs.writeFile(
      path.join(repoDir, 'crates', 'core', 'src', 'lib.rs'),
      'pub fn parse() -> usize { 0 }\n',
    );

    const project = await detectProject(repoDir);
    assert.equal(project.primaryLanguage, 'rust');
    assert.ok(
      commandTimeoutMs('test', 'rust') > commandTimeoutMs('test', 'typescript'),
    );

    const heuristics = await detectSourceHeuristics(repoDir, project);
    assert.equal(heuristics.hasCargoWorkspace, true);

    const pkg = await rustPackage(repoDir, 'crates/core/src/lib.rs', 'parse_zero');
    assert.equal(pkg.packageName, 'acme-core');
    assert.equal(pkg.isWorkspace, true);

    const resolvedProbe = await resolveProbeFile(repoDir, project, {
      targetFile: 'crates/core/src/lib.rs',
      probeName: 'parse_zero',
    });
    assert.ok(
      resolvedProbe.endsWith(path.join('crates', 'core', 'tests', 'parse_zero.rs')),
    );

    const cmd = await buildProbeCommand(repoDir, project, resolvedProbe);
    assert.equal(
      cmd.command,
      'cargo test -p acme-core --test parse_zero -- --nocapture',
    );
  });
});

describe('Phase 4: Anti-False-Positive Filter & Static Reachability', () => {
  it('rejects mock stand-ins, remote 401/403 errors, and environmental failures while accepting real bugs', () => {
    // 1. Mock stand-in: redefines target function locally without importing target module
    const mockResult = assessProbeFailure({
      language: 'typescript',
      exitCode: 1,
      stdout: '',
      stderr: 'AssertionError: expected 0 to equal 3000',
      targetFile: 'src/config.ts',
      targetSymbol: 'resolvePort',
      probeContent: `
        function resolvePort(port?: number) { return port || 3000; }
        if (resolvePort(0) !== 0) throw new Error('failed');
      `,
    });
    assert.equal(mockResult.valid, false);
    assert.equal(mockResult.mockStandin, true);
    assert.equal(mockResult.category, 'mock_standin');

    // 2. Remote 401/403 failure
    const authEnv = isEnvironmentalProbeFailure(
      'HttpError: Request failed with status code 401 Unauthorized',
    );
    assert.equal(authEnv.isEnvironmental, true);
    assert.equal(authEnv.isRemoteAuth, true);

    const authResult = assessProbeFailure({
      language: 'typescript',
      exitCode: 1,
      stdout: '',
      stderr: 'HTTP 403 Forbidden: Bad credentials',
      targetFile: 'src/client.ts',
      targetSymbol: 'fetchIssues',
      probeContent: `import { fetchIssues } from '../src/client.js'; await fetchIssues();`,
    });
    assert.equal(authResult.valid, false);
    assert.equal(authResult.environmental, true);
    assert.equal(authResult.category, 'remote_auth_failure');

    // 3. Probe syntax/compile error
    const compileResult = assessProbeFailure({
      language: 'typescript',
      exitCode: 1,
      stdout: '',
      stderr: 'src/probe.ts(3,1): error TS2304: Cannot find name "foo".',
      targetFile: 'src/parser.ts',
      targetSymbol: 'parse',
      probeContent: `import { parse } from '../src/parser.js'; foo();`,
    });
    assert.equal(compileResult.valid, false);
    assert.equal(compileResult.category, 'probe_syntax_or_compile_error');

    // 4. Genuine bug reproduction calling real imported code
    const realResult = assessProbeFailure({
      language: 'typescript',
      exitCode: 1,
      stdout: '',
      stderr: 'AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n3000 !== 0',
      targetFile: 'src/config.ts',
      targetSymbol: 'resolvePort',
      probeContent: `
        import assert from 'node:assert/strict';
        import { resolvePort } from '../src/config.js';
        assert.equal(resolvePort(0), 0);
      `,
    });
    assert.equal(realResult.valid, true);
    assert.equal(realResult.category, 'genuine_failure');
  });

  it('distinguishes reachable production symbols from unreachable dead code', async () => {
    const files: Record<string, string> = {
      'src/index.ts': `
        import { handleRequest } from './router.js';
        export function start() { return handleRequest('/ping'); }
      `,
      'src/router.ts': `
        import { parseHeader } from './parser.js';
        export function handleRequest(path: string) {
          return parseHeader(path);
        }
      `,
      'src/parser.ts': `
        export function parseHeader(raw: string) {
          return raw.trim();
        }
        function unusedInternalHelper(x: string) {
          return x || 'default';
        }
      `,
      'tests/parser.test.ts': `
        // Test mentions unusedInternalHelper only in test file, should be ignored
        unusedInternalHelper('a');
      `,
    };

    const reachable = await assessReachability(
      '/tmp/fake-repo',
      'src/parser.ts',
      'parseHeader',
      { files },
    );
    assert.equal(reachable.reachable, true);
    assert.equal(reachable.isExported, true);
    assert.equal(reachable.callers.length, 1);
    assert.equal(reachable.callers[0]?.file, 'src/router.ts');

    const unreachable = await assessReachability(
      '/tmp/fake-repo',
      'src/parser.ts',
      'unusedInternalHelper',
      { files },
    );
    assert.equal(unreachable.reachable, false);
    assert.equal(unreachable.isExported, false);
    assert.equal(unreachable.callers.length, 0);
  });
});

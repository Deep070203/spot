import fs from 'node:fs/promises';
import path from 'node:path';
import {
  normalizeCommandWithVenv,
  resolveRepoEnv,
  type ResolvedRepoEnv,
} from './env.js';
import { isHuntSource } from './hunt-scope.js';

export type ProjectLanguage =
  | 'python'
  | 'typescript'
  | 'javascript'
  | 'go'
  | 'rust'
  | 'unknown';

export type PackageManager =
  | 'npm'
  | 'pnpm'
  | 'yarn'
  | 'bun'
  | 'uv'
  | 'poetry'
  | 'pip'
  | 'go'
  | 'cargo'
  | 'unknown';

export interface DetectedProject {
  repoDir: string;
  primaryLanguage: ProjectLanguage;
  languages: ProjectLanguage[];
  packageManager: PackageManager;
  manifests: string[];
  isEsm: boolean;
  hasTypeScript: boolean;
  testRunner: string;
  testCommand?: string;
  buildCommand?: string;
  typecheckCommand?: string;
}

export interface SourceHeuristics {
  language: ProjectLanguage;
  isEsm: boolean;
  requiresJsExtensionsInImports: boolean;
  hasPytest: boolean;
  hasVitest: boolean;
  hasJest: boolean;
  hasNodeTest: boolean;
  hasCargoWorkspace: boolean;
  goModulePath?: string;
  frameworks: string[];
  notes: string[];
}

export interface PreparedWorkspace {
  project: DetectedProject;
  env: ResolvedRepoEnv;
  heuristics: SourceHeuristics;
  ready: boolean;
  notes: string[];
}

export interface RustPackageInfo {
  packageName: string;
  manifestPath: string;
  crateDir: string;
  isWorkspace: boolean;
  testFilePath: string;
}

export interface ResolveProbeOptions {
  probeName?: string;
  targetFile?: string;
  language?: ProjectLanguage;
}

export interface BuildProbeCommandOptions {
  testName?: string;
  rustPackageName?: string;
  useModuleRun?: boolean;
}

export interface ProbeCommandSpec {
  command: string;
  cwd: string;
  language: ProjectLanguage;
  probeFilePath: string;
  timeoutMs: number;
}

export interface AssessProbeFailureInput {
  language: ProjectLanguage;
  exitCode: number | null;
  stdout: string;
  stderr: string;
  probeFilePath?: string;
  probeContent?: string;
  targetFile?: string;
  targetSymbol?: string;
  expectedErrorPattern?: RegExp | string;
}

export interface ProbeFailureAssessment {
  valid: boolean;
  category:
    | 'genuine_failure'
    | 'passed_unexpectedly'
    | 'mock_standin'
    | 'environmental_failure'
    | 'remote_auth_failure'
    | 'probe_syntax_or_compile_error'
    | 'insufficient_evidence';
  reason: string;
  environmental: boolean;
  mockStandin: boolean;
}

export interface CallerReference {
  file: string;
  line: number;
  snippet: string;
  isEntrypoint: boolean;
}

export interface ReachabilityAssessment {
  reachable: boolean;
  targetFile: string;
  targetSymbol: string;
  definedInTarget: boolean;
  isExported: boolean;
  isEntrypoint: boolean;
  callers: CallerReference[];
  reason: string;
}

export interface AssessReachabilityOptions {
  files?: Record<string, string>;
  candidateFiles?: string[];
  maxFiles?: number;
}

const EXTENSIONS_BY_LANGUAGE: Record<ProjectLanguage, string[]> = {
  python: ['.py'],
  typescript: ['.ts', '.tsx', '.mts', '.cts'],
  javascript: ['.js', '.jsx', '.mjs', '.cjs'],
  go: ['.go'],
  rust: ['.rs'],
  unknown: ['.ts', '.js', '.py', '.go', '.rs'],
};

const ENVIRONMENTAL_PATTERNS: Array<{ pattern: RegExp; isAuth: boolean; label: string }> = [
  {
    pattern: /\b(?:HTTP\s*)?(?:401|403)\b.*(?:Unauthorized|Forbidden|Bad credentials|rate limit|authentication)/i,
    isAuth: true,
    label: 'Remote HTTP 401/403 authentication or authorization error',
  },
  {
    pattern: /\b(?:status(?:Code)?[:= ]+(?:401|403)|Request failed with status code (?:401|403)|Unauthorized|Forbidden)\b/i,
    isAuth: true,
    label: 'Remote 401/403 access failure',
  },
  {
    pattern: /\b(?:ECONNREFUSED|ENOTFOUND|ETIMEDOUT|EAI_AGAIN|ECONNRESET|socket hang up|network timeout|Failed to fetch|ConnectionRefusedError)\b/i,
    isAuth: false,
    label: 'Network or remote service unavailability',
  },
  {
    pattern: /\b(?:No module named|ModuleNotFoundError|Cannot find module|ERR_MODULE_NOT_FOUND|command not found|ENOENT: no such file)\b/i,
    isAuth: false,
    label: 'Missing dependency, binary, or workspace module',
  },
  {
    pattern: /\b(?:Missing (?:API|auth|access)[ _-]?(?:key|token)|GITHUB_TOKEN|OPENAI_API_KEY|META_API_KEY).*(?:not set|missing|required|invalid)\b/i,
    isAuth: true,
    label: 'Missing environment credential',
  },
];

const PROBE_COMPILE_PATTERNS: RegExp[] = [
  /\bSyntaxError:/,
  /\bIndentationError:/,
  /\berror TS\d+:/,
  /\bcannot find package\b/i,
  /\berror\[E\d+\]:/,
];

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readTextIfExists(filePath: string): Promise<string | null> {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {
    return null;
  }
}

function sanitizeProbeSlug(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'spot_probe';
}

export function huntExtensions(
  projectOrLanguage: DetectedProject | ProjectLanguage,
): string[] {
  if (typeof projectOrLanguage === 'string') {
    return [...EXTENSIONS_BY_LANGUAGE[projectOrLanguage]];
  }
  const exts = new Set<string>();
  for (const lang of projectOrLanguage.languages) {
    for (const ext of EXTENSIONS_BY_LANGUAGE[lang] ?? []) {
      exts.add(ext);
    }
  }
  if (exts.size === 0) {
    return [...EXTENSIONS_BY_LANGUAGE.unknown];
  }
  return [...exts];
}

export function commandTimeoutMs(
  kind: 'probe' | 'test' | 'build' | 'install',
  language: ProjectLanguage = 'unknown',
): number {
  const baseByKind: Record<typeof kind, number> = {
    probe: 30_000,
    test: 120_000,
    build: 180_000,
    install: 240_000,
  };
  let timeout = baseByKind[kind];
  if (language === 'rust') {
    timeout = Math.round(timeout * 2);
  } else if (language === 'go' && (kind === 'build' || kind === 'test')) {
    timeout = Math.round(timeout * 1.5);
  }
  return timeout;
}

export async function detectProject(repoDir: string): Promise<DetectedProject> {
  const resolved = path.resolve(repoDir);
  const manifests: string[] = [];
  const languages: ProjectLanguage[] = [];

  const pkgJsonRaw = await readTextIfExists(path.join(resolved, 'package.json'));
  const hasTsConfig = await fileExists(path.join(resolved, 'tsconfig.json'));
  const pyprojectRaw = await readTextIfExists(path.join(resolved, 'pyproject.toml'));
  const hasSetupPy = await fileExists(path.join(resolved, 'setup.py'));
  const hasRequirements = await fileExists(path.join(resolved, 'requirements.txt'));
  const goModRaw = await readTextIfExists(path.join(resolved, 'go.mod'));
  const cargoTomlRaw = await readTextIfExists(path.join(resolved, 'Cargo.toml'));

  let isEsm = false;
  let hasTypeScript = hasTsConfig;
  let packageManager: PackageManager = 'unknown';
  let testRunner = 'unknown';
  let testCommand: string | undefined;
  let buildCommand: string | undefined;
  let typecheckCommand: string | undefined;

  if (pkgJsonRaw !== null) {
    manifests.push('package.json');
    try {
      const pkg = JSON.parse(pkgJsonRaw) as {
        type?: string;
        scripts?: Record<string, string>;
        dependencies?: Record<string, string>;
        devDependencies?: Record<string, string>;
      };
      isEsm = pkg.type === 'module';
      const allDeps = {
        ...(pkg.dependencies ?? {}),
        ...(pkg.devDependencies ?? {}),
      };
      if (allDeps.typescript || hasTsConfig) {
        hasTypeScript = true;
      }
      if (await fileExists(path.join(resolved, 'pnpm-lock.yaml'))) {
        packageManager = 'pnpm';
      } else if (await fileExists(path.join(resolved, 'yarn.lock'))) {
        packageManager = 'yarn';
      } else if (await fileExists(path.join(resolved, 'bun.lockb'))) {
        packageManager = 'bun';
      } else {
        packageManager = 'npm';
      }

      if (hasTypeScript) {
        languages.push('typescript');
      } else {
        languages.push('javascript');
      }

      const scripts = pkg.scripts ?? {};
      if (scripts.test) testCommand = `${packageManager} test`;
      if (scripts.build) buildCommand = `${packageManager} run build`;
      if (scripts.typecheck) {
        typecheckCommand = `${packageManager} run typecheck`;
      } else if (hasTypeScript) {
        typecheckCommand = 'npx tsc --noEmit';
      }

      if (allDeps.vitest || scripts.test?.includes('vitest')) {
        testRunner = 'vitest';
      } else if (allDeps.jest || scripts.test?.includes('jest')) {
        testRunner = 'jest';
      } else if (allDeps.tsx || scripts.test?.includes('tsx')) {
        testRunner = 'tsx';
      } else {
        testRunner = 'node:test';
      }
    } catch {
      languages.push(hasTsConfig ? 'typescript' : 'javascript');
    }
  } else if (hasTsConfig) {
    manifests.push('tsconfig.json');
    languages.push('typescript');
    hasTypeScript = true;
    packageManager = 'npm';
    testRunner = 'node:test';
    typecheckCommand = 'npx tsc --noEmit';
  }

  if (pyprojectRaw !== null || hasSetupPy || hasRequirements) {
    if (pyprojectRaw !== null) manifests.push('pyproject.toml');
    if (hasSetupPy) manifests.push('setup.py');
    if (hasRequirements) manifests.push('requirements.txt');
    languages.push('python');

    if (packageManager === 'unknown') {
      if (await fileExists(path.join(resolved, 'uv.lock'))) {
        packageManager = 'uv';
      } else if (await fileExists(path.join(resolved, 'poetry.lock'))) {
        packageManager = 'poetry';
      } else {
        packageManager = 'pip';
      }
    }

    if (testRunner === 'unknown') {
      const usesPytest =
        pyprojectRaw?.includes('pytest') ||
        (await fileExists(path.join(resolved, 'pytest.ini'))) ||
        (await fileExists(path.join(resolved, 'conftest.py')));
      testRunner = usesPytest ? 'pytest' : 'unittest';
      testCommand = testCommand ?? (usesPytest ? 'pytest' : 'python -m unittest');
    }
  }

  if (goModRaw !== null) {
    manifests.push('go.mod');
    languages.push('go');
    if (packageManager === 'unknown') packageManager = 'go';
    if (testRunner === 'unknown') {
      testRunner = 'go test';
      testCommand = 'go test ./...';
      buildCommand = 'go build ./...';
    }
  }

  if (cargoTomlRaw !== null) {
    manifests.push('Cargo.toml');
    languages.push('rust');
    if (packageManager === 'unknown') packageManager = 'cargo';
    if (testRunner === 'unknown') {
      testRunner = 'cargo test';
      testCommand = 'cargo test';
      buildCommand = 'cargo build';
    }
  }

  const primaryLanguage = languages[0] ?? 'unknown';

  return {
    repoDir: resolved,
    primaryLanguage,
    languages,
    packageManager,
    manifests,
    isEsm,
    hasTypeScript,
    testRunner,
    testCommand,
    buildCommand,
    typecheckCommand,
  };
}

export async function typescriptProbeFile(
  repoDir: string,
  probeName = 'spot_probe',
): Promise<string> {
  const resolved = path.resolve(repoDir);
  const slug = sanitizeProbeSlug(probeName);
  const pkgRaw = await readTextIfExists(path.join(resolved, 'package.json'));
  let isEsm = true;
  if (pkgRaw) {
    try {
      const pkg = JSON.parse(pkgRaw) as { type?: string };
      isEsm = pkg.type === 'module';
    } catch {
      isEsm = true;
    }
  }

  const ext = isEsm ? '.mts' : '.ts';
  const testsDir = path.join(resolved, 'tests');
  const testDir = path.join(resolved, 'test');

  if (await fileExists(testsDir)) {
    return path.join(testsDir, `${slug}.probe${ext}`);
  }
  if (await fileExists(testDir)) {
    return path.join(testDir, `${slug}.probe${ext}`);
  }
  return path.join(resolved, `${slug}.probe${ext}`);
}

export async function goProbeFile(
  repoDir: string,
  targetFile?: string,
  probeName = 'spot_probe',
): Promise<{ filePath: string; packageName: string }> {
  const resolved = path.resolve(repoDir);
  const slug = sanitizeProbeSlug(probeName);
  const targetDir = targetFile
    ? path.dirname(path.resolve(resolved, targetFile))
    : resolved;

  let packageName = path.basename(targetDir) || 'main';
  try {
    const entries = await fs.readdir(targetDir);
    for (const entry of entries) {
      if (!entry.endsWith('.go')) continue;
      const content = await readTextIfExists(path.join(targetDir, entry));
      const pkgMatch = content?.match(/^\s*package\s+([A-Za-z_]\w*)/m);
      if (pkgMatch?.[1]) {
        packageName = pkgMatch[1].replace(/_test$/, '');
        break;
      }
    }
  } catch {
    // Directory might not exist yet; keep inferred package name.
  }

  return {
    filePath: path.join(targetDir, `${slug}_test.go`),
    packageName,
  };
}

export async function rustPackage(
  repoDir: string,
  targetFile?: string,
  probeName = 'spot_probe',
): Promise<RustPackageInfo> {
  const resolved = path.resolve(repoDir);
  const slug = sanitizeProbeSlug(probeName);
  const rootCargoPath = path.join(resolved, 'Cargo.toml');
  const rootCargo = (await readTextIfExists(rootCargoPath)) ?? '';
  const isWorkspace = /\[workspace\]/.test(rootCargo);

  let selectedManifest = rootCargoPath;
  let selectedDir = resolved;

  if (targetFile) {
    let cursor = path.dirname(path.resolve(resolved, targetFile));
    while (cursor.startsWith(resolved)) {
      const candidate = path.join(cursor, 'Cargo.toml');
      const content = await readTextIfExists(candidate);
      if (content && /\[package\]/.test(content)) {
        selectedManifest = candidate;
        selectedDir = cursor;
        break;
      }
      if (cursor === resolved) break;
      cursor = path.dirname(cursor);
    }
  }

  const manifestText =
    selectedManifest === rootCargoPath
      ? rootCargo
      : (await readTextIfExists(selectedManifest)) ?? rootCargo;
  const nameMatch = manifestText.match(
    /\[package\][\s\S]*?\bname\s*=\s*["']([^"']+)["']/,
  );
  const packageName = nameMatch?.[1] ?? path.basename(selectedDir);

  return {
    packageName,
    manifestPath: selectedManifest,
    crateDir: selectedDir,
    isWorkspace,
    testFilePath: path.join(selectedDir, 'tests', `${slug}.rs`),
  };
}

export async function detectSourceHeuristics(
  repoDir: string,
  project?: DetectedProject,
): Promise<SourceHeuristics> {
  const detected = project ?? (await detectProject(repoDir));
  const pkgRaw = await readTextIfExists(path.join(detected.repoDir, 'package.json'));
  const tsconfigRaw = await readTextIfExists(path.join(detected.repoDir, 'tsconfig.json'));
  const pyprojectRaw = await readTextIfExists(path.join(detected.repoDir, 'pyproject.toml'));
  const goModRaw = await readTextIfExists(path.join(detected.repoDir, 'go.mod'));
  const cargoRaw = await readTextIfExists(path.join(detected.repoDir, 'Cargo.toml'));

  const requiresJsExtensionsInImports = Boolean(
    detected.isEsm &&
      tsconfigRaw &&
      /NodeNext|Node16/i.test(tsconfigRaw),
  );

  const hasPytest = Boolean(
    detected.testRunner === 'pytest' || pyprojectRaw?.includes('pytest'),
  );
  const hasVitest = Boolean(pkgRaw?.includes('vitest'));
  const hasJest = Boolean(pkgRaw?.includes('jest'));
  const hasNodeTest = detected.testRunner === 'node:test';
  const hasCargoWorkspace = Boolean(cargoRaw && /\[workspace\]/.test(cargoRaw));

  const goModMatch = goModRaw?.match(/^\s*module\s+(\S+)/m);
  const goModulePath = goModMatch?.[1];

  const frameworks: string[] = [];
  const combinedManifests = [pkgRaw, pyprojectRaw, goModRaw, cargoRaw]
    .filter(Boolean)
    .join('\n');
  for (const fw of [
    'express',
    'fastify',
    'next',
    'fastapi',
    'django',
    'flask',
    'gin',
    'tokio',
    'axum',
  ]) {
    if (new RegExp(`\\b${fw}\\b`, 'i').test(combinedManifests)) {
      frameworks.push(fw);
    }
  }

  const notes: string[] = [];
  if (requiresJsExtensionsInImports) {
    notes.push('TypeScript uses ESM NodeNext: relative imports must end in .js/.mjs.');
  }
  if (detected.primaryLanguage === 'python') {
    notes.push('Python probes must run with PYTHONSAFEPATH=1 and python -P to avoid stdlib shadowing.');
  }
  if (hasCargoWorkspace) {
    notes.push('Rust Cargo workspace detected: scope cargo test using -p <package>.');
  }

  return {
    language: detected.primaryLanguage,
    isEsm: detected.isEsm,
    requiresJsExtensionsInImports,
    hasPytest,
    hasVitest,
    hasJest,
    hasNodeTest,
    hasCargoWorkspace,
    goModulePath,
    frameworks,
    notes,
  };
}

export function auditNote(
  project: DetectedProject,
  heuristics?: SourceHeuristics,
): string {
  const lines = [
    `Project Language: ${project.primaryLanguage} (all: ${project.languages.join(', ') || 'unknown'})`,
    `Package Manager: ${project.packageManager} | Test Runner: ${project.testRunner}`,
  ];
  if (project.typecheckCommand) {
    lines.push(`Typecheck: ${project.typecheckCommand}`);
  }
  if (project.testCommand) {
    lines.push(`Test Command: ${project.testCommand}`);
  }
  if (heuristics && heuristics.notes.length > 0) {
    lines.push(`Heuristics: ${heuristics.notes.join(' ')}`);
  }
  lines.push(
    'Proof-First Rule: Every claimed bug must be backed by a concrete failing probe calling real repository symbols (no mock stand-ins).',
  );
  return lines.join('\n');
}

export function probeInstructions(
  project: DetectedProject,
  targetFile?: string,
  heuristics?: SourceHeuristics,
): string {
  const targetNote = targetFile ? `targeting "${targetFile}"` : 'targeting the real repository module';
  const common = [
    `Write a minimal, self-contained reproduction probe ${targetNote}.`,
    'Import and execute the actual production function/class from the repository.',
    'Do NOT reimplement the buggy function inline and do NOT mock away the unit under test.',
    'Assert the expected invariant so the probe exits non-zero (fails) on the bug and exits 0 after the fix.',
  ];

  switch (project.primaryLanguage) {
    case 'python':
      common.push(
        'Use standard library `assert` or `unittest`/`pytest` without relying on CWD imports (`python -P`).',
      );
      break;
    case 'typescript':
      if (heuristics?.requiresJsExtensionsInImports) {
        common.push('Use `.js` extensions on relative imports to satisfy TypeScript NodeNext resolution.');
      }
      common.push('Use `node:assert/strict` (or the repo test runner) and throw on invariant violation.');
      break;
    case 'go':
      common.push('Write a standard `TestSpotProbe*(t *testing.T)` function in the target package (`*_test.go`).');
      break;
    case 'rust':
      common.push('Write an integration test (`#[test] fn spot_probe_*()`) under `tests/` importing the crate.');
      break;
    default:
      break;
  }

  return common.join('\n');
}

export async function prepareToolchainWorkspace(
  repoDir: string,
  project?: DetectedProject,
): Promise<PreparedWorkspace> {
  const resolvedProject = project ?? (await detectProject(repoDir));
  const [env, heuristics] = await Promise.all([
    resolveRepoEnv(resolvedProject.repoDir),
    detectSourceHeuristics(resolvedProject.repoDir, resolvedProject),
  ]);

  const notes = [...heuristics.notes];
  let ready = resolvedProject.primaryLanguage !== 'unknown';
  if (!ready) {
    notes.push('No recognized language manifest found; falling back to generic heuristics.');
  }

  return {
    project: resolvedProject,
    env,
    heuristics,
    ready,
    notes,
  };
}

export async function resolveProbeFile(
  repoDir: string,
  project: DetectedProject,
  options: ResolveProbeOptions = {},
): Promise<string> {
  const language = options.language ?? project.primaryLanguage;
  const slug = sanitizeProbeSlug(options.probeName ?? 'spot_probe');

  switch (language) {
    case 'typescript':
      return typescriptProbeFile(repoDir, slug);
    case 'javascript': {
      const ext = project.isEsm ? '.mjs' : '.js';
      const testsDir = path.join(path.resolve(repoDir), 'tests');
      return (await fileExists(testsDir))
        ? path.join(testsDir, `${slug}.probe${ext}`)
        : path.join(path.resolve(repoDir), `${slug}.probe${ext}`);
    }
    case 'go': {
      const goProbe = await goProbeFile(repoDir, options.targetFile, slug);
      return goProbe.filePath;
    }
    case 'rust': {
      const rustPkg = await rustPackage(repoDir, options.targetFile, slug);
      return rustPkg.testFilePath;
    }
    case 'python':
    default: {
      const testsDir = path.join(path.resolve(repoDir), 'tests');
      return (await fileExists(testsDir))
        ? path.join(testsDir, `test_${slug}.py`)
        : path.join(path.resolve(repoDir), `${slug}_probe.py`);
    }
  }
}

export async function buildProbeCommand(
  repoDir: string,
  project: DetectedProject,
  probeFilePath: string,
  options: BuildProbeCommandOptions = {},
): Promise<ProbeCommandSpec> {
  const resolvedRepo = path.resolve(repoDir);
  const resolvedProbe = path.resolve(resolvedRepo, probeFilePath);
  const relProbe = path.relative(resolvedRepo, resolvedProbe) || path.basename(resolvedProbe);
  const env = await resolveRepoEnv(resolvedRepo);
  const ext = path.extname(resolvedProbe).toLowerCase();

  let language: ProjectLanguage = project.primaryLanguage;
  if (ext === '.py') language = 'python';
  else if (['.ts', '.tsx', '.mts', '.cts'].includes(ext)) language = 'typescript';
  else if (['.js', '.jsx', '.mjs', '.cjs'].includes(ext)) language = 'javascript';
  else if (ext === '.go') language = 'go';
  else if (ext === '.rs') language = 'rust';

  let command: string;

  switch (language) {
    case 'python': {
      const rawCmd = options.useModuleRun || project.testRunner === 'pytest'
        ? `pytest ${relProbe}`
        : `python3 ${relProbe}`;
      command = normalizeCommandWithVenv(rawCmd, env);
      break;
    }
    case 'typescript': {
      const localTsx = path.join(resolvedRepo, 'node_modules', '.bin', 'tsx');
      if (await fileExists(localTsx)) {
        command = `${localTsx} ${relProbe}`;
      } else {
        command = `npx tsx ${relProbe}`;
      }
      break;
    }
    case 'javascript': {
      command = `node ${relProbe}`;
      break;
    }
    case 'go': {
      const relDir = path.posix.dirname(relProbe.replace(/\\/g, '/'));
      const pkgTarget = relDir === '.' ? '.' : `./${relDir}`;
      const runFlag = options.testName ? `-run ${options.testName} ` : '';
      command = `go test -v ${runFlag}${pkgTarget}`;
      break;
    }
    case 'rust': {
      const rustPkg = await rustPackage(resolvedRepo, resolvedProbe);
      const pkgName = options.rustPackageName ?? rustPkg.packageName;
      const testStem = path.basename(resolvedProbe, '.rs');
      const pkgFlag = rustPkg.isWorkspace || options.rustPackageName ? `-p ${pkgName} ` : '';
      command = `cargo test ${pkgFlag}--test ${testStem} -- --nocapture`;
      break;
    }
    default: {
      command = `node ${relProbe}`;
      break;
    }
  }

  return {
    command,
    cwd: resolvedRepo,
    language,
    probeFilePath: resolvedProbe,
    timeoutMs: commandTimeoutMs('probe', language),
  };
}

export function isEnvironmentalProbeFailure(output: string): {
  isEnvironmental: boolean;
  isRemoteAuth: boolean;
  reason?: string;
} {
  for (const entry of ENVIRONMENTAL_PATTERNS) {
    if (entry.pattern.test(output)) {
      return {
        isEnvironmental: true,
        isRemoteAuth: entry.isAuth,
        reason: entry.label,
      };
    }
  }
  return { isEnvironmental: false, isRemoteAuth: false };
}

function detectMockStandin(
  probeContent: string,
  targetFile?: string,
  targetSymbol?: string,
): { isMockStandin: boolean; reason?: string } {
  const hasRealImport = (() => {
    const importLines = probeContent
      .split(/\r?\n/)
      .filter((line) => /^\s*(?:import|from|require\(|use\s+)/.test(line));

    if (targetFile) {
      const stem = path
        .basename(targetFile, path.extname(targetFile))
        .replace(/[^a-zA-Z0-9_-]/g, '');
      if (
        stem &&
        importLines.some((line) => line.includes(stem))
      ) {
        return true;
      }
      if (
        probeContent.includes(targetFile) ||
        (stem && new RegExp(`["'\`./]${stem}["'\`./]`).test(probeContent))
      ) {
        return true;
      }
    }

    // Check Go package / crate usage if no explicit import line matched stem.
    return importLines.some(
      (line) =>
        /\.\.?\//.test(line) ||
        /\bsrc\b/.test(line) ||
        /\bcrate::/.test(line),
    );
  })();

  // If the probe redefines the target symbol locally instead of importing it, reject as a stand-in.
  if (targetSymbol) {
    const escaped = targetSymbol.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const localRedefinition = new RegExp(
      `(?:^|\\n)\\s*(?:export\\s+)?(?:async\\s+)?(?:function|def|func|fn|class|const)\\s+${escaped}\\b`,
    ).test(probeContent);

    if (localRedefinition && !hasRealImport) {
      return {
        isMockStandin: true,
        reason: `Probe redefines "${targetSymbol}" locally instead of importing it from "${targetFile ?? 'repository'}".`,
      };
    }

    const mocksTargetSymbol = new RegExp(
      `(?:jest\\.fn|vi\\.fn|mock\\.patch|MagicMock|sinon\\.stub)\\([^\\n]*${escaped}`,
    ).test(probeContent);
    if (mocksTargetSymbol) {
      return {
        isMockStandin: true,
        reason: `Probe mocks out target symbol "${targetSymbol}" instead of exercising real code.`,
      };
    }
  }

  // Generic stand-in heuristics: stub/fake/mock stand-in comments or local mock implementation without importing targetFile.
  if (targetFile && !hasRealImport) {
    if (
      /\b(?:mock|stub|fake|stand-?in|reimplement)\b/i.test(probeContent) ||
      /(?:function|def|func|fn)\s+\w+/.test(probeContent)
    ) {
      return {
        isMockStandin: true,
        reason: `Probe does not import target module "${targetFile}" and appears to use a local stand-in.`,
      };
    }
  }

  return { isMockStandin: false };
}

export function assessProbeFailure(
  input: AssessProbeFailureInput,
): ProbeFailureAssessment {
  const combinedOutput = `${input.stdout}\n${input.stderr}`.trim();

  if (input.probeContent) {
    const mockCheck = detectMockStandin(
      input.probeContent,
      input.targetFile,
      input.targetSymbol,
    );
    if (mockCheck.isMockStandin) {
      return {
        valid: false,
        category: 'mock_standin',
        reason: mockCheck.reason ?? 'Probe uses a mock stand-in instead of real repository code.',
        environmental: false,
        mockStandin: true,
      };
    }
  }

  const envCheck = isEnvironmentalProbeFailure(combinedOutput);
  if (envCheck.isEnvironmental) {
    return {
      valid: false,
      category: envCheck.isRemoteAuth
        ? 'remote_auth_failure'
        : 'environmental_failure',
      reason: envCheck.reason ?? 'Environmental failure during probe execution.',
      environmental: true,
      mockStandin: false,
    };
  }

  if (input.exitCode === 0) {
    return {
      valid: false,
      category: 'passed_unexpectedly',
      reason: 'Probe exited with code 0 and failed to reproduce the bug.',
      environmental: false,
      mockStandin: false,
    };
  }

  for (const compileRe of PROBE_COMPILE_PATTERNS) {
    if (compileRe.test(combinedOutput)) {
      return {
        valid: false,
        category: 'probe_syntax_or_compile_error',
        reason: 'Probe failed to compile or parse rather than reproducing a runtime bug.',
        environmental: false,
        mockStandin: false,
      };
    }
  }

  if (!combinedOutput) {
    return {
      valid: false,
      category: 'insufficient_evidence',
      reason: 'Probe exited non-zero without producing any stdout/stderr trace.',
      environmental: false,
      mockStandin: false,
    };
  }

  if (input.expectedErrorPattern) {
    const matches =
      typeof input.expectedErrorPattern === 'string'
        ? combinedOutput.includes(input.expectedErrorPattern)
        : input.expectedErrorPattern.test(combinedOutput);
    if (!matches) {
      return {
        valid: false,
        category: 'insufficient_evidence',
        reason: 'Probe output did not match the expected failure signature.',
        environmental: false,
        mockStandin: false,
      };
    }
  }

  return {
    valid: true,
    category: 'genuine_failure',
    reason: 'Probe reproduced a genuine non-environmental failure against real repository code.',
    environmental: false,
    mockStandin: false,
  };
}

function isEntrypointPath(filePath: string): boolean {
  const normalized = filePath.replace(/\\/g, '/').toLowerCase();
  const base = path.posix.basename(normalized);
  return (
    /^(?:index|main|app|server|cli|__main__|mod|lib)\.[^.]+$/.test(base) ||
    normalized.includes('/routes/') ||
    normalized.includes('/handlers/') ||
    normalized.includes('/cmd/') ||
    normalized.includes('/bin/')
  );
}

async function collectRepoSourceFiles(
  repoDir: string,
  maxFiles: number,
): Promise<string[]> {
  const results: string[] = [];

  async function walk(dir: string): Promise<void> {
    if (results.length >= maxFiles) return;
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      if (results.length >= maxFiles) break;
      const fullPath = path.join(dir, entry.name);
      const rel = path.relative(repoDir, fullPath).replace(/\\/g, '/');
      if (entry.isDirectory()) {
        if (
          entry.name.startsWith('.') ||
          ['node_modules', 'dist', 'build', 'target', 'vendor', '__pycache__', '.venv', 'venv', 'tests', 'test', '__tests__'].includes(
            entry.name,
          )
        ) {
          continue;
        }
        await walk(fullPath);
      } else if (entry.isFile() && isHuntSource(rel)) {
        results.push(rel);
      }
    }
  }

  await walk(repoDir);
  return results;
}

export async function assessReachability(
  repoDir: string,
  targetFile: string,
  targetSymbol: string,
  options: AssessReachabilityOptions = {},
): Promise<ReachabilityAssessment> {
  const resolvedRepo = path.resolve(repoDir);
  const normalizedTarget = targetFile.replace(/\\/g, '/').replace(/^\.\//, '');
  const isTargetEntrypoint = isEntrypointPath(normalizedTarget);

  const fileCache = new Map<string, string>();
  if (options.files) {
    for (const [k, v] of Object.entries(options.files)) {
      fileCache.set(k.replace(/\\/g, '/').replace(/^\.\//, ''), v);
    }
  }

  const getContent = async (relPath: string): Promise<string | null> => {
    const norm = relPath.replace(/\\/g, '/').replace(/^\.\//, '');
    if (fileCache.has(norm)) {
      return fileCache.get(norm) ?? null;
    }
    const content = await readTextIfExists(path.join(resolvedRepo, norm));
    if (content !== null) {
      fileCache.set(norm, content);
    }
    return content;
  };

  const targetContent = await getContent(normalizedTarget);
  if (targetContent === null) {
    return {
      reachable: false,
      targetFile: normalizedTarget,
      targetSymbol,
      definedInTarget: false,
      isExported: false,
      isEntrypoint: isTargetEntrypoint,
      callers: [],
      reason: `Target file "${normalizedTarget}" could not be read.`,
    };
  }

  const escapedSym = targetSymbol.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const defRe = new RegExp(
    `(?:export\\s+(?:default\\s+)?)?(?:async\\s+)?(?:function\\s+${escapedSym}\\b|class\\s+${escapedSym}\\b|(?:const|let|var)\\s+${escapedSym}\\s*=|def\\s+${escapedSym}\\s*\\(|func\\s+(?:\\([^)]*\\)\\s*)?${escapedSym}\\s*\\(|(?:pub\\s+)?(?:async\\s+)?fn\\s+${escapedSym}\\b)`,
  );
  const defLineIdx = targetContent
    .split(/\r?\n/)
    .findIndex((line) => defRe.test(line));
  const definedInTarget = defLineIdx !== -1 || new RegExp(`\\b${escapedSym}\\b`).test(targetContent);

  const isExported =
    new RegExp(
      `export\\s+(?:default\\s+)?(?:async\\s+)?(?:function|class|const|let|var)?\\s*${escapedSym}\\b|export\\s*\\{[^}]*\\b${escapedSym}\\b`,
    ).test(targetContent) ||
    new RegExp(`\\bpub\\s+(?:async\\s+)?fn\\s+${escapedSym}\\b`).test(targetContent) ||
    /^[A-Z]/.test(targetSymbol) && normalizedTarget.endsWith('.go') && definedInTarget ||
    (normalizedTarget.endsWith('.py') && !targetSymbol.startsWith('_') && definedInTarget);

  const candidateFiles =
    options.candidateFiles ??
    (options.files
      ? Object.keys(options.files).filter((f) => isHuntSource(f))
      : await collectRepoSourceFiles(resolvedRepo, options.maxFiles ?? 300));

  const callRe = new RegExp(`\\b${escapedSym}\\s*\\(`);
  const callers: CallerReference[] = [];

  for (const relFile of candidateFiles) {
    const normFile = relFile.replace(/\\/g, '/').replace(/^\.\//, '');
    if (!isHuntSource(normFile)) continue;

    const content = await getContent(normFile);
    if (!content) continue;

    const lines = content.split(/\r?\n/);
    for (let i = 0; i < lines.length; i += 1) {
      const lineText = lines[i]!;
      const trimmed = lineText.trim();
      if (
        !trimmed ||
        trimmed.startsWith('//') ||
        trimmed.startsWith('#') ||
        trimmed.startsWith('*')
      ) {
        continue;
      }
      if (normFile === normalizedTarget && i === defLineIdx) {
        continue;
      }
      if (defRe.test(lineText)) {
        continue;
      }
      if (callRe.test(lineText)) {
        callers.push({
          file: normFile,
          line: i + 1,
          snippet: trimmed,
          isEntrypoint: isEntrypointPath(normFile),
        });
      }
    }
  }

  const hasEntrypointCaller = callers.some((c) => c.isEntrypoint);
  const reachable =
    definedInTarget &&
    (callers.length > 0 || isTargetEntrypoint || isExported);

  let reason: string;
  if (!definedInTarget) {
    reason = `Symbol "${targetSymbol}" was not found in "${normalizedTarget}".`;
  } else if (callers.length > 0) {
    reason = `Found ${callers.length} static caller(s)${hasEntrypointCaller ? ' including entrypoint caller(s)' : ''}.`;
  } else if (isTargetEntrypoint) {
    reason = `Symbol is defined inside entrypoint "${normalizedTarget}".`;
  } else if (isExported) {
    reason = `Symbol "${targetSymbol}" is part of the exported public surface of "${normalizedTarget}".`;
  } else {
    reason = `Unexported symbol "${targetSymbol}" in "${normalizedTarget}" has zero static callers in production source.`;
  }

  return {
    reachable,
    targetFile: normalizedTarget,
    targetSymbol,
    definedInTarget,
    isExported,
    isEntrypoint: isTargetEntrypoint || hasEntrypointCaller,
    callers,
    reason,
  };
}

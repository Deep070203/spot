import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export interface ResolvedRepoEnv {
  repoDir: string;
  env: Record<string, string>;
  venvDir: string | null;
  venvBinDir: string | null;
  pythonBinary: string | null;
  pythonRoots: string[];
}

export interface ResolveRepoEnvOptions {
  baseEnv?: NodeJS.ProcessEnv;
  extraEnv?: Record<string, string>;
  includePythonRoots?: boolean;
  maxDepth?: number;
}

const IGNORED_DIRS = new Set([
  '.git',
  '.hg',
  '.svn',
  'node_modules',
  '.venv',
  'venv',
  '.env',
  'env',
  '__pycache__',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
  '.tox',
  '.nox',
  'dist',
  'build',
  'out',
  'target',
  'coverage',
]);

const NON_PACKAGE_TOP_DIRS = new Set([
  'tests',
  'test',
  '__tests__',
  'docs',
  'doc',
  'examples',
  'example',
  'scripts',
  'benchmarks',
  'fixtures',
]);

const PROJECT_MARKERS = ['pyproject.toml', 'setup.py', 'setup.cfg'];
const VENV_CANDIDATES = ['.venv', 'venv', '.env', 'env'];

async function pathExists(targetPath: string): Promise<boolean> {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function isDirectory(targetPath: string): Promise<boolean> {
  try {
    const stat = await fs.stat(targetPath);
    return stat.isDirectory();
  } catch {
    return false;
  }
}

async function hasAnyMarker(dir: string, markers: string[]): Promise<boolean> {
  for (const marker of markers) {
    if (await pathExists(path.join(dir, marker))) {
      return true;
    }
  }
  return false;
}

async function containsPythonFiles(
  dir: string,
  depth = 0,
  maxDepth = 4,
): Promise<boolean> {
  if (depth > maxDepth) {
    return false;
  }

  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return false;
  }

  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith('.py')) {
      return true;
    }
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (IGNORED_DIRS.has(entry.name) || entry.name.startsWith('.')) continue;
    if (await containsPythonFiles(path.join(dir, entry.name), depth + 1, maxDepth)) {
      return true;
    }
  }

  return false;
}

async function hasImmediateOrNamespacePackage(dir: string): Promise<boolean> {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return false;
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (
      IGNORED_DIRS.has(entry.name) ||
      NON_PACKAGE_TOP_DIRS.has(entry.name) ||
      entry.name.startsWith('.') ||
      entry.name === 'src'
    ) {
      continue;
    }

    const childDir = path.join(dir, entry.name);
    if (await hasAnyMarker(childDir, PROJECT_MARKERS)) {
      // Subproject boundary is inspected separately, not as a package under parent.
      continue;
    }

    if (await pathExists(path.join(childDir, '__init__.py'))) {
      return true;
    }

    // PEP 420 namespace package: directory without __init__.py containing Python modules/subpackages.
    if (await containsPythonFiles(childDir, 0, 3)) {
      return true;
    }
  }

  return false;
}

async function collectProjectRoots(
  dir: string,
  depth: number,
  maxDepth: number,
  out: Set<string>,
): Promise<void> {
  if (depth > maxDepth) {
    return;
  }

  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }

  // Never descend into regular Python packages (package interiors).
  if (depth > 0 && (await pathExists(path.join(dir, '__init__.py')))) {
    return;
  }

  if (depth === 0 || (await hasAnyMarker(dir, PROJECT_MARKERS))) {
    out.add(dir);
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (IGNORED_DIRS.has(entry.name) || entry.name.startsWith('.')) continue;
    if (entry.name === 'src') continue;
    const childDir = path.join(dir, entry.name);
    await collectProjectRoots(childDir, depth + 1, maxDepth, out);
  }
}

/**
 * Discovers Python import roots (PYTHONPATH entries) for a repository.
 * Supports `src/` layouts, regular packages, monorepo subprojects, and PEP 420
 * namespace packages while never adding package interiors (preventing stdlib shadowing).
 */
export async function discoverPythonPackageRoots(
  repoDir: string,
  options: { maxDepth?: number } = {},
): Promise<string[]> {
  const resolvedRepo = path.resolve(repoDir);
  const maxDepth = options.maxDepth ?? 4;
  const projectRoots = new Set<string>();

  await collectProjectRoots(resolvedRepo, 0, maxDepth, projectRoots);

  const importRoots = new Set<string>();

  for (const projRoot of projectRoots) {
    const srcDir = path.join(projRoot, 'src');
    if (
      (await isDirectory(srcDir)) &&
      (await containsPythonFiles(srcDir, 0, 4))
    ) {
      importRoots.add(srcDir);
    }

    if (await hasImmediateOrNamespacePackage(projRoot)) {
      importRoots.add(projRoot);
    }
  }

  return [...importRoots];
}

export async function findRepoVenv(
  repoDir: string,
): Promise<{ venvDir: string; venvBinDir: string; pythonBinary: string | null } | null> {
  const resolvedRepo = path.resolve(repoDir);

  for (const candidate of VENV_CANDIDATES) {
    const venvDir = path.join(resolvedRepo, candidate);
    if (!(await isDirectory(venvDir))) continue;

    for (const binName of ['bin', 'Scripts']) {
      const binDir = path.join(venvDir, binName);
      if (!(await isDirectory(binDir))) continue;

      for (const pyName of ['python3', 'python', 'python.exe']) {
        const pyPath = path.join(binDir, pyName);
        if (await pathExists(pyPath)) {
          return { venvDir, venvBinDir: binDir, pythonBinary: pyPath };
        }
      }

      return { venvDir, venvBinDir: binDir, pythonBinary: null };
    }
  }

  return null;
}

export async function resolveRepoEnv(
  repoDir: string,
  options: ResolveRepoEnvOptions = {},
): Promise<ResolvedRepoEnv> {
  const resolvedRepo = path.resolve(repoDir);
  const baseEnv = options.baseEnv ?? process.env;
  const env: Record<string, string> = {};

  for (const [key, value] of Object.entries(baseEnv)) {
    if (typeof value === 'string') {
      env[key] = value;
    }
  }

  const venvInfo = await findRepoVenv(resolvedRepo);
  const pythonRoots =
    options.includePythonRoots === false
      ? []
      : await discoverPythonPackageRoots(resolvedRepo, {
          maxDepth: options.maxDepth,
        });

  const pathEntries: string[] = [];
  if (venvInfo?.venvBinDir) {
    pathEntries.push(venvInfo.venvBinDir);
    env.VIRTUAL_ENV = venvInfo.venvDir;
  }

  const nodeBinDir = path.join(resolvedRepo, 'node_modules', '.bin');
  if (await isDirectory(nodeBinDir)) {
    pathEntries.push(nodeBinDir);
  }

  if (env.PATH) {
    pathEntries.push(...env.PATH.split(path.delimiter).filter(Boolean));
  }
  env.PATH = [...new Set(pathEntries)].join(path.delimiter);

  // PYTHONSAFEPATH=1 prevents Python 3.11+ from prepending CWD/script dir to sys.path,
  // avoiding accidental shadowing of stdlib modules (e.g. json.py, logging.py, types.py).
  env.PYTHONSAFEPATH = '1';
  env.PYTHONDONTWRITEBYTECODE = env.PYTHONDONTWRITEBYTECODE ?? '1';

  const existingPythonPath = env.PYTHONPATH
    ? env.PYTHONPATH.split(path.delimiter).filter(Boolean)
    : [];
  const mergedPythonPath = [...new Set([...pythonRoots, ...existingPythonPath])];
  if (mergedPythonPath.length > 0) {
    env.PYTHONPATH = mergedPythonPath.join(path.delimiter);
  }

  if (options.extraEnv) {
    Object.assign(env, options.extraEnv);
  }

  return {
    repoDir: resolvedRepo,
    env,
    venvDir: venvInfo?.venvDir ?? null,
    venvBinDir: venvInfo?.venvBinDir ?? null,
    pythonBinary: venvInfo?.pythonBinary ?? null,
    pythonRoots,
  };
}

function splitLeadingEnvAssignments(tokens: string[]): {
  envPrefix: string[];
  rest: string[];
} {
  const envPrefix: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    const token = tokens[i];
    if (token && /^[A-Za-z_][A-Za-z0-9_]*=/.test(token)) {
      envPrefix.push(token);
      i += 1;
    } else {
      break;
    }
  }
  return { envPrefix, rest: tokens.slice(i) };
}

function tokenizeCommand(command: string): string[] {
  const tokens: string[] = [];
  let current = '';
  let quote: '"' | "'" | null = null;
  let escaped = false;

  for (let i = 0; i < command.length; i += 1) {
    const ch = command[i]!;
    if (escaped) {
      current += ch;
      escaped = false;
      continue;
    }
    if (ch === '\\' && quote !== "'") {
      escaped = true;
      current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      if (quote === ch) {
        quote = null;
      } else if (quote === null) {
        quote = ch;
      }
      current += ch;
      continue;
    }
    if (/\s/.test(ch) && quote === null) {
      if (current.length > 0) {
        tokens.push(current);
        current = '';
      }
      continue;
    }
    current += ch;
  }

  if (current.length > 0) {
    tokens.push(current);
  }
  return tokens;
}

function injectPythonSafeFlag(args: string[]): string[] {
  let hasSafeFlag = false;
  for (const arg of args) {
    if (arg === '--' || !arg.startsWith('-')) {
      break;
    }
    if (arg === '-P' || /^-[A-Za-z]*P[A-Za-z]*$/.test(arg)) {
      hasSafeFlag = true;
      break;
    }
  }
  return hasSafeFlag ? args : ['-P', ...args];
}

/**
 * Normalizes a shell command to use the repository's virtualenv Python binary
 * (when present) and injects `-P` (safe path mode) for Python/pytest invocations
 * so script/CWD directories cannot shadow Python stdlib modules.
 */
export function normalizeCommandWithVenv(
  command: string,
  resolved: Pick<ResolvedRepoEnv, 'venvBinDir' | 'pythonBinary'>,
): string {
  const trimmed = command.trim();
  if (!trimmed) {
    return trimmed;
  }

  const tokens = tokenizeCommand(trimmed);
  const { envPrefix, rest } = splitLeadingEnvAssignments(tokens);
  if (rest.length === 0) {
    return trimmed;
  }

  const exe = rest[0]!;
  const exeBase = path.basename(exe);
  const args = rest.slice(1);

  if (exeBase === 'python' || exeBase === 'python3' || /^python3\.\d+$/.test(exeBase)) {
    const targetPython = resolved.pythonBinary ?? exe;
    const normalizedArgs = injectPythonSafeFlag(args);
    return [...envPrefix, targetPython, ...normalizedArgs].join(' ');
  }

  if (exeBase === 'pytest' || exeBase === 'py.test') {
    const targetPython = resolved.pythonBinary ?? 'python3';
    return [...envPrefix, targetPython, '-P', '-m', 'pytest', ...args].join(' ');
  }

  if (
    resolved.venvBinDir &&
    !exe.includes('/') &&
    !exe.includes('\\') &&
    ['pip', 'pip3', 'mypy', 'ruff', 'flake8', 'black', 'isort'].includes(exeBase)
  ) {
    return [...envPrefix, path.join(resolved.venvBinDir, exeBase), ...args].join(' ');
  }

  return trimmed;
}

export async function runInRepoEnv(
  repoDir: string,
  command: string,
  args: string[],
  options: ResolveRepoEnvOptions & { timeoutMs?: number } = {},
): Promise<{ stdout: string; stderr: string; env: ResolvedRepoEnv }> {
  const resolved = await resolveRepoEnv(repoDir, options);
  const normalized = normalizeCommandWithVenv(
    [command, ...args].join(' '),
    resolved,
  );
  const tokens = tokenizeCommand(normalized);
  const cmd = tokens[0] ?? command;
  const cmdArgs = tokens.slice(1);

  const { stdout, stderr } = await execFileAsync(cmd, cmdArgs, {
    cwd: resolved.repoDir,
    env: resolved.env,
    timeout: options.timeoutMs ?? 120_000,
    maxBuffer: 10 * 1024 * 1024,
  });

  return { stdout, stderr, env: resolved };
}

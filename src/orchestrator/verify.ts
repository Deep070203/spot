import { exec } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import {
  setBlackboardSection,
  type TaskBlackboard,
  type VerificationCheck,
  type VerificationContext,
} from './blackboard.js';
import { preserveProbe } from './codefix.js';
import type { StateManager } from './state.js';
import { normalizeCommandWithVenv, resolveRepoEnv } from '../utils/env.js';
import { commandTimeoutMs, detectProject } from '../utils/toolchain.js';

const execAsync = promisify(exec);

export interface RunVerificationOptions {
  stateManager?: StateManager;
  issueId?: string;
  reproFailedBeforeFix?: boolean;
  reproTestPath?: string;
  smokeCommand?: string;
  runTypecheck?: boolean;
  runSuite?: boolean;
}

export interface RunVerificationResult {
  blackboard: TaskBlackboard;
  verification: VerificationContext;
  passed: boolean;
}

async function executeCheck(
  name: string,
  command: string,
  cwd: string,
  env: Record<string, string>,
  timeoutMs: number,
): Promise<VerificationCheck> {
  const start = Date.now();
  try {
    const { stdout, stderr } = await execAsync(command, {
      cwd,
      env,
      timeout: timeoutMs,
      maxBuffer: 10 * 1024 * 1024,
    });
    return {
      name,
      command,
      passed: true,
      exitCode: 0,
      durationMs: Date.now() - start,
      stdout,
      stderr,
    };
  } catch (error) {
    const err = error as {
      code?: number | string;
      stdout?: string;
      stderr?: string;
      message?: string;
    };
    return {
      name,
      command,
      passed: false,
      exitCode: typeof err.code === 'number' ? err.code : 1,
      durationMs: Date.now() - start,
      stdout: err.stdout ?? '',
      stderr: err.stderr ?? err.message ?? '',
    };
  }
}

export async function runVerification(
  blackboard: TaskBlackboard,
  options: RunVerificationOptions = {},
): Promise<RunVerificationResult> {
  const repoDir = path.resolve(
    blackboard.repo.worktreePath ?? blackboard.repo.localPath,
  );

  if (options.stateManager && options.issueId && options.stateManager.has(options.issueId)) {
    options.stateManager.transition(options.issueId, 'verifying', {
      reason: 'Verifying reproduction probe pass and regression/smoke suite',
    });
  }

  const [project, resolvedEnv] = await Promise.all([
    detectProject(repoDir),
    resolveRepoEnv(repoDir),
  ]);

  const selectedId =
    blackboard.fix?.hypothesisId ??
    blackboard.localization?.hypothesisId ??
    blackboard.evaluator?.selectedHypothesisId ??
    blackboard.hunter?.hypotheses[0]?.id;

  const hypothesis = blackboard.hunter?.hypotheses.find((h) => h.id === selectedId);
  const verdict = blackboard.evaluator?.verdicts.find(
    (v) => v.hypothesisId === selectedId,
  );

  // Guardrail: verify/restore probe content before running verification.
  if (hypothesis?.probeFilePath && hypothesis.probeContent) {
    await preserveProbe(repoDir, {
      probeFilePath: path.resolve(repoDir, hypothesis.probeFilePath),
      content: hypothesis.probeContent,
    });
  }

  const checks: VerificationCheck[] = [];

  // 1. Re-run reproduction probe (must now exit 0!)
  const reproCommand = verdict?.reproCommand ?? hypothesis?.probeCommand;
  let reproPassedAfterFix = false;
  if (reproCommand) {
    const normalizedRepro = normalizeCommandWithVenv(reproCommand, resolvedEnv);
    const reproCheck = await executeCheck(
      'repro_probe',
      normalizedRepro,
      repoDir,
      resolvedEnv.env,
      commandTimeoutMs('probe', project.primaryLanguage),
    );
    checks.push(reproCheck);
    reproPassedAfterFix = reproCheck.passed;
  } else {
    reproPassedAfterFix = false;
  }

  // 2. Run typecheck/build if applicable
  let typecheckPassed = true;
  if (options.runTypecheck !== false && project.typecheckCommand) {
    const typecheckCmd = normalizeCommandWithVenv(
      project.typecheckCommand,
      resolvedEnv,
    );
    const tcCheck = await executeCheck(
      'typecheck',
      typecheckCmd,
      repoDir,
      resolvedEnv.env,
      commandTimeoutMs('build', project.primaryLanguage),
    );
    checks.push(tcCheck);
    typecheckPassed = tcCheck.passed;
  }

  // 3. Run regression / smoke test suite
  let suitePassed = true;
  const rawSuiteCommand = options.smokeCommand ?? blackboard.repo.testCommand ?? project.testCommand;
  if (options.runSuite !== false && rawSuiteCommand) {
    const suiteCmd = normalizeCommandWithVenv(rawSuiteCommand, resolvedEnv);
    const suiteCheck = await executeCheck(
      'test_suite',
      suiteCmd,
      repoDir,
      resolvedEnv.env,
      commandTimeoutMs('test', project.primaryLanguage),
    );
    checks.push(suiteCheck);
    suitePassed = suiteCheck.passed;
  }

  const reproTestPath =
    options.reproTestPath ??
    blackboard.fix?.edits.find((e) => /(?:test|spec)/i.test(e.filePath))?.filePath;

  let reproTestAdded = Boolean(reproTestPath);
  if (!reproTestAdded && hypothesis?.probeFilePath) {
    try {
      await fs.access(path.resolve(repoDir, hypothesis.probeFilePath));
      reproTestAdded = true;
    } catch {
      reproTestAdded = false;
    }
  }

  const passed = reproPassedAfterFix && typecheckPassed && suitePassed;

  const verification: VerificationContext = {
    passed,
    reproTestAdded,
    reproTestPath: reproTestPath ?? hypothesis?.probeFilePath,
    reproFailedBeforeFix:
      options.reproFailedBeforeFix ?? Boolean(verdict?.accepted || hypothesis?.probeOutput),
    reproPassedAfterFix,
    suitePassed,
    typecheckPassed,
    checks,
    regressionNotes: passed
      ? 'All verification checks passed after the fix.'
      : `Verification failed: ${checks
          .filter((c) => !c.passed)
          .map((c) => `${c.name} (exit=${c.exitCode})`)
          .join(', ')}`,
    completedAt: new Date().toISOString(),
  };

  const nextBlackboard = setBlackboardSection(
    blackboard,
    'verification',
    verification,
  );

  return {
    blackboard: nextBlackboard,
    verification,
    passed,
  };
}

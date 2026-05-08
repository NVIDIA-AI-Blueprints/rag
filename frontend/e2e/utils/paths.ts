/**
 * Absolute paths to e2e fixture files, resolved once so specs don't need to
 * redo `fileURLToPath(import.meta.url)` plumbing.
 */
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const e2eRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export const FILES = {
  samplePng: path.join(e2eRoot, 'utils/files/sample.png'),
  sampleTxt: path.join(e2eRoot, 'utils/files/sample.txt'),
  samplePdf: path.join(e2eRoot, 'utils/files/sample.pdf'),
};

export const E2E_ROOT = e2eRoot;

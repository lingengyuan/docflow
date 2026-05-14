import { execFileSync } from 'node:child_process';
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const roots = ['frontend/js', 'scripts'];
const files = [];

function collectJavaScriptFiles(path) {
  for (const name of readdirSync(path).sort()) {
    const current = join(path, name);
    const stat = statSync(current);
    if (stat.isDirectory()) {
      collectJavaScriptFiles(current);
    } else if (name.endsWith('.js') || name.endsWith('.mjs')) {
      files.push(current);
    }
  }
}

for (const root of roots) {
  collectJavaScriptFiles(root);
}

for (const file of files) {
  execFileSync(process.execPath, ['--check', file], { stdio: 'inherit' });
}


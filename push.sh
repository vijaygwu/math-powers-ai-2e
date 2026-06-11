#!/bin/bash
# One-shot init + push for math-powers-ai-2e. Safe to re-run.
set -e
cd "$(dirname "$0")"

echo "== repo dir: $(pwd)"

if [ ! -d .git ]; then
    git init -b main
fi

git add -A
git commit -m "Companion code for The Math That Powers AI, 2nd edition

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
    || echo "== nothing new to commit"

if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin https://github.com/vijaygwu/math-powers-ai-2e.git
fi

echo "== pushing to origin/main"
if ! git push -u origin main; then
    echo "== push rejected; rebasing onto remote (repo had initial files)"
    git pull --rebase origin main --allow-unrelated-histories
    git push -u origin main
fi

echo "== DONE: https://github.com/vijaygwu/math-powers-ai-2e"
git log --oneline -1

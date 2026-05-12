## Description
<!-- Provide a standalone description of changes in this PR. -->
<!-- Reference any issues closed by this PR with "closes #1234". -->

## Checklist
- [ ] I am familiar with the [Contributing Guidelines](../CONTRIBUTING.md).
- [ ] All commits are signed-off (`git commit -s`) and GPG signed (`git commit -S`).
- [ ] New or existing tests cover these changes.
- [ ] The documentation is up to date with these changes.
- [ ] If adjusting docker-compose.yaml environment variables have you ensured those are mimicked in the Helm values.yaml file.

## Skills / OpenClaw Changes
<!-- Complete this section when touching skills/**, .openclaw/**, or scripts/skill_compliance_check.py. -->
- [ ] I ran `python3 scripts/skill_compliance_check.py --skills-dir skills --openclaw-dir .openclaw --strict`.
- [ ] Skill changes have updated `evals/evals.json` with positive and negative/security cases.
- [ ] Destructive actions require explicit confirmation.
- [ ] Runtime data classification and secret-handling gates are documented.
- [ ] Reviewer/approver signoff is recorded per `skills/PLAYBOOK.md`.

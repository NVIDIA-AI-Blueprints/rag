# Legacy RAG Blueprint Agent Skill

This directory is legacy migration material for the original all-in-one
`rag-blueprint` skill. Do not install it for new work.

The canonical preview catalog now lives in [`../skills/`](../skills/), with
focused `rag-*` skills for deployment, ingestion, querying, configuration,
troubleshooting, evaluation, MCP, VLM, and guardrails.

## Migration Guidance

If `rag-blueprint` is already installed, remove it before installing the focused
catalog to avoid trigger conflicts:

```bash
rm -rf ~/.codex/skills/rag-blueprint
rm -rf ~/.claude/skills/rag-blueprint
rm -rf ~/.agents/skills/rag-blueprint
```

Then install the focused catalog using the instructions in
[`../skills/README.md`](../skills/README.md).

## Legacy Structure

```text
skill-source/.agents/skills/rag-blueprint/
  SKILL.md
  references/
    deploy.md
    deploy/
    configure/
    shutdown.md
    troubleshoot.md
```

The legacy references remain useful as source material while the split catalog
is deepened. New operational guidance should be added to `../skills/`, not here.

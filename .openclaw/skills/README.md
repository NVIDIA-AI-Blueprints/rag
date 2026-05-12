# OpenClaw Skill Routing

OpenClaw loads the canonical RAG skills through symlinks in this directory.

Do not maintain separate OpenClaw-only copies of the skill instructions. If
OpenClaw needs wrappers or manifest metadata, keep those wrappers thin and route
back to the matching `rag-*` skill.

The plugin manifest declares these symlinked directories relative to the plugin
root. If an archive or copy operation does not preserve symlinks, recreate them
or install the focused skills directly from `../../skills/`.

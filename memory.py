import config


def format_memory(memories):
    """Plain bullet list, for the judge, rubric writer and data generator (not the policy)."""
    return "\n".join(f"- {m}" for m in memories)


def format_persistbench_memories(memories):
    """PersistBench's rendering (prompts.formatted_memories), used in the policy prompt and its judge prompt."""
    return "<memories>\n" + "\n".join(f"- {m}" for m in memories) + "\n</memories>"


def policy_system_prompt(memories):
    """The policy's system prompt: PersistBench's generic template, filled in the way PersistBench fills it,
    so training prompts match evaluation prompts."""
    template = (config.PERSISTBENCH_DIR / "generic_prompt.txt").read_text()
    return template.replace("{model_name}", config.POLICY_NAME).replace("{memories}", format_persistbench_memories(memories))

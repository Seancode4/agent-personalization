def format_memory(memories):
    # TODO: switch to PersistBench's memory system prompt template so training matches evaluation.
    return "\n".join(f"- {m}" for m in memories)

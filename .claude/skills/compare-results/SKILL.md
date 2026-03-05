---
name: compare-results
description: Extract MAE scores from reproduction result directories and generate a comparison table against paper targets
argument-hint: <data-dir> [data-dir2 ...]
disable-model-invocation: true
allowed-tools: Bash(python *)
---

# Compare Results

Run the comparison script on the provided data directories:

```bash
python .claude/skills/compare-results/scripts/compare.py $ARGUMENTS
```

Print the generated markdown table to the user after the script completes:

```bash
cat comparison_table.md
```

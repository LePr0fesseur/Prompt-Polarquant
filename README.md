# PromptPolarQuant

**Token optimizer for AI prompts** — inspired by [PolarQuant](https://arxiv.org/abs/2502.02617) (Google/KAIST, 2025) and [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (Google, ICLR 2026).

Reduce your prompt token count by **30–66%** without losing semantic meaning — using pure math, zero ML dependencies.

---

## How it works

PolarQuant originally compresses KV cache vectors in LLM inference by converting them to polar coordinates `(radius, angle)` and applying quantization. This project adapts those same mathematical principles to **prompt-level token compression**:

| PolarQuant (KV cache) | PromptPolarQuant (prompting) |
|---|---|
| Key vector `K` → polar `(r, θ)` | Token → `(information radius, semantic angle)` |
| `r = √(K[2j]² + K[2j+1]²)` | `r = √(IDF² + positional_weight²) × content_density` |
| θ = direction in embedding space | θ = semantic pole (8 axes: time, place, action, cause…) |
| Optimal codebook quantization | Abbreviation codebook (FR + EN, ~50 rules) |
| Deduplicate vectors by angle bin | Deduplicate sentences with same semantic angle |
| Radius threshold = bit depth | Low-radius unit filtering = bit depth |

**Pipeline:**
1. **Codebook** — replace verbose phrases with short equivalents (`"in order to"` → `"to"`, `"je voudrais"` → `"je veux"`, etc.)
2. **Segmentation** — split prompt into semantic units (sentences/clauses)
3. **Polar representation** — compute `(radius, angle)` for each unit
4. **Quantization** — remove low-radius units, deduplicate by angle bin
5. **Token compression** — strip stopwords below the radius threshold
6. **Reconstruction** — reassemble in original order

---

## Results

| Mode | Token reduction | Use case |
|---|---|---|
| `--bits 4` | ~50–66% | Long prompts, cost-sensitive pipelines |
| `--bits 6` | ~30–50% | Daily use (default) |
| `--bits 8` | ~15–25% | Short prompts, preserve phrasing |

### Example (bits=6)

**Before (76 tokens):**
> Could you please provide me with a comprehensive and detailed explanation of how quantum computing works, with regard to its fundamental principles, in other words the qubits and superposition, and furthermore how it differs from classical computing? Thank you in advance for your thorough response.

**After (30 tokens, -60.5%):**
> please provide comprehensive explanation quantum computing works fundamental principles qubits superposition differs classical computing

---

## Install & usage

No dependencies beyond Python 3.10+ standard library.

```bash
git clone https://github.com/LePr0fesseur/Prompt-Polarquant.git
cd Prompt-Polarquant
```

### Interactive mode
```bash
python prompt_polarquant.py
```

### From a file
```bash
python prompt_polarquant.py -f my_prompt.txt --bits 6
```

### Pipeline / stdin
```bash
echo "my prompt here" | python prompt_polarquant.py --stdin --quiet --bits 6
```

### Built-in demo
```bash
python prompt_polarquant.py --demo
```

### As a Python library
```python
from prompt_polarquant import optimize_prompt

result, stats = optimize_prompt(
    "Could you please explain in detail how neural networks work?",
    n_bits=6,    # compression level: 4 (aggressive) to 8 (light)
    verbose=True # print compression report
)
print(result)
# "please explain detail neural networks work"
print(stats)
# {'original_tokens': 13, 'optimized_tokens': 6, 'token_reduction': 53.8, ...}
```

---

## CLI options

```
usage: prompt_polarquant.py [-h] [-f FILE] [-b BITS] [--stdin] [--quiet] [--demo] [prompt]

  -f, --file FILE    Input file containing the prompt
  -b, --bits BITS    Quantization depth: 4=aggressive, 6=balanced (default), 8=light
  --stdin            Read prompt from stdin
  --quiet            Output only the optimized prompt (no stats)
  --demo             Run built-in demonstration
```

---

## Mathematical foundations

The algorithm adapts two key ideas from the PolarQuant papers:

**1. Polar radius as information density**
```
r(token) = sqrt(IDF(token)^2 + pos_weight(token)^2) / sqrt(2)  *  novelty
```
- `IDF` = inverse document frequency across semantic units (rarity = informativeness)
- `pos_weight` = U-shaped positional weight (start/end carry more information — Jost's law)
- `novelty` = 1.0 for content words, 0.25 for stopwords

**2. Semantic angle for deduplication**
```
theta(token) = base_angle(semantic_pole) + sub_angle(sha256_hash)
```
8 semantic poles: entities, actions, qualities, quantities, location, time, cause, meta.
Sentences in the same angle bin = semantically redundant → keep only the highest-radius one.

**Unit-level radius** (content-word weighted):
```
R(unit) = mean(r_content) * sqrt(max(r_content)) * (0.4 + 0.6 * content_density)
```
This penalizes formulaic units (greetings, sign-offs) that have low content word density.

---

## References

- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617) — KAIST, AISTATS 2026
- [PolarQuant: Leveraging Polar Transformation for Efficient Key Cache Quantization](https://arxiv.org/abs/2502.00527)
- [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google, ICLR 2026
- [LLMLingua: Compressing Prompts for Accelerated Inference](https://llmlingua.com/llmlingua.html) — Microsoft Research

---

## License

MIT — free for any use, commercial or otherwise. See [LICENSE](LICENSE).

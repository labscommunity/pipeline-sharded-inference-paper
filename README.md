# Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets

**Tate Berenbaum** (Community Labs) and **Muthiah Venkatachalam** (Intel Corporation)

> We distribute large language model inference across fleets of commodity Intel AI PCs
> using pre-compiled pipeline shards composed with mask-based speculative decoding. Per-stage
> INT4 OpenVINO IR subgraphs are exported via `torch.jit.trace` with precomputed rotary
> embeddings and a post-export `beam_idx` Gather injection that unlocks the OpenVINO GPU
> plugin's `IndirectKVCache` fusion. We introduce a mask-based KV-cache rewind that makes
> speculative decoding practical on stateful OpenVINO without paged attention (1.35× mean
> speedup on a single node; 1.65× at 2048 tokens). The sharded + speculative + 2-stream
> micro-batched stack serves two concurrent users of Llama 3.1 8B at 41.25 tok/s total,
> 1.89× the monolithic single-user baseline on the same hardware, and 4.04× naïve
> distributed decode under a simulated 100 ms/hop WAN where naïve pipeline decode falls
> below the interactive threshold.

**arXiv:** [coming soon]

## Repository Structure

```
main.tex          Paper source
references.bib    Bibliography
arxiv.sty         Preprint style (NIPS-derived, single-column)
Makefile           Build via pdflatex + bibtex
figures/           Diagrams and plots
LICENSE            CC BY 4.0
```

## Building the Paper

Requires a TeX Live installation with `pdflatex` and `bibtex`.

```bash
make          # pdflatex → bibtex → pdflatex → pdflatex
make clean    # remove build artifacts
```

The compiled PDF will be written to `main.pdf`.

## Citation

```bibtex
@article{berenbaum2026shards,
  title={Pre-Compiled Pipeline Shards for Distributed {LLM} Inference on Intel {AI} {PC} Fleets},
  author={Berenbaum, Tate and Venkatachalam, Muthiah},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The accompanying source code in the [Rainier](https://github.com/labscommunity/rainier) repository is licensed separately.

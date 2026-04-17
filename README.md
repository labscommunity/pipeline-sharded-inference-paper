# Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets

**Tate Berenbaum** (Community Labs) and **Muthiah Venkatachalam** (Intel Corporation)

> We distribute large language model inference across fleets of commodity Intel AI PCs
> using pre-compiled pipeline shards. Per-stage INT4 OpenVINO IR subgraphs are exported
> via `torch.jit.trace` with precomputed rotary embeddings, producing stateful KV-cached
> shards chained over TCP. We report three findings: (1) compiled shards outperform
> monolithic inference by up to 22% on GPU and 45% on CPU, (2) a three-node fleet
> achieves 14.73 tok/s on Llama 3.1 8B at 85% of single-node throughput over WiFi, and
> (3) micro-batching via independent stateful InferRequests improves multi-user throughput
> by 1.38–1.66×.

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

This paper is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The accompanying source code in the [Rainier](https://github.com/communitylabs/rainier) repository is licensed separately.

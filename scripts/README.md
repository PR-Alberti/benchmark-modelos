# scripts/: os comandos de cada etapa

Cada script faz `source common.sh` (caminhos, ambiente Python, GPU) e chama uma etapa de `src/etapas/`.
O jeito recomendado de treinar é o `src/treina.py`, que chama estes scripts com a configuração do
experimento. Chamá-los direto também funciona, com variáveis de ambiente (veja o cabeçalho de cada um).

| Script | O que faz |
|---|---|
| `common.sh` | configuração lida por todos: `MINDEYE_ENV`, `MINDEYE_DATA`, caminhos, `PYTHONPATH`, e a atualização do site ao fim |
| **Treino** | |
| `run_frr.sh` | FRR |
| `run_frr_ms.sh` | FRR multi-sujeito |
| `run_ridgeonly_prior.sh` | MindEye2 ridge-only |
| `me1_run.sh` | MindEye1, uma etapa por vez: `train`, `lowlevel`, `recon`, `metrics` |
| **Reconstrução e métricas** | |
| `run_recon.sh` | MindEye2: reconstruções e refino |
| `run_recon_embeddings.sh` | FRR: reconstruções a partir do embedding, refino e métricas |
| `run_evals.sh` | métricas de um modelo (`base`, `enhanced`, `published`) |
| **Lotes** | |
| `run_reexecutar.sh` | refaz os modelos do benchmark com o `treina.py` (prefixo `repro_`), do treino às métricas; é o lote em uso |
| `run_me1_benchmark.sh` | MindEye1 com 1 e 40 sessões, do treino às métricas |
| `run_benchmark.sh` | o lote original do benchmark, com os nomes antigos (sem `repro_`). Os resultados dele foram substituídos pelas reproduções; fica como registro de como o benchmark foi montado |
| **Utilitários** | |
| `me1_setup.sh` | prepara o MindEye1 sobre o ambiente do MindEye2 |
| `me1_fake_data.py` | dados sintéticos do MindEye1, para medir memória e tempo sem baixar os dados |
| `inspect_ckpt.py` | mostra as opções com que um checkpoint foi treinado |

Todos são retomáveis: se interromper, rode o mesmo comando de novo.

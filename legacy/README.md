# Legado

Arquivos que ja nao fazem parte do fluxo, guardados por proveniencia. Nada aqui e chamado pelo
resto do repositorio.

| Pasta / arquivo | O que foi |
|---|---|
| `generators/make_*.py` | Geraram `train_ridgeonly.py`, `recon_inference.py`, `enhanced_recon_inference.py`, `final_evaluations.py` e `verify_retrieval.py` a partir dos notebooks de `notebooks/`. Hoje os scripts de `src/` sao a fonte da verdade, editados a mao: **rodar um gerador apaga essas edicoes** (o `train_ridgeonly.py` e o `recon_inference.py` diferem do que o gerador produz em 92 e 58 linhas; o `enhanced_recon_inference.py` e o `final_evaluations.py` ainda coincidem). Rodam de dentro de uma pasta com os notebooks. |
| `slurm/` | Jobs do cluster do repositorio original (MedARC). |
| `setup.sh` | Instalacao com os pins do repositorio original; substituida por `setup_env.sh`, que aplica as correcoes para maquinas atuais. |
| `run_ridgeonly.sh` | O wrapper com que o checkpoint `subj01_ridgeonly_1sess` do release foi treinado: sem prior e sem `--embedder_fp16`. Hoje, `PRIOR=0 scripts/run_ridgeonly_prior.sh` faz o equivalente (com o fp16 do embedder). |
| `run_fulltune_probe.sh` | Sonda de memoria do fine-tune completo (nao so a ridge) numa GPU de 20 GB. |

# Documentação

Quem chega agora: [GUIA.md](GUIA.md) primeiro (o problema, o glossário e o mapa do código), depois
[NOVO_COMPUTADOR.md](NOVO_COMPUTADOR.md) para preparar a máquina.

| Arquivo | O que tem | Para quem |
|---|---|---|
| [GUIA.md](GUIA.md) | o problema em uma página, glossário, o pipeline de ponta a ponta, onde fica cada coisa, receitas e armadilhas | quem vai mexer no código |
| **Esta branch** | | |
| [PROCEDIMENTO_ME1_MULTISUJ.md](PROCEDIMENTO_ME1_MULTISUJ.md) | ordem de serviço da branch: grade de pré-treinos multi-sujeito do MindEye1 (2/4/7 sujeitos) e ajustes no subj01 | o Claude que vai executar |
| **Instalar e rodar** | | |
| [NOVO_COMPUTADOR.md](NOVO_COMPUTADOR.md) | do computador zerado ao primeiro treino, só com terminal | quem vai rodar |
| [SETUP.md](SETUP.md) | referência da instalação: todas as bibliotecas, todos os arquivos externos, onde cada coisa fica, erros conhecidos | quando algo da instalação falha |
| **Os modelos** | | |
| [EXPERIMENTO.md](EXPERIMENTO.md) | o MindEye2 ridge-only e o FRR: onde a ridge entra, como o artigo treina e como este treina, os primeiros resultados (registro de set.–out./2026, com os nomes antigos) | MindEye2, FRR |
| [FRR_MULTISUJEITO.md](FRR_MULTISUJEITO.md) | o FRR multi-sujeito: os modelos A, B e C, como é ajustado, resultados | FRR |
| [MINDEYE1.md](MINDEYE1.md) | o MindEye1: instalação, o que mudou no código original, o protocolo no benchmark, as sessões intermediárias e o MindEye1 multi-sujeito | MindEye1 |
| [AGREGACAO_MEDIA.md](AGREGACAO_MEDIA.md) | o protocolo da branch `agregation`: média das repetições no treino, ME1 e ME2, 1 a 40 sessões | quem roda as filas |
| **Registro do trabalho** (retratos datados) | | |
| [ESTAGIO.md](ESTAGIO.md) | o plano de estágio e o que já está no repositório (atualizado em 14/11/2026) | orientação |
| [REEXECUTAR.md](REEXECUTAR.md) | o que foi refeito para o benchmark ficar completo e quanto as reproduções se afastam dos originais | quem confere os números |
| [REVISAO.md](REVISAO.md) | revisão crítica do benchmark (22/10/2026) e a ideia de decompor os modelos em peças | orientação, próximos passos |
| [IDEIAS_NEUROCIENCIA.md](IDEIAS_NEUROCIENCIA.md) | cinco perguntas de neurociência que o dataset permite testar (cansaço, erasing de regiões, brilho e contraste, memória, categorias nunca vistas), com desenho e cuidados; nenhuma rodada ainda | orientação, próximos passos |

Os números de todos os modelos, em dia, estão no site (`results/site/index.html`) e, resumidos, no
[README](../README.md#combinações-testadas). O [BENCHMARK.md](../BENCHMARK.md) é a foto de 19/10/2026, só com 1 e 40
sessões.

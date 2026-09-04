# Catálogo de datasets para BrainSniffer

Este catálogo separa dados que podem treinar a tarefa atual daqueles que servem
para validação externa, mecanismos fisiológicos ou uma tarefa diferente. O
BrainSniffer não mistura fontes automaticamente: cada expansão precisa de um
protocolo, uma licença/DUA e uma divisão por paciente.

**Última conferência das fontes:** 2026-09-02.

| Dataset | O que contém | Papel possível | Compatibilidade com o checkpoint atual | Ação |
|---|---|---|---|---|
| [EEG and BIS raw data — Figshare](https://figshare.com/articles/dataset/EEG_and_BIS_raw_data/5589841) | 24 casos de EEG e BIS; corpus usado pelo MVP; licença CC BY 4.0 | Treino e holdout principal | Compatível com o leitor atual e com o alvo BIS operacional | Já integrado; download pela interface/CLI |
| [VitalDB Open Dataset](https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md) | Casos perioperatórios com `BIS/EEG1_WAV`, `BIS/EEG2_WAV`, `BIS/BIS` e outros tracks | Pool de desenvolvimento auditado e validação externa congelada | Taxa/unidade nominais ajudam, mas ganho, referência, montagem e população diferem | Já integrado seletivamente; só entra no treino após gate e manifesto |
| [DOSE-I — Zenodo](https://zenodo.org/records/18483292) | 171 gravações/78,5 h de sedação com propofol em endoscopia; EEG frontotemporal de dois canais a 125 Hz; MOAA/S, estado de consciência, artefatos e eventos | Validação externa com anotação clínica independente do BIS; futura tarefa de classificação/ordinal | Não é compatível diretamente: sedação/endoscopia, taxa 125 Hz, dois canais e alvo MOAA/S/SoC em vez de BIS | Catalogado; exigir leitura do acordo de uso e adaptador/protocolo separado |
| [EEG dynamics during unconsciousness mediated by GABAergic anesthetics — PhysioNet](https://physionet.org/content/eeg-gaba-anesthesia/1.0.0/) | Quatro sujeitos, incluindo pacientes cirúrgicos; sinais alpha/slow, espectrogramas e concentrações/infusões; acesso credenciado com DUA | Validação mecanística de padrões de anestesia e teste de características | Não oferece o mesmo par contínuo EEG bruto+BIS do treino atual e tem poucos sujeitos | Não misturar ao treino; usar somente após credenciamento e protocolo |
| [Risk factors for burst suppression ratio — Dryad](https://doi.org/10.5061/dryad.r7sqv9sqg) | Coorte tabular de 10.827 pacientes e variáveis associadas a burst suppression/BIS | Análise auxiliar de risco e hipótese de subgrupos | Não contém o waveform EEG necessário para alimentar a CNN | Não usar como entrada do modelo; pode informar análise de subgrupos |

## Decisão de incorporação

O treino BIS original permanece reproduzível no Figshare, e o novo candidato
misto usa 23 casos Figshare mais 10 casos VitalDB aprovados pelo manifesto
`reports/corpus_manifest.json`. Os 15 VitalDB que já mediam mudança de domínio
continuam congelados como teste externo; não são reutilizados no ajuste. A
separação e os hashes dos arquivos ficam documentados em
[`docs/mixed_corpus.md`](mixed_corpus.md), para que o ganho observado não seja
confundido com uma validação feita no mesmo material.

DOSE-I só deve entrar em um experimento separado, com uma cabeça de modelo/tarefa
que declare MOAA/S ou estado de consciência como alvo e com split por pessoa.
PhysioNet GABA e Dryad não são substitutos do holdout BIS: o primeiro é pequeno e
tem acesso controlado, e o segundo é tabular sem waveform.

Essa separação é importante porque “profundidade anestésica” pode significar
coisas diferentes: um índice processado do monitor (BIS), uma escala clínica de
sedação (MOAA/S), resposta/consciência ou um marcador fisiológico de transição.
Um modelo treinado para uma dessas referências não pode ser apresentado como se
tivesse aprendido todas as outras.

## Próximo experimento recomendado

1. Obter acesso e ler o acordo de uso do DOSE-I.
2. Confirmar se a distribuição disponibilizada inclui o waveform bruto que será
   usado, além dos parâmetros EEG processados.
3. Criar um adaptador separado para 125 Hz e dois canais, preservando o
   checkpoint BIS sem alteração.
4. Definir previamente se o alvo será MOAA/S, SoC ou apenas detecção de
   transição; não converter esses rótulos arbitrariamente para BIS.
5. Avaliar por pessoa e por gravação, reportando abstention, artefatos e
   diferença entre sedação em endoscopia e cirurgia geral.

## Corpus misto implementado

O comando `brainsniffer build-corpus` cria um manifesto com gates de finitude,
lacunas, qualidade global, BIS válido e janelas aproveitáveis. O comando
`brainsniffer train-corpus` usa somente os arquivos elegíveis, faz amostragem
balanceada por grupo e fonte e registra o manifesto no checkpoint candidato.
Casos reprovados podem ser incluídos somente com `--include-quarantined`, para
que um experimento de estresse não seja confundido com o treino principal.

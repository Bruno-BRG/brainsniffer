# Catálogo de datasets para BrainSniffer

Este catálogo separa dados que podem treinar a tarefa atual daqueles que servem
para validação externa, mecanismos fisiológicos ou uma tarefa diferente. O
BrainSniffer não mistura fontes automaticamente: cada expansão precisa de um
protocolo, uma licença/DUA e uma unidade de separação explicitada. No Figshare,
a divisão é por caso, sem independência por pessoa comprovada; no VitalDB,
usa-se `subject_id` quando disponível, com fallback por caso da fonte.

**Última conferência das fontes:** 2026-09-02.

| Dataset | O que contém | Papel possível | Compatibilidade com o checkpoint atual | Ação |
|---|---|---|---|---|
| [EEG and BIS raw data — Figshare](https://figshare.com/articles/dataset/EEG_and_BIS_raw_data/5589841) | 24 casos de EEG e BIS; corpus usado pelo MVP; licença CC BY 4.0 | Treino e holdout principal | Compatível com o leitor atual e com o alvo BIS operacional | Já integrado; download pela interface/CLI |
| [VitalDB Open Dataset](https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md) | Casos perioperatórios com `BIS/EEG1_WAV`, `BIS/EEG2_WAV`, `BIS/BIS` e outros tracks | Pool de desenvolvimento auditado; avaliação cruzada de dataset exploratória do ativo e holdout histórico do misto | Taxa/unidade nominais ajudam, mas ganho, referência, montagem e população diferem | Já integrado seletivamente; só entra no treino após gate e manifesto |
| [DOSE-I — Zenodo](https://zenodo.org/records/18483292) | 171 gravações/78,5 h de sedação com propofol em endoscopia; EEG frontotemporal de dois canais a 125 Hz; MOAA/S, estado de consciência, artefatos e eventos | Validação externa com anotação clínica independente do BIS; futura tarefa de classificação/ordinal | Não é compatível diretamente: sedação/endoscopia, taxa 125 Hz, dois canais e alvo MOAA/S/SoC em vez de BIS | Catalogado; exigir leitura do acordo de uso e adaptador/protocolo separado |
| [EEG dynamics during unconsciousness mediated by GABAergic anesthetics — PhysioNet](https://physionet.org/content/eeg-gaba-anesthesia/1.0.0/) | Quatro sujeitos, incluindo pacientes cirúrgicos; sinais alpha/slow, espectrogramas e concentrações/infusões; acesso credenciado com DUA | Validação mecanística de padrões de anestesia e teste de características | Não oferece o mesmo par contínuo EEG bruto+BIS do treino atual e tem poucos sujeitos | Não misturar ao treino; usar somente após credenciamento e protocolo |
| [Risk factors for burst suppression ratio — Dryad](https://doi.org/10.5061/dryad.r7sqv9sqg) | Coorte tabular de 10.827 pacientes e variáveis associadas a burst suppression/BIS | Análise auxiliar de risco e hipótese de subgrupos | Não contém o waveform EEG necessário para alimentar a CNN | Não usar como entrada do modelo; pode informar análise de subgrupos |

## Decisão de incorporação

O manifesto histórico `reports/corpus_manifest.json` registra um pool elegível
de 23 Figshare + 10 VitalDB. Para o candidato `fixed`, retiram-se os cinco casos
do holdout Figshare: 18 Figshare + 10 VitalDB = 28 casos/grupos de desenvolvimento,
divididos em 16/6/6 (treino/validação/teste interno). O ajuste dos pesos usa
somente 12 Figshare + 4 VitalDB. A avaliação do ativo Figshare-only nos 15
VitalDB é avaliação cruzada de dataset exploratória. Para o candidato misto,
esses mesmos 15 casos são holdout histórico após exposição ao domínio e
reutilização para comparação, não validação externa confirmatória; não entram
no ajuste dos pesos. O checkpoint ativo não é promovido por esses resultados.
Partições, hashes e limites constam em [`docs/mixed_corpus.md`](mixed_corpus.md).
Reavaliar pesos congelados é diferente de reproduzir o treino histórico:
executar a receita no código atual produz um novo experimento, sem garantia de
identidade dos pesos. Os relatórios não comprovam a implementação atual.

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

## Corpus misto: caminho documentado, validação atual pendente

O comando `brainsniffer build-corpus` cria um manifesto com gates de finitude,
lacunas, qualidade global, BIS válido e janelas aproveitáveis. O comando
`brainsniffer train-corpus` usa somente os arquivos elegíveis, faz amostragem
balanceada por grupo e fonte e registra o manifesto no checkpoint candidato.
Casos reprovados podem ser incluídos somente com `--include-quarantined`, para
que um experimento de estresse não seja confundido com o treino principal.
Esta revisão não executou esses comandos nem revalidou o código em alteração.

O SQI dos gates é heurístico, não confiança clínica. A avaliação histórica
VitalDB registra interpolação linear de não finitos offline; ela pode usar
amostras futuras e não equivale à rejeição online antes do filtro/resampler.
Os gates de lacuna do pool não certificam retrospectivamente o holdout histórico.
Com offset zero, o alvo corresponde ao início da janela de 5 s, mas a emissão
online ocorre ao final; isso não valida alinhamento clínico ou atraso do BIS.
A arquitetura robusta é API experimental, não integrada à CLI e sem incerteza
validada; não é a arquitetura dos checkpoints históricos.

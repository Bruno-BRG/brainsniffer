# Ledger de fontes e decisões

Este ledger liga as decisões do BrainSniffer às fontes consultadas. Ele separa
três coisas diferentes: o que a fonte demonstra, o que foi adotado como
hipótese de engenharia e o que continua sem evidência. Uma referência não
transforma o protótipo em dispositivo médico e nenhum resultado de outro
trabalho é reutilizado como validação do BrainSniffer.

**Conferência histórica das URLs registrada neste ledger:** 2026-09-02. A revisão científica histórica está datada de 2026-09-04. A atualização documental abaixo não realizou nova consulta web nem leitura integral; datas históricas não constituem evidência de disponibilidade atual.

| Fonte primária ou normativa | Evidência relevante para este projeto | Decisões/artefatos influenciados | O que não é inferido |
|---|---|---|---|
| [Ma, 2017 — EEG and BIS raw data](https://doi.org/10.6084/m9.figshare.5589841.v1) | Corpus público pareando EEG frontal e referência BIS | `data/figshare.py`, `data/preprocess.py`, D1–D2, checkpoint e artigo | O BIS publicado não é uma verdade universal de consciência; o corpus não prova generalização |
| [Nsugbe & Connelly, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9160818/) | EEG frontal a 128 Hz, BIS em intervalos de 5 s e faixas de pesquisa derivadas do BIS; enfatiza variação entre pacientes | Janela de 5 s, taxa inicial, quatro estágios, split por caso e baseline comparável | A acurácia do artigo não é a acurácia da nossa CNN e não substitui coorte externa |
| [Park et al., 2020 — AnesNET](https://pubmed.ncbi.nlm.nih.gov/32746339/) | CNN compacta e inferência em tempo real; resumo previamente verificado informa alvo EEGMAC e 374 sujeitos (DOI 10.1109/TBCAS.2020.2998172), não BIS | `models/cnn.py`, escolha de CNN 1-D pequena e `benchmark-latency` | O hardware, índice EEGMAC e resultados do AnesNET não validam este checkpoint |
| [Hajat, Ahmad & Andrzejowski, 2017](https://pubmed.ncbi.nlm.nih.gov/28044337/) | Limitações de índices EEG processados, artefatos, fármacos e contexto clínico | D1, D7, D10, avisos de interface e linguagem “referência BIS estimada” | Uma correlação com BIS não equivale a medir consciência ou adequação anestésica |
| [Open reimplementation of BIS algorithms, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481655/) | Distingue índice proprietário, sinais intermediários e reimplementação independente | D1 e decisão de não chamar o BrainSniffer de reimplementação do BIS | O BrainSniffer não reproduz nem certifica o algoritmo proprietário do BIS |
| [Lee et al., 2022 — VitalDB](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/) e [documentação oficial de tracks](https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md) | Base perioperatória independente; tracks BIS/EEG e taxa declarada de EEG | `data/vitaldb.py`, avaliação externa seletiva e D14 | Compatibilidade nominal de track/unidade não prova compatibilidade de ganho, referência, aparelho ou população |
| [VitalDB Open Dataset API](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md) | Interface oficial para solicitar tracks do corpus aberto | Downloader seletivo, proveniência e termos documentados | O download não autoriza redistribuição nem uso clínico; os termos devem ser conferidos |
| [VitalDB clinical_data.csv](https://physionet.org/files/vitaldb/1.0.0/clinical_data.csv) | Mapeia `caseid` para `subjectid`, permitindo agrupar reoperações | Manifesto do corpus misto e split por grupo | A presença do identificador não resolve diferenças de aparelho, centro ou população |
| [DOSE-I](https://zenodo.org/records/18483292) | EEG frontotemporal a 125 Hz com MOAA/S, estado de consciência e anotações de artefato em sedação com propofol | Catálogo de expansão externa; não misturar ao alvo BIS sem tarefa e protocolo próprios | Sedação em endoscopia, dois canais, acesso/DUA e alvo diferentes não provam compatibilidade com o checkpoint |
| [PhysioNet GABA anesthesia](https://physionet.org/content/eeg-gaba-anesthesia/1.0.0/) | Quatro sujeitos com sinais alpha/slow, espectrogramas e dados de anestésico; acesso credenciado | Catálogo de validação mecanística, não treino supervisionado BIS | Poucos sujeitos, dados derivados e acesso controlado não formam um holdout BIS equivalente |
| [Dryad burst-suppression study](https://doi.org/10.5061/dryad.r7sqv9sqg) | Coorte tabular ampla sobre fatores associados a burst suppression | Possível análise auxiliar de subgrupos, não entrada da CNN | Sem waveform EEG, não pode alimentar o modelo atual |
| [Lab Streaming Layer — introdução oficial](https://labstreaminglayer.readthedocs.io/info/intro.html) e [Kothe et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12434378/) | Descoberta de streams, chunks, timestamps e sincronização em pesquisa | Transporte histórico removido: antiga CLI `stream-lsl`, D8/D16/D21; relatórios sintéticos preservados para auditoria, não capacidade atual | LSL não valida o EEG, não corrige perdas e não substitui o SDK ou aceite do fabricante |
| [FDA/Health Canada/MHRA — GMLP](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles) | Princípios de ciclo de vida, contexto de uso, dados e monitoramento de software de ML médico | D7, D15–D18, gates de aquisição e relatórios sem decisão clínica | Seguir princípios de desenvolvimento não constitui autorização regulatória |
| [DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) e [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) | Avaliação gradual em contexto, transparência, população e incerteza | `docs/_archive/prospective_protocol.md`, D15 e D20, bootstrap por cirurgia | O cumprimento do checklist não demonstra segurança ou utilidade neste sistema |

## Complemento: rastreabilidade e nível de verificação

Esta rodada conferiu arquivos locais e a consistência das 26 chaves, sem novas chamadas externas, downloads ou leitura integral de artigos. “Histórico” abaixo significa informação registrada pela pesquisa anterior, não verificação independente repetida. Metadados e resumo não bastam para reproduzir protocolo, avaliação de viés ou licença completa.

| Fonte/identificador | Nível de evidência disponível nesta rodada | Uso e limite |
|---|---|---|
| Figshare API `https://api.figshare.com/v2/articles/5589841`, DOI `10.6084/m9.figshare.5589841.v1` | Resultado de consulta pública anterior informado na pesquisa: 24 arquivos, CC BY 4.0; não se baixaram dados nesta rodada | Proveniência/licença declarada do registro; 24 arquivos não prova 24 pessoas únicas, anonimização ou aprovação própria. Confirmar versão e termos antes de redistribuir. |
| Nsugbe e Connelly, DOI `10.1049/htl2.12025` | Metadados locais e revisão histórica; sem releitura integral | Corrige “Li et al.” em D1/D19. Li Ma é autor do dataset; método multiescala/LDA não é evidência de CNN. |
| Park/AnesNET, DOI `10.1109/TBCAS.2020.2998172` | Resumo verificado pela pesquisa anterior: EEGMAC, 374 sujeitos; sem integral nesta rodada | Viabilidade arquitetural, não equivalência BIS/EEGMAC nem reprodução de split/hardware. |
| Pilge, DOI `10.1097/00000542-200603000-00016` | Revisão histórica/resultado anterior: atrasos variáveis sob condições artificiais; sem integral nesta rodada | Motiva sensibilidade de alinhamento, não offset universal nem estimativa causal de +20 s no Figshare. |
| TRIPOD+AI, DOI `10.1136/bmj-2023-078378` | Metadados presentes no `.bib` e referência histórica oficial BMJ 385:e078378 (2024) | Roteiro de relato adaptado ao alvo instrumental; não certificação. Conferência item a item oficial ainda humana. |
| PROBAST+AI, DOI `10.1136/bmj-2024-082505` | Metadados presentes no `.bib` e referência histórica oficial BMJ 388:e082505 (2025) | Roteiro de qualidade/risco de viés/aplicabilidade; nenhum parecer formal emitido. |
| EEGNet, DOI `10.1088/1741-2552/aace8c` | Somente metadados verificados pela pesquisa anterior | Candidato a comparação futura; sem inferir desempenho em BIS e sem inserir entrada não citada na bibliografia atual. |
| `LICENSE` e `CITATION.cff` locais | Texto integral local lido nesta rodada: MIT para software; CFF identifica software | Não licencia datasets de terceiros nem textos/figuras de artigos citados; citação não é permissão de uso. |

**Auditoria bibliográfica local:** na revisão de 24/09/2026, a bibliografia do artigo passou a 24 entradas, todas citadas; saíram `lsl` e `kothe2025` (suporte LSL removido do escopo) e o duplicado de PhysioNet (`clinicaldata`), e entrou a referência primária do BIS (`johansen2000`, Anesthesiology 93(5):1336--1344, 2000). Conferidos DOI, autoria, título, ano e veículo contra o registro histórico local, sem declarar validação remota exaustiva. Connor (`openbis`), Nsugbe (`nsugbe2022`), TRIPOD+AI e PROBAST+AI já estavam corrigidos; `dosei` já estava incluída. As ausências e a bibliografia manual citadas na revisão antiga estão marcadas como históricas/superadas, sem apagar a evidência. Autoria abreviada com `others`, datas de acesso, versões e termos de provedores ainda devem ser conferidos antes da submissão; não foram completados por suposição.

O [checklist de relato e ética](_archive/research/reporting_ethics_checklist.md) registra comparabilidade, agrupamento, missingness, abstenção e pendências de aprovação/licenciamento. A avaliação externa requer independência do desenvolvimento; transportabilidade exige caracterizar diferenças relevantes, não simplesmente chamar outro hospital de “externo”.

## Regra de uso do ledger

Antes de alterar taxa, janela, rótulo, arquitetura, split, métrica, política de
abstenção ou entrada de equipamento, atualize a decisão correspondente com:

1. a hipótese que motivou a mudança;
2. a fonte que sustenta a hipótese ou a indicação de que é uma decisão de
   engenharia ainda não sustentada;
3. o experimento que pode falsificá-la;
4. o limite que impede extrapolação clínica.

Se a fonte não estiver disponível, a mudança pode ser testada como hipótese,
mas deve permanecer marcada como exploratória e não deve substituir o
protocolo pré-especificado.

## Estado atual

O escopo definitivo do TCC é retrospectivo: arquivos existentes, comparação por
caso e replay com BIS como referência operacional. LSL foi removido; suas fontes
e relatórios permanecem como proveniência histórica, não justificativa de uma
capacidade atual nem evidência transferível ao JSONL local opcional.

Não há etapa de EEG físico ou validação prospectiva para concluir este TCC.
Essas questões exigiriam projeto separado, equipe clínica, ética, segurança e
revisão regulatória. As fontes não provam compatibilidade de hardware, segurança
ou utilidade clínica do BrainSniffer.

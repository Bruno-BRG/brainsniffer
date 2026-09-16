# Revisão de evidência científica para o artigo BrainSniffer

**Data da auditoria:** 4 de setembro de 2026

**Escopo:** estimativa de uma referência BIS a partir de EEG frontal, validade do rótulo, desenho de validação, incerteza e aquisição quase em tempo real.

**Estado desta revisão:** registro histórico da auditoria inicial. Os números de linha e achados de código/artigo abaixo descrevem aquela versão, não uma nova auditoria do trabalho em paralelo. A conferência documental posterior verificou as 26 entradas atuais de `docs/referencias.bib`: `nsugbe2022`, `openbis`, `clinicaldata`, `dosei`, `kothe2025`, `tripodai`, `probastai` e `imdrfn88` já estão presentes com as correções indicadas. Esses itens bibliográficos estão **atendidos no `.bib`**; sua integração/renderização no artigo cabe ao responsável pelo artigo. Nenhum texto integral foi novamente consultado nesta conferência. Ver [checklist de relato e ética](reporting_ethics_checklist.md) e [ledger(../../source_ledger.md).

**Artefatos auditados:** `docs/tcc_brainsniffer.tex`, `docs/referencias.bib`, `docs/source_ledger.md`, manifesto e relatórios JSON, metadados dos checkpoints, implementação do bootstrap e histórico Git.

## Método e regra de interpretação

Foram conferidas primeiro as referências já usadas no repositório. Novas fontes foram aceitas somente quando eram: artigo original, registro primário de dataset, página do editor/DOI/PubMed, documentação oficial ou orientação de autoridade regulatória/metodológica. DOI, título, autoria e periódico foram confrontados com Crossref, PubMed, editor ou repositório oficial.

Esta revisão separa três níveis:

- **Evidência científica primária:** resultado de estudo original; vale apenas para a população, agente, equipamento, alvo e desenho avaliados.
- **Autoridade/guideline:** orienta indicação, desenvolvimento ou relato; não valida o BrainSniffer e não substitui experimento.
- **Evidência interna:** relatório, código ou histórico do projeto; sustenta somente o que ocorreu nesta execução, não eficácia clínica.

## Síntese executiva

1. O alvo atual é o **valor produzido por um monitor BIS**, não consciência, adequação anestésica nem dose. O título, resumo, figuras e conclusão devem preferir “estimativa do rótulo/referência BIS” a “estimativa da profundidade anestésica”. A indicação oficial do BIS é contextual e supervisionada; experimentos originais mostram sensibilidade a bloqueio neuromuscular/EMG, cetamina e atraso variável de processamento.
2. CNNs e modelos temporais conseguem aproximar índices derivados de EEG, mas os estudos existentes diferem em rótulo (BIS, PSI ou EEGMAC), canais, agentes, equipamento e validação. O artigo de Nsugbe e Connelly atualmente citado não é evidência de CNN: seu resultado principal usa características multiescala e LDA.
3. A separação por caso do BrainSniffer protege contra vazamento de janelas da mesma cirurgia. Ela **não demonstra separação por paciente no Figshare**, porque `subject_id` está ausente e o fallback é o próprio caso. A alegação de impedir vazamento por reoperações só é sustentada onde há identificador de sujeito, como no VitalDB.
4. Para o checkpoint ativo treinado apenas no Figshare, VitalDB é uma avaliação cruzada de dataset exploratória. Para o candidato misto, que usou dez casos VitalDB no desenvolvimento, os 15 casos VitalDB são um holdout por participante da **mesma fonte/domínio**, não validação externa de domínio. Além disso, o benchmark VitalDB já constava do histórico antes do candidato misto; logo, a avaliação reutilizada não é confirmatória nem intocada pelo ciclo de desenvolvimento.
5. O bootstrap atual reamostra casos inteiros, o que é melhor que reamostrar janelas. Entretanto, concatena todas as janelas dos casos sorteados e recalcula a métrica; portanto, o estimando continua ponderado pela duração/número de janelas. Com cinco casos, o intervalo é apenas exploratório, e 1.000 réplicas não equivalem a 1.000 participantes.
6. TRIPOD+AI é guia de relato, DECIDE-AI é destinado à avaliação clínica inicial em ambiente real com decisões que afetam o cuidado, e GMLP é orientação de ciclo de vida. Nenhum deles certifica qualidade, segurança ou conformidade do protótipo. Para risco de viés, o complemento apropriado é PROBAST+AI.
7. LSL sustenta transporte e sincronização de pesquisa, não latência clínica determinística. O passo de 1 s é cadência de atualização; a janela causal de 5 s exige enchimento antes da primeira estimativa. Os números de 0,57/0,79 ms medem somente software sintético e atualmente não possuem relatório bruto versionado em `reports/`.

## Matriz claim → fonte → aplicação no artigo

| Claim defensável | Fonte primária/autoridade | Aplicação concreta no artigo | Força e limite |
|---|---|---|---|
| O BIS é uma saída de EEG processado usada como auxílio sob indicação e supervisão definidas, não uma verdade universal de consciência. | FDA, 510(k) K230693; Connor (2022) | Introdução, objetivo, resumo e conclusão: nomear o alvo como “referência/rótulo BIS”. | **Direta/autoridade.** A autorização é do BIS Advance e não se transfere ao BrainSniffer. |
| Protocolos guiados por BIS tiveram resultados dependentes do comparador e do contexto. | B-Aware, B-Unaware e BAG-RECALL | Fundamentação: apresentar benefício contextual e resultados conflitantes, sem concluir inutilidade nem superioridade universal. | **Direta, clínica e contextual.** São ensaios de protocolos, não validação do BIS como ground truth. |
| Bloqueio neuromuscular/EMG pode reduzir BIS em voluntários acordados; cetamina pode elevar o índice durante aprofundamento hipnótico. | Schuller et al. (2015); Hans et al. (2005) | Limitações do rótulo e discussão de artefatos/fármacos. | **Direta**, mas estudos pequenos e condições específicas. |
| Índices processados podem reagir com atraso variável a mudanças de sinal. | Pilge et al. (2006) | Justificar análise de alinhamento; não escolher offset pelo melhor resultado pós-hoc. | **Direta em bancada:** atrasos de 14–155 s nos índices testados; não estima automaticamente o offset destes arquivos. |
| Modelos convolucionais podem aproximar índices de EEG e operar em hardware compacto. | Park et al. (2020); Shi et al. (2023) | Trabalhos relacionados e motivação da CNN. | **Suporte de viabilidade.** Park prevê EEGMAC; Shi prevê PSI com quatro canais em 18 pacientes, não BIS de um canal. |
| No mesmo corpus público de 24 casos, desempenho por segmentos é mais otimista que avaliação agrupada por sujeito/caso. | Sukriti et al. (2026) | Fortalecer a justificativa do split agrupado e evitar comparação de leaderboard incompatível. | **Direta e muito relevante:** MAE 4,499 no split por segmentos versus 6,03±0,38 no GroupKFold; ainda é um único dataset. |
| Misturar registros do mesmo indivíduo entre treino e teste pode inflar desempenho. | Saeb et al. (2017); TRIPOD+AI, item 12c | Métodos: afirmar que todas as janelas de um grupo ficam na mesma partição e documentar o identificador. | **Direta/metodológica.** “Caso” só equivale a “paciente” quando a identidade é conhecida. |
| Avaliação externa exige dados não usados em nenhuma etapa do desenvolvimento e descrição explícita das diferenças de população, centro, hardware e alvo. | TRIPOD+AI; IMDRF N88; Archer et al. (2021) | Renomear as avaliações: cruzada de dataset para o ativo; holdout VitalDB da mesma fonte para o misto. | **Autoridade/metodológica.** Independência por participante não implica independência de domínio. |
| Exposição ao domínio VitalDB pode melhorar desempenho nele, mas não prova transporte a um novo domínio. | Lee et al. (2022) + relatórios internos | Resultados/discussão: “desempenho após exposição ao domínio VitalDB”, não “ganho de generalização externa”. | **Inferência interna**, não resultado estabelecido pela publicação VitalDB. |
| Reamostrar clusters inteiros preserva melhor a dependência intraclasse que reamostrar janelas. | Field e Welsh (2007) | Métodos do bootstrap: declarar unidade, estimando, algoritmo, semente e percentis. | **Direta/metodológica.** A validade depende do modelo de amostragem e não elimina fragilidade com poucos clusters. |
| Tamanho de validação deve ser justificado pela precisão desejada, inclusive calibração, e não pelo número de réplicas bootstrap. | Archer et al. (2021); TRIPOD+AI | Limitações e protocolo futuro: planejar número de casos e largura-alvo dos intervalos. | **Direta para desfecho contínuo.** Requer hipóteses de desempenho e variância da população-alvo. |
| Pearson mede associação; acordo requer análise adicional. | TRIPOD+AI; exemplos originais de Sukriti et al. (2026) e Kavuncu et al. (2026) | Resultados/figuras: acrescentar identidade, calibração, CCC e/ou Bland–Altman, mais métricas por caso. | **Metodológica.** Nenhuma métrica isolada estabelece utilidade clínica. |
| TRIPOD+AI organiza relato de desenvolvimento/avaliação, incluindo leakage, hiperparâmetros, clusters, intervalos, limitações e disponibilidade. | Collins et al. (2024) | Usar checklist preenchido como suplemento; não citar o guideline como fonte de números internos. | **Autoridade de relato**, não ferramenta de risco de viés. |
| DECIDE-AI começa quando o sistema é avaliado em ambiente clínico real e a decisão apoiada afeta o cuidado. | Vasey et al. (2022) | Manter como roteiro futuro após bancada e modo sombra; retirar qualquer impressão de conformidade clínica atual. | **Autoridade de relato por estágio.** Não se aplica como validação da execução retrospectiva/laboratorial. |
| GMLP exige população representativa, independência treino-teste, referência adequada ao uso e monitoramento no ciclo de vida. | IMDRF N88 (2025) | Discussão/protocolo: ligar cada princípio a uma ação futura mensurável. | **Autoridade regulatória internacional**, não aprovação de produto. |
| LSL oferece transporte, timestamps e sincronização para pesquisa quase em tempo real. | Kothe et al. (2025); documentação oficial LSL | Métodos: caracterizar LSL como middleware de pesquisa; medir cadeia completa no hardware alvo. | **Direta para infraestrutura.** Não valida canal, unidade, dispositivo nem prazo clínico. |

## Auditoria das alegações atuais

### Claims não sustentados ou excessivos

| Local no `.tex` | Problema | Redação/aplicação recomendada |
|---|---|---|
| Linhas 31 e 35 | “evitando leakage ... de reoperações” excede a evidência: no Figshare, `subject_id=null` e `group_id=figshare:case:*`. | “A separação por caso evita compartilhar janelas da mesma cirurgia; no VitalDB, `subjectid` também agrupa reoperações conhecidas. O Figshare não permite excluir repetição de paciente.” |
| Linha 44 | `li2022` aparece como sustentação de CNN, mas o estudo de Nsugbe e Connelly seleciona características multiescala e usa LDA; os autores dizem que deep learning não era apropriado à amostra. | Citar Nsugbe e Connelly para corpus, frequência, faixas e variabilidade; usar Park, Shi e Sukriti para redes profundas. |
| Linhas 74, 100, 108–110, 129 e 133 | Percentual do caso 24, thresholds 0,20/0,35, timeout 2 s, gates 90%/80% e recursos da interface são resultados/decisões internos, não achados de Figshare, GMLP ou DECIDE-AI. | Citar relatório/código/decisão internos e rotular os números como heurísticas de engenharia ainda não calibradas. |
| Linhas 52, 135, 179, 203–205 | “externo” e “out-of-dataset” são corretos para o checkpoint ativo Figshare-only, mas não para o candidato que já usou dez casos VitalDB no desenvolvimento. | Relatar cada modelo separadamente e expor sua origem de treino. Para o misto: “holdout VitalDB por participante, mesma fonte, sem ajuste nesses 15 casos”. |
| Linha 158 | A afirmação de que nenhuma seleção ocorreu após observar o conjunto externo não está sustentada. O relatório VitalDB aparece no commit `29478ed` (03/09, 17:51), antes do candidato misto `6a10042` (03/09, 23:43). | Dizer que a comparação é retrospectiva/exploratória e que o conjunto foi reutilizado após sua primeira inspeção; reservar “confirmatório” para uma nova coorte travada. |
| Linhas 168–177 | A tabela permite ler 23 Figshare + 5 holdout como grupos distintos, embora os cinco sejam subconjunto histórico; “treinado com 28 casos” também mistura corpus total e fit. | Explicitar: corpus fixo misto = 18 Figshare + 10 VitalDB = 28 grupos; metadados registram 16 treino, 6 validação e 6 teste. Dizer “desenvolvido em corpus de 28”, não “ajustado em 28”, salvo nova evidência. |
| Linhas 179, 220 e 228 | “melhora de generalização” não é demonstrada: o modelo recebeu dados do domínio VitalDB e foi comparado no mesmo benchmark já observado. | “Redução exploratória de erro após incorporar VitalDB ao desenvolvimento”; não inferir transporte a novo centro/equipamento. |
| Linhas 188 e 210 | O bootstrap é descrito como se cinco cirurgias produzissem intervalo robusto. O ponto reportado é a média bootstrap, não o estimador observado, e casos longos pesam mais. | Exibir estimativa observada + IC percentil, `n_cases`, `B`, seed, réplicas finitas e estimando; acrescentar versão macro por caso. |
| Linhas 190 e 127–129 | “p50/p95 por emissão” e passo de 1 s podem ser confundidos com latência ponta a ponta. Não há JSON bruto versionado com os números históricos. | Usar “tempo de processamento sintético local”; registrar hardware/OS/commit e medir aquisição→timestamp→buffer→resampling→inferência→UI. |
| Linhas 194 e 199 | O melhor resultado em `+20 s` é pós-hoc e compatível com várias explicações além de atraso do BIS. | Manter como sensibilidade; não selecionar offset. Mostrar diferenças pareadas por caso e incerteza contra offset 0. |
| Linhas 151–153, 179 e figuras de comparação | Pearson alto pode coexistir com viés/discordância importante e os agregados por janela ocultam heterogeneidade entre casos. | Acrescentar gráficos de acordo/calibração, distribuição por caso, erros por faixa BIS e exposição de treino de cada modelo. |

### Evidência interna que sustenta a auditoria

- `reports/corpus_manifest.json`: 59 arquivos; 33 casos elegíveis no desenvolvimento; 15 VitalDB congelados. Nos casos Figshare, o identificador de sujeito está ausente; nos 15 VitalDB congelados, os `subject_id` são preenchidos e distintos.
- `models/brainsniffer_corpus_fixed.json`: 49.948 janelas e 28 grupos (25.532 janelas Figshare; 24.416 VitalDB), com split registrado de 16/6/6 grupos para treino/validação/teste.
- `reports/vitaldb_external_validation.json`: checkpoint ativo, 38.730 janelas, MAE 12,43 e Pearson 0,024.
- `reports/mixed_vitaldb_external.json`: candidato misto, as mesmas 38.730 janelas, MAE 8,60 e Pearson 0,688.
- `src/brainsniffer/pipeline/metrics.py`: sorteia casos com reposição, concatena seus índices e executa métricas sobre todas as janelas; usa 1.000 réplicas, seed 42 e percentis 2,5/97,5.
- Histórico Git dos relatórios: `29478ed` antecede `6a10042`; isso prova a ordem dos artefatos, não a intenção de quem desenvolveu o modelo.
- Busca no repositório: 0,57/0,79 ms aparece em documentação, mas não há relatório bruto versionado de benchmark de latência em `reports/`.

## Recomendações por seção do artigo

### Título, resumo e objetivo

- Trocar a proposição principal por “estimativa de uma referência BIS a partir de EEG frontal”.
- Declarar número de **casos** antes de janelas e identificar o tipo de validação de cada modelo.
- Evitar “por paciente” para Figshare e “externa” para o candidato misto.
- Manter explícito: protótipo retrospectivo/laboratorial, sem decisão clínica e sem indicação de dose.

### Introdução e fundamentação

- Separar três construtos: efeito cortical do anestésico, responsividade/consciência e saída BIS.
- Acrescentar FDA, ensaios B-Aware/B-Unaware/BAG-RECALL, Schuller, Hans e Pilge; Hajat pode permanecer como revisão de contexto, mas não como evidência principal desta seção.
- Corrigir a atribuição para Nsugbe e Connelly (Li Ma é autor do dataset) e criar uma tabela de comparabilidade: fonte, pacientes, canal, agente, rótulo, split e validação externa. A atribuição em `decisions.md` e a tabela do checklist estão atendidas; a integração no artigo é responsabilidade de seu editor.

### Métodos

- Definir unidade de análise e identificador de agrupamento por fonte; admitir a limitação do Figshare.
- Informar exatamente quais transformações são fixas e quais são aprendidas; qualquer normalização, imputação, seleção, calibração ou tuning aprendido deve usar somente treino/validação.
- Descrever o candidato misto como corpus de 28 grupos com split 16/6/6, distinguindo ajuste, validação, teste interno e dois benchmarks históricos.
- Especificar no bootstrap: cluster=caso, estimando ponderado por janela, `B=1000`, seed 42, IC percentil e limitações. Planejar também média macro de métricas por caso e bootstrap pareado da diferença entre modelos.
- Preencher e anexar o checklist TRIPOD+AI; usar PROBAST+AI separadamente para risco de viés/aplicabilidade.

### Resultados

- Reportar por modelo: dados usados em desenvolvimento, casos/janelas do teste e momento em que o teste foi acessado.
- Dar estimativa observada e IC; não apresentar apenas média das réplicas.
- Incluir métricas por caso e por fonte, faixas BIS, qualidade/abstenção e dados ausentes.
- Acrescentar acordo/calibração: predito versus referência com linha identidade e suavizador, viés, limites de concordância e, se pré-especificado, CCC. Pearson deve permanecer como associação.
- Comparar modelos no mesmo caso com diferenças pareadas e IC; evitar inferência baseada em CIs separados.

### Discussão e conclusão

- Tratar a queda do ativo no VitalDB como evidência interna de sensibilidade a mudança de domínio, não prova causal de qual fator mudou.
- Tratar o resultado misto como adaptação/exposição ao domínio VitalDB e hipótese para novo teste.
- Distinguir incerteza de amostragem dos casos, variabilidade de seed/treino, seleção de modelo e incerteza do rótulo BIS.
- Propor nova validação pré-especificada em centro/equipamento/população não usados; DOSE-I deve ser tarefa separada de MOAA/S/estado de consciência, não conversão artificial para BIS.
- Reservar DECIDE-AI para futura avaliação clínica ao vivo; modo sombra sem efeito no cuidado ainda é etapa anterior.

### Figuras

- **Pipeline:** se “SQI” aparece sem validação própria, renomear para “heurística de qualidade”; usar “estimativa do rótulo BIS”.
- **Comparação:** mostrar, junto a cada modelo, fontes vistas no desenvolvimento; adicionar `n` de casos, pontos por caso e IC pareado. Não rotular o holdout do misto como externo ao domínio.
- **Bootstrap:** usar marcador da estimativa observada, IC com método/estimando no caption e aviso visível de `n=5`; considerar painel macro por caso.
- **Offset:** mostrar delta pareado por caso em relação a zero e marcar “pós-hoc; não usado para seleção”. Evitar eixo truncado que amplifique diferenças pequenas.
- **Nova figura prioritária:** acordo/calibração por fonte, com linha identidade e painel Bland–Altman; depois, erro por caso/faixa BIS/qualidade.
- **Latência:** somente após medir hardware real, usar waterfall com aquisição, buffer de 5 s, transporte, resampling, inferência, suavização e renderização; separar warm-up, cadência e latência.

## Auditoria das referências existentes — histórico, não lista atual de pendências

**Conferência posterior:** as linhas abaixo preservam o diagnóstico original. `li2022` foi substituída pela chave atual `nsugbe2022`; `openibis` corresponde agora a `openbis`, com Connor/*Anesthesia & Analgesia*/DOI corrigidos; `clinicaldata` e `dosei` foram incluídas; `kothe2025` tem *Imaging Neuroscience* e DOI; `imdrfn88` e `probastai` estão presentes. Não reintroduzir chaves antigas nem adicionar fontes apenas para aumentar a bibliografia. `hajat2017` e `gmlp` são identificadores históricos desta tabela, não chaves ausentes a recriar.

| Chave | Estado | Correção necessária |
|---|---|---|
| `ma2017` | DOI e autoria válidos. | Sustenta o dataset de 24 casos, não os percentuais de auditoria do caso 24. |
| `li2022` | DOI `10.1049/htl2.12025` válido. | A chave é pouco descritiva; o artigo é de Nsugbe e Connelly e não demonstra uma CNN. |
| `park2020` | DOI, autores e periódico válidos. | Não transferir EEGMAC, hardware, erro ou latência ao BrainSniffer. |
| `shi2023` | DOI, autores e periódico válidos. | Registrar alvo PSI, quatro canais, 18 pacientes de 66–92 anos e sedação com midazolam; comparabilidade indireta. |
| `hajat2017` | DOI válido, mas é revisão. | Usar apenas como síntese; claims centrais devem apontar para estudos originais. |
| `openibis` | Metadados incorretos no `.bib` e na bibliografia manual. | Correto: Christopher W. Connor, *Anesthesia & Analgesia* 135(4):855–864, 2022, DOI `10.1213/ANE.0000000000006119`; não *Scientific Reports*. |
| `lee2022` | DOI e metadados válidos. | Sustenta descrição do VitalDB, não compatibilidade automática de domínio. |
| `clinicaldata` | Citada no `.tex`, ausente em `docs/referencias.bib`. | Inserir futuramente a página/arquivo oficial PhysioNet com data de versão/acesso. |
| `dosei` | Citada no `.tex`, ausente em `docs/referencias.bib`. | Usar o registro Zenodo completo e não chamar seus rótulos de BIS. |
| `lsl` | Documentação oficial aceitável. | Manter separada do artigo científico de LSL. |
| `kothe2025` | Periódico errado e DOI ausente. | Correto: *Imaging Neuroscience* 3, IMAG.a.136 (2025), DOI `10.1162/IMAG.a.136`; não *Frontiers in Neuroscience*. |
| `gmlp` | Página triagência de 2021 é oficial. | Acrescentar o documento final IMDRF/AIML WG/N88 (2025), sem sugerir aprovação. |
| `decideai` | DOI e metadados válidos. | Citar apenas como guideline de futura avaliação clínica inicial ao vivo. |
| `tripodai` | DOI e metadados válidos. | É checklist de relato, não evidência de que thresholds ou resultados sejam corretos. |

**Registro histórico superado quanto ao `.bib`:** na versão inicialmente auditada, o `.tex` usava `thebibliography` manual e havia 15 chaves; `clinicaldata` e `dosei` faltavam no `.bib`. A ausência dessas entradas já foi corrigida. A frase histórica não descreve o mecanismo atual de compilação: migração bibliográfica, sincronização das citações e PDF devem ser verificadas pelo editor do artigo após suas alterações, não inferidas desta auditoria.

## Fontes primárias e oficiais verificadas

### BIS: indicação, mecanismo e limitações

- **U.S. Food and Drug Administration/Covidien (2024).** *BIS™ Advance Monitoring System — K230693, Indications for Use.* Sem DOI. [Registro FDA](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/pmn.cfm?ID=K230693); [PDF oficial](https://www.accessdata.fda.gov/cdrh_docs/pdf23/K230693.pdf).
- **Christopher W. Connor (2022).** *Open Reimplementation of the BIS Algorithms for Depth of Anesthesia.* *Anesthesia & Analgesia*, 135(4), 855–864. [DOI 10.1213/ANE.0000000000006119](https://doi.org/10.1213/ANE.0000000000006119); [PubMed](https://pubmed.ncbi.nlm.nih.gov/35767469/).
- **P. J. Schuller, S. Newell, P. A. Strickland, J. J. Barry (2015).** *Response of bispectral index to neuromuscular block in awake volunteers.* *British Journal of Anaesthesia*, 115(Suppl 1), i95–i103. [DOI 10.1093/bja/aev072](https://doi.org/10.1093/bja/aev072); [PubMed](https://pubmed.ncbi.nlm.nih.gov/26174308/).
- **P. Hans, P.-Y. Dewandre, J. F. Brichant, V. Bonhomme (2005).** *Comparative effects of ketamine on Bispectral Index and spectral entropy of the electroencephalogram under sevoflurane anaesthesia.* *British Journal of Anaesthesia*, 94(3), 336–340. [DOI 10.1093/bja/aei047](https://doi.org/10.1093/bja/aei047); [PubMed](https://pubmed.ncbi.nlm.nih.gov/15591328/).
- **Stefanie Pilge, Robert Zanner, Gerhard Schneider, Jasmin Blum, Matthias Kreuzer, Eberhard F. Kochs (2006).** *Time delay of index calculation: analysis of cerebral state, bispectral, and Narcotrend indices.* *Anesthesiology*, 104(3), 488–494. [DOI 10.1097/00000542-200603000-00016](https://doi.org/10.1097/00000542-200603000-00016); [PubMed](https://pubmed.ncbi.nlm.nih.gov/16508396/).
- **P. S. Myles, K. Leslie, J. McNeil, A. Forbes, M. T. V. Chan (2004).** *Bispectral index monitoring to prevent awareness during anaesthesia: the B-Aware randomised controlled trial.* *The Lancet*, 363, 1757–1763. [DOI 10.1016/S0140-6736(04)16300-9](https://doi.org/10.1016/S0140-6736(04)16300-9); [PubMed](https://pubmed.ncbi.nlm.nih.gov/15172773/).
- **Michael S. Avidan et al. (2008).** *Anesthesia Awareness and the Bispectral Index.* *New England Journal of Medicine*, 358, 1097–1108. [DOI 10.1056/NEJMoa0707361](https://doi.org/10.1056/NEJMoa0707361); [PubMed](https://pubmed.ncbi.nlm.nih.gov/18337600/).
- **Michael S. Avidan et al.; BAG-RECALL Research Group (2011).** *Prevention of Intraoperative Awareness in a High-Risk Surgical Population.* *New England Journal of Medicine*, 365, 591–600. [DOI 10.1056/NEJMoa1100403](https://doi.org/10.1056/NEJMoa1100403); [PubMed](https://pubmed.ncbi.nlm.nih.gov/21848460/).

### EEG frontal, redes e datasets

- **Li Ma (2017).** *EEG and BIS raw data* [dataset], Figshare, 24 casos, CC BY 4.0. [DOI 10.6084/m9.figshare.5589841.v1](https://doi.org/10.6084/m9.figshare.5589841.v1); [API Figshare](https://api.figshare.com/v2/articles/5589841).
- **Ejay Nsugbe, Stephanie Connelly (2022).** *Multiscale depth of anaesthesia prediction for surgery using frontal cortex electroencephalography.* *Healthcare Technology Letters*, 9, 43–53. [DOI 10.1049/htl2.12025](https://doi.org/10.1049/htl2.12025); [texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9160818/).
- **Yongjae Park, Su-Hyun Han, Wooseok Byun, Ji-Hoon Kim, Hyung-Chul Lee, Seong-Jin Kim (2020).** *A Real-Time Depth of Anesthesia Monitoring System Based on Deep Neural Network With Large EDO Tolerant EEG Analog Front-End.* *IEEE Transactions on Biomedical Circuits and Systems*, 14, 825–837. [DOI 10.1109/TBCAS.2020.2998172](https://doi.org/10.1109/TBCAS.2020.2998172); [PubMed](https://pubmed.ncbi.nlm.nih.gov/32746339/).
- **Meng Shi, Ziyu Huang, Guowen Xiao, Bowen Xu, Quansheng Ren, Hong Zhao (2023).** *Estimating the Depth of Anesthesia from EEG Signals Based on a Deep Residual Shrinkage Network.* *Sensors*, 23(2), 1008. [DOI 10.3390/s23021008](https://doi.org/10.3390/s23021008); [texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9865536/).
- **Sukriti, Chirag Kriplani, Suman Kumar, Abhishek Singh (2026).** *Predicting depth of anaesthesia from single-channel EEG using a deep TCN-BiLSTM-attention model with EWMA.* *Scientific Reports*, 16, 25470. [DOI 10.1038/s41598-026-54608-8](https://doi.org/10.1038/s41598-026-54608-8); [editor](https://www.nature.com/articles/s41598-026-54608-8).
- **Saliha Kevser Kavuncu, Mehmet Yalvaç, Alper Baştürk (2026).** *A Multitask Time–Frequency Deep Learning Approach for Anesthesia Depth Monitoring and Transition Prediction.* *Diagnostics*, 16(12), 1937. [DOI 10.3390/diagnostics16121937](https://doi.org/10.3390/diagnostics16121937); [editor](https://www.mdpi.com/2075-4418/16/12/1937). Exemplo recente de split por caso, métricas por caso e piloto externo; não é validação do BrainSniffer.
- **Hyung-Chul Lee, Yoonsang Park, Soo Bin Yoon, Seong Mi Yang, Dongnyeok Park, Chul-Woo Jung (2022).** *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* *Scientific Data*, 9. [DOI 10.1038/s41597-022-01411-5](https://doi.org/10.1038/s41597-022-01411-5); [texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/); [documentação oficial](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md).
- **Jakob Garbe, Quang Vu Nguyen, Jan W. Kantelhardt, Florian Dünninghaus, Karla Erffmeier, Katja Seeliger, Thomas Schmid (2026).** *DOSE-I: A Multimodal Biosignal Dataset of Procedural Sedation for Endoscopy* [dataset]. Zenodo. [DOI 10.5281/zenodo.18483292](https://doi.org/10.5281/zenodo.18483292); [registro](https://zenodo.org/records/18483292).

### Leakage, validação e incerteza

- **Sohrab Saeb, Luca Lonini, Arun Jayaraman, David C. Mohr, Konrad P. Kording (2017).** *The need to approximate the use-case in clinical machine learning.* *GigaScience*, 6. [DOI 10.1093/gigascience/gix019](https://doi.org/10.1093/gigascience/gix019); [texto integral](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/).
- **C. A. Field, A. H. Welsh (2007).** *Bootstrapping Clustered Data.* *Journal of the Royal Statistical Society: Series B*, 69(3), 369–390. [DOI 10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x); [editor](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2007.00593.x).
- **Lucinda Archer, Kym I. E. Snell, Joie Ensor, Mohammed T. Hudda, Gary S. Collins, Richard D. Riley (2021; online 2020).** *Minimum sample size for external validation of a clinical prediction model with a continuous outcome.* *Statistics in Medicine*, 40(1), 133–146. [DOI 10.1002/sim.8766](https://doi.org/10.1002/sim.8766); [PubMed](https://pubmed.ncbi.nlm.nih.gov/33150684/).

### Relato, ciclo de vida e aquisição

- **Gary S. Collins et al. (2024).** *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods.* *BMJ*, 385, e078378. [DOI 10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378); [checklist oficial](https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf).
- **Baptiste Vasey et al.; DECIDE-AI Expert Group (2022).** *Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI.* *Nature Medicine*, 28, 924–933. [DOI 10.1038/s41591-022-01772-9](https://doi.org/10.1038/s41591-022-01772-9); [editor](https://www.nature.com/articles/s41591-022-01772-9).
- **Karel G. M. Moons et al. (2025).** *PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods.* *BMJ*, 388, e082505. [DOI 10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505); [editor](https://www.bmj.com/content/388/bmj-2024-082505).
- **International Medical Device Regulators Forum, AI/ML-enabled Working Group (2025).** *Good machine learning practice for medical device development: Guiding principles*, IMDRF/AIML WG/N88 FINAL:2025. Sem DOI. [Página oficial](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles); [PDF oficial](https://www.imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf).
- **Christian Kothe, Seyed Yahya Shirazi, Tristan Stenner, David Medine, Chadwick Boulay, Matthew I. Grivich, Fiorenzo Artoni, Tim Mullen, Arnaud Delorme, Scott Makeig (2025).** *The Lab Streaming Layer for Synchronized Multimodal Recording.* *Imaging Neuroscience*, 3, IMAG.a.136. [DOI 10.1162/IMAG.a.136](https://doi.org/10.1162/IMAG.a.136); [documentação/repositório oficial](https://github.com/sccn/labstreaminglayer).

## Conclusão operacional para o agente do artigo

As correções de maior impacto são: (1) redefinir o alvo como referência BIS; (2) corrigir o nível de agrupamento Figshare; (3) separar avaliação cruzada de dataset do ativo de holdout mesma-fonte do misto; (4) retirar a alegação de benchmark intocado; (5) explicitar corpus 28 versus split 16/6/6; (6) reportar bootstrap e métricas por caso com estimando claro; (7) substituir “tempo real” por caracterização mensurada da cadeia; e (8) corrigir os metadados bibliográficos. O item (8) está atendido no `.bib` atual (chave `openbis`, além de `kothe2025`, `clinicaldata` e `dosei`); a conferência final das citações/PDF permanece com o editor do artigo.

Até uma nova avaliação pré-especificada em dados independentes do desenvolvimento, com diferenças de centro/equipamento/população explicitadas para avaliar transportabilidade (outro hospital não é requisito definidor de independência), e uma aquisição física ponta a ponta, o claim máximo defensável é: **protótipo de engenharia reproduzível para estimar uma referência BIS em replay e streaming de pesquisa, com resultados retrospectivos exploratórios e limitações explícitas de rótulo, amostra e domínio**.

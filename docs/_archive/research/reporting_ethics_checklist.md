# Checklist de relato, comparabilidade e ética — BrainSniffer

## Escopo e estado

Roteiro documental para estudo retrospectivo/laboratorial que estima **a referência instrumental BIS**, não consciência, adequação anestésica, dose ou benefício clínico. Não é checklist oficial integral preenchido, parecer PROBAST+AI, aprovação ética, dispensa de consentimento nem autorização regulatória. A revisão desta rodada foi local; não houve consulta web, download de dados ou leitura integral nova de artigos. Evidências externas abaixo provêm da [revisão histórica](article_evidence_review.md) e do [ledger com níveis de verificação(../../source_ledger.md).

- **TRIPOD+AI:** Collins et al., BMJ 385:e078378 (2024), DOI [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378), chave `tripodai`. Orienta transparência do desenvolvimento/avaliação de modelos preditivos.
- **PROBAST+AI:** Moons et al., BMJ 388:e082505 (2025), DOI [10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505), chave `probastai`. Orienta avaliação de qualidade, risco de viés e aplicabilidade; não é intercambiável com um guia de relato.

**Aplicabilidade justificada, com adaptação:** embora o BIS seja uma saída de monitor e não um desfecho clínico independente, seleção de participantes, definição de preditores/alvo, dados ausentes, agrupamento, tuning, avaliação e disponibilidade continuam relevantes. Usar esses guias para tornar tais decisões verificáveis não converte regressão instrumental em modelo clínico validado. Itens especificamente clínicos devem ser marcados como não aplicáveis ao uso atual ou pendentes, com justificativa, após confronto humano com os formulários oficiais. DECIDE-AI permanece orientação para eventual avaliação clínica inicial, não validação deste replay.

## Checklist adaptado de relato e risco de viés

Estados: **documentado** = descrição local disponível, não eficácia comprovada; **parcial** = há informação, mas falta verificação/artefato; **pendente humano** = requer decisão ou revisão responsável. Não foram atribuídos julgamentos formais de risco baixo/alto por domínio PROBAST+AI.

| Tema | O que deve ser relatado/verificado | Evidência local e estado | Pendência/responsável |
|---|---|---|---|
| Objetivo e uso | Regressão de BIS; população, canal e contexto pretendidos; exclusão de decisão/dose/alarme | D1/D7 em `../decisions.md`; documentado como limite de escopo | Autor e anestesiologista: confirmar redação final, evitar equivaler estágios derivados a consciência |
| Seleção e proveniência | Critérios por fonte, versão, casos elegíveis/excluídos e fluxograma com denominadores | Revisão histórica: Figshare e VitalDB; parcial | Autor/dados: auditar manifesto atual e descrever exclusão do case24 como decisão interna, não achado da publicação |
| Agrupamento/leakage | Todas as janelas do mesmo caso no mesmo split; reoperações do mesmo sujeito agrupadas quando identificáveis | Figshare sem `subject_id`: caso não comprova paciente único. VitalDB permite `subjectid`; parcial | Analista: checar sobreposição entre treino/validação/teste, inclusive reoperações; explicitar incerteza residual Figshare |
| Preditores e alvo | Montagem, unidade, taxa, filtros, janela causal, relógios, suavização e disponibilidade temporal do BIS | D2/D3/D16/D19; parcial | Engenheiro/analista: separar transformações fixas das aprendidas; ajustar imputação/normalização aprendida apenas no desenvolvimento |
| Missingness | Faltas de EEG, BIS, metadados e covariáveis; casos/janelas/duração afetados antes e depois das exclusões | D10 distingue imputação offline e rejeição online de não finitos; parcial | Analista: relatar por fonte e caso, causas conhecidas, mecanismo como hipótese (não presumir MCAR), método e sensibilidade |
| Qualidade e abstenção | Limiar heurístico, saturação, linha plana, perda e stale; denominador completo e cobertura | D10/D18/D21; documentado como engenharia, não SQI validado | Analista/engenheiro: cobertura por caso/faixa/fonte e erro nas janelas aceitas; investigar se exclusões selecionam casos fáceis |
| Desenvolvimento | Arquitetura, features, hiperparâmetros, sementes, treinamento, versões e regra de seleção | D4/D9 e revisão histórica; parcial | Autor/analista: vincular cada resultado ao checkpoint e split efetivos, sem contar corpus total como conjunto de ajuste |
| Independência da avaliação | Dados não usados para pesos, tuning, escolha de offset/limiar ou decisões após inspeção | Benchmark VitalDB reutilizado no histórico; exploratório | Analista: separar ativo Figshare-only de candidato exposto ao VitalDB; pré-especificar nova avaliação sem reuso para seleção |
| Transportabilidade | Descrever diferenças de população, tempo, aparelho, montagem, agentes e assistência | VitalDB é outra fonte; isso não isola a causa da mudança de domínio | Equipe: independência não exige outro hospital; outra instituição tampouco garante independência ou representatividade |
| Métricas e incerteza | MAE/RMSE/viés, associação versus acordo/calibração; por caso e fonte; estimativa observada + IC | Revisão histórica: bootstrap por caso, ponderado por janelas e apenas cinco casos no holdout; parcial | Analista: verificar implementação atual; declarar unidade, estimando, B, seed, réplicas finitas e IC; incluir macro por caso e diferenças pareadas quando calculadas |
| Precisão e amostra | Justificar casos/participantes pela precisão desejada, não por número de janelas/réplicas | Archer et al., chave `archer2021`; plano, não cálculo efetuado | Estatístico: definir precisão e hipóteses para nova amostra; não tratar 1.000 réplicas como participantes |
| Alinhamento | Distinguir relógio, janela e atraso variável do monitor; sensibilidade pós-hoc | D19; Pilge sob condições artificiais, não offset universal | Analista/engenheiro: pré-especificar alinhamento e não escolher +20 s pelo melhor holdout |
| Latência e uso | Cadência, enchimento de janela, processamento sintético e cadeia ponta a ponta separados | D6/D11 e protocolo prospectivo; parcial | Engenheiro: medir hardware/bridge/UI, registrar ambiente e relatório bruto; não transferir latência de outro artigo |
| Disponibilidade | Versão de código/checkpoint, configurações, proveniência, restrições e instruções de reprodução | `../../LICENSE`, `../../CITATION.cff`, ledger; parcial | Autor: verificar disponibilidade real e permissões antes de qualquer publicação; não prometer dados restritos |

## Comparabilidade: não há conversão automática BIS ↔ PSI ↔ EEGMAC

Os números de participantes e detalhes abaixo são os registrados na pesquisa anterior, não uma nova extração de texto integral. Ausência de detalhe é pendência, não equivalência presumida. Métricas de artigos distintos não formam um ranking comparável.

| Sistema/estudo | Alvo e amostra conhecida | Canal/contexto conhecido | Split/independência e limite para BrainSniffer |
|---|---|---|---|
| BrainSniffer / Figshare (Ma, `ma2017`) | BIS; API anterior registra 24 arquivos; isso não comprova 24 pessoas únicas | EEG frontal, 128 Hz, rótulo em intervalos de 5 s descritos na literatura | Agrupamento por caso; identidade longitudinal ausente. Não equivale a validação de consciência |
| Nsugbe e Connelly (`nsugbe2022`, DOI 10.1049/htl2.12025) | Faixas derivadas do BIS no corpus público | EEG frontal; características multiescala e LDA | Contexto de corpus e variabilidade, não comprovação de CNN. Conferir protocolo original antes de comparação numérica |
| VitalDB (`lee2022`, DOI 10.1038/s41597-022-01411-5) | BIS, fonte perioperatória distinta; piloto local histórico com 15 casos | Tracks BIS/EEG e BIS; unidade/montagem/ganho precisam de compatibilidade | Ativo Figshare-only: avaliação cruzada de dataset exploratória. Candidato misto: holdout de fonte já usada; independência de participantes não prova domínio novo |
| Shi et al. (`shi2023`, DOI 10.3390/s23021008) | PSI; revisão histórica registra 18 pacientes, 66–92 anos | Quatro canais, sedação com midazolam, segundo revisão anterior | PSI tem algoritmo/escala próprios; não converter para BIS. Split e seleção precisam ser extraídos do original antes de comparação quantitativa |
| Park et al. / AnesNET (`park2020`, DOI 10.1109/TBCAS.2020.2998172) | EEGMAC; resumo previamente verificado informa 374 sujeitos | Sistema neural/analógico compacto; sem nova leitura integral | Resumo não basta para reconstruir split, canais ou agentes. Não transferir erro/latência/hardware ao BIS ou BrainSniffer |
| EEGNet (DOI 10.1088/1741-2552/aace8c) | Somente metadados conhecidos nesta pesquisa | Arquitetura candidata a estudo futuro, não experimento executado | Não presumir desempenho em BIS. Comparar futuramente com mesmo corpus/splits/recursos; não adicionada ao `.bib` sem citação no artigo |

BIS, PSI e EEGMAC são referências distintas. Igualdade de escala nominal, correlação alta ou a expressão “depth of anesthesia” no título não estabelece equivalência semântica, calibração ou validade clínica. Comparação direta requer protocolo comum, alvo explícito, população compatível, aquisição documentada e avaliação independente; mapear entre índices seria uma tarefa nova.

## Ética, dados públicos e licenças

| Objeto | O que está documentado | O que não pode ser inferido / ação humana pendente |
|---|---|---|
| Software BrainSniffer | `LICENSE` local: MIT, copyright Bruno Rocha Guimarães, 2026; `CITATION.cff` identifica software v0.1.0 e MIT | CFF é metadado de citação, não licença de terceiros. Autor deve confirmar direitos sobre contribuições, dependências e material incorporado |
| Artigos e figuras citados | DOI e bibliografia identificam obras; licença deve ser conferida na versão/editor de cada obra | Acesso aberto não significa autorização irrestrita; licença do artigo não se transfere ao dataset nem ao software. Reprodução de figura/texto exige licença ou permissão pertinente |
| Figshare 5589841 | Consulta API anterior informada: 24 arquivos, CC BY 4.0, DOI versionado `.v1` | Confirmar termos/versionamento, atribuição e alterações indicadas antes de redistribuir. A licença de copyright não comprova consentimento nem anonimização |
| VitalDB | Ledger/D14 registram CC BY-NC-SA 4.0 e termos do provedor como referência a conferir | Não presumir que licença do artigo Scientific Data ou MIT cobre os dados. Responsável deve verificar termos da versão/endpoint, uso não comercial, compartilhamento e demais condições aplicáveis |
| DOSE-I e demais corpora | Apenas catálogo/referências nesta tarefa; nenhuma nova coleta ou incorporação | Conferir licença, DUA, credenciamento e finalidade por provedor antes de obter/usar/redistribuir; não presumir acesso por haver DOI público |
| Pesos, relatórios e sinais derivados | Proveniência e direitos dependem dos insumos e do artefato | Licença MIT do código não decide automaticamente a licença/permissão de checkpoints, dados derivados ou relatórios com informação sensível; responsável deve avaliar antes de publicar |

### Anonimização, pseudonimização e revisão ética

- **Pseudonimização:** substituir identificadores por IDs/códigos com possibilidade de ligação (por chave ou informação adicional) não torna os dados anônimos. `subjectid`, `caseid` e IDs de estudo não comprovam anonimização.
- **Anonimização:** requer avaliação de risco de reidentificação no contexto e com dados auxiliares disponíveis; remover nome não basta. Não houve auditoria independente de anonimização nesta rodada, nem tentativa de reidentificar pessoas.
- Dados publicamente acessíveis continuam sujeitos a finalidade, licença, termos e requisitos de proteção de dados aplicáveis. Não concluir dispensa automática de apreciação ética, consentimento ou base legal apenas por serem públicos.
- **Não foi localizada nos arquivos lidos comprovação de aprovação ética própria ou dispensa institucional específica do BrainSniffer.** Isso não prova inexistência fora do repositório. Não inventar número de parecer, comitê, aprovação, consentimento ou isenção; aprovação do estudo que originou o dataset não equivale à aprovação desta pesquisa secundária.
- Para a redação final, pesquisador responsável e instância institucional competente devem definir/documentar necessidade de apreciação ou dispensa para uso secundário, referência ao parecer original quando comprovado, base legal aplicável, minimização, retenção, controle de acesso e resposta a incidentes. A avaliação jurídica/de proteção de dados cabe ao responsável competente.
- Qualquer aquisição futura com pacientes ou voluntários, inclusive modo sombra sem efeito assistencial, depende das autorizações e do protocolo pertinente. O [protocolo prospectivo](../prospective_protocol.md) é plano, não aprovação. Não conectar saída a cuidado, bombas, dose ou alarmes por força deste checklist.

## Pendências humanas e critérios de encerramento

1. **Autor/editor do artigo:** confrontar o manuscrito final e suplemento com cada item oficial TRIPOD+AI, registrar localização ou justificativa de não aplicabilidade; conferir citações e compilação. Esta tarefa não editou o artigo.
2. **Revisor metodológico/estatístico:** realizar avaliação PROBAST+AI usando formulários oficiais e evidência suficiente; não há parecer formal emitido aqui. Auditar agrupamento, missingness, seleção por abstenção, estimando e reutilização do teste.
3. **Pesquisador responsável + instituição/comitê competente:** confirmar e registrar aprovação/dispensa aplicável, consentimento ou fundamento pertinente e proteção de dados; resolver a ausência documental antes de alegações éticas na submissão ou nova aquisição.
4. **Responsável por dados/direitos:** verificar licenças por objeto/versão e autorização de redistribuição de dados, figuras e pesos. Nenhuma publicação ou redistribuição foi executada.
5. **Equipe clínica/engenharia:** aprovar finalidade de pesquisa e plano independente de bancada/aquisição; pré-especificar novo estudo para transportabilidade e alinhamento. Não há validação clínica ou decisão regulatória nesta rodada.

**Registro de mudanças e limites:** corrigida atribuição em D1/D19; mantida auditoria antiga com marcação de itens bibliográficos já atendidos; preservadas 26 chaves; estruturado o DOI já existente de `clinicaldata`. Este checklist e o ledger documentam recomendações, fontes e incertezas, sem acrescentar referências não citadas para inflar bibliografia. Nenhum dado foi baixado e nenhuma aprovação, submissão, publicação ou teste clínico foi realizado.

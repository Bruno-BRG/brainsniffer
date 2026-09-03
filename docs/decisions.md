# Registro de decisões técnicas e científicas

Este documento evita que o protótipo vire uma coleção de hiperparâmetros escolhidos apenas porque melhoraram uma métrica. Cada decisão pode mudar, mas deve mudar junto com uma hipótese e uma referência.

## Matriz de sistemas conhecidos

| Sistema/recurso | Papel conhecido | Decisão no BrainSniffer |
|---|---|---|
| BIS | Índice proprietário processado de EEG, usualmente apresentado em escala 0–100; o registro regulatório descreve 40–60 como faixa de referência para anestesia geral. | Usar apenas como rótulo operacional do dataset; não declarar que a CNN mede consciência diretamente. |
| PSI | Outro índice proprietário de EEG processado, com escala e algoritmo próprios. | Não usar como alvo porque não está disponível no corpus escolhido; comparar índices só em estudo que contenha ambos. |
| Entropy / Narcotrend | Famílias de monitores processados que motivam medidas espectrais e classificação de estados. | Usar potência de banda, entropia e frequência de borda em baseline transparente; não copiar algoritmos proprietários. |
| Openibis | Reimplementação aberta para estudar etapas e limitações de um índice BIS. | Usar como referência de auditabilidade; não apresentar o BrainSniffer como reimplementação do BIS. |
| AnesNET e CNNs de DoA | Trabalhos que demonstram CNN compacta e inferência rápida em EEG pré-gravado. | Inspirar a escolha de uma CNN 1-D pequena e medir latência, sem reutilizar resultados como validação clínica. |
| Lab Streaming Layer | Camada aberta para descoberta, transporte, timestamps e sincronização de streams de pesquisa. | Usar como interface vendor-neutral; o driver/bridge e a verificação do canal continuam obrigatórios. |

**Fontes:** [FDA K202621](https://www.accessdata.fda.gov/cdrh_docs/pdf20/K202621.pdf), [review de monitoramento pEEG](https://pubmed.ncbi.nlm.nih.gov/34392880/), [limitações do BIS](https://pubmed.ncbi.nlm.nih.gov/28044337/), [Openibis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481655/), [AnesNET](https://pubmed.ncbi.nlm.nih.gov/32746339/) e [LSL](https://labstreaminglayer.readthedocs.io/info/intro.html).

## D1 — Tarefa do MVP: regressão do BIS + estágio derivado

**Escolha:** a CNN prevê um valor contínuo entre 0 e 100. A interface deriva quatro estágios: `deep`, `general`, `light`, `awake`.

**Motivo:** o dataset público pareia EEG com BIS a cada 5 s; os trabalhos que usam esse corpus descrevem as faixas 0–40, 40–60, 60–80 e 80–100. Treinar somente uma classe perderia informação ordinal e dificultaria medir erro próximo às fronteiras.

**Cuidado:** BIS é um índice processado e dependente do monitor, não uma verdade clínica universal. A literatura também documenta artefatos, atrasos, influência de drogas e leituras incompatíveis com o estado clínico. Por isso o produto mostra “referência BIS estimada” e qualidade do sinal, não “consciência do paciente”.

**Referências:** [dataset Figshare](https://doi.org/10.6084/m9.figshare.5589841.v1), [Li et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9160818/), [Hajat et al.](https://pubmed.ncbi.nlm.nih.gov/28044337/), [FDA K202621](https://www.accessdata.fda.gov/cdrh_docs/pdf20/K202621.pdf).

## D2 — Janela de 5 s e EEG a 128 Hz

**Escolha:** 640 amostras por janela, sem sobreposição no dataset de treino; no replay, uma nova previsão pode ser emitida a cada 1 s. O alinhamento é exposto como `label_offset_seconds`; o padrão `0` associa a janela ao ponto BIS correspondente ao seu início, e um valor positivo escolhe um ponto BIS posterior.

**Motivo:** a fonte pública é descrita com EEG frontal de 128 Hz e BIS a cada 5 s. O passo de 1 s no replay é apenas uma decisão de responsividade da interface, não aumenta a resolução do rótulo BIS.

**Risco:** monitores processados podem ter atraso e suavização. O experimento deve registrar explicitamente se alinha o rótulo ao início, centro ou fim da janela; o MVP usa `label_offset_seconds=0` e o ponto publicado correspondente ao início de cada bloco. O deslocamento efetivo é quantizado pelo intervalo do BIS e deve ser pré-especificado antes de comparar modelos.

## D3 — Pré-processamento explícito e compartilhado

**Escolha:** remoção/controle de amplitude, passa-banda 0,5–45 Hz, notch opcional de 50 Hz, escala de amplitude documentada e tratamento de não finitos. Treino e replay usam o mesmo filtro causal com estado contínuo; a função `preprocess_window` mantém uma rota zero-phase apenas para análise offline explícita.

**Motivo:** a faixa preserva bandas EEG usuais e evita depender de decomposição proprietária. Filtragem zero-phase em uma avaliação declarada como online teria acesso a amostras futuras; por isso o caminho usado pelo modelo mantém estado do filtro entre chunks e por caso.

**Referência de implementação:** [`scipy.signal.sosfiltfilt`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).

## D4 — CNN 1-D como baseline, não como arquitetura final

**Escolha:** quatro blocos convolucionais leves e pooling adaptativo, com saída de regressão limitada a 0–100.

**Motivo:** trabalhos de DoA já exploraram CNN e sistemas compactos de tempo real; começar pelo sinal cru torna o baseline auditável e reduz decisões de features manuais. Depois devemos comparar com espectrograma, potência por banda, entropia, TCN e modelos multitarefa.

**Referências:** [AnesNET/real-time CNN](https://pubmed.ncbi.nlm.nih.gov/32746339/), [DRSN](https://pubmed.ncbi.nlm.nih.gov/36679805/), [CNN + EEMD](https://doi.org/10.3934/mbe.2021257).

## D5 — Divisão por caso

**Escolha:** todas as janelas de um caso entram em exatamente um conjunto.

**Motivo:** janelas de 5 s adjacentes são altamente correlacionadas. Uma divisão aleatória por janela pode produzir números excelentes sem provar generalização para um paciente novo.

**Métrica mínima:** MAE e RMSE do BIS, viés, correlação e acurácia/F1 das quatro faixas. Sempre reportar número de casos, número de janelas, distribuição de estágios e intervalo de confiança em experimentos posteriores.

Para análises de sensibilidade, a baseline também oferece cinco folds agrupados por caso. A CNN e a baseline só devem ser comparadas quando usarem o mesmo protocolo de folds, sementes e pré-processamento.

## D6 — Replay antes de hardware clínico

**Escolha:** o primeiro modo “tempo real” reproduz casos gravados usando um buffer, com timestamp, valor bruto, valor suavizado e qualidade.

**Motivo:** permite medir latência e depurar alinhamento sem conectar um equipamento de aquisição. Um adaptador futuro deve definir formato, taxa, sincronização, estado do filtro, falhas de eletrodo e política de perda de dados.

## D7 — Segurança e limites de escopo

O software não controla bombas, não recomenda dose e não cria alarmes clínicos. Antes de qualquer estudo prospectivo, precisamos de revisão de anestesiologista, protocolo de ética, anonimização, análise de riscos, plano de incidentes, validação externa e conformidade regulatória. A interface mantém um aviso persistente para não confundir demo de ML com monitor aprovado.

## D8 — Entrada ao vivo por LSL e JSON

**Escolha:** o núcleo de inferência recebe chunks; a primeira integração de rede usa LSL, e um modo JSON pela entrada padrão permite conectar um driver proprietário sem acoplar o projeto a uma marca.

**Motivo:** LSL foi projetado para descoberta de streams, timestamps por amostra, sincronização e acesso quase em tempo real em pesquisa. Isso separa aquisição de sinal, modelo e interface. O JSON é um contrato mínimo para testes e bridges locais.

**Limite:** LSL não garante que o canal é EEG válido, nem que o monitor está autorizado para uso clínico. O adaptador mantém sobreposição/estado do resampler entre chunks e segura a cauda do filtro; ainda assim, a taxa declarada, a latência, a fidelidade e a política de perda/reconexão precisam ser validadas no hardware real.

**Referência:** [LSL introduction](https://labstreaminglayer.readthedocs.io/info/intro.html) e [artigo de sincronização multimodal](https://pmc.ncbi.nlm.nih.gov/articles/PMC12434378/).

## D9 — Baseline espectral antes de otimizar a CNN

**Escolha:** comparar a CNN com uma regressão Random Forest sobre potência absoluta/relativa por banda, frequência de borda de 90%, entropia espectral, RMS e line length.

**Motivo:** bandas, frequência de borda e entropia são descritores auditáveis frequentemente usados em EEG anestésico. O baseline responde se a complexidade da CNN agrega valor, sem alegar que Random Forest é um sistema clínico.

**Comando:** `uv run brainsniffer benchmark-baseline`.

## D10 — Qualidade e abstenção

**Escolha atual:** cada janela e predição carrega um score heurístico de qualidade. O limiar padrão 0,2 é aplicado no treino, baseline e inferência: abaixo dele, a janela é excluída do conjunto ou o estimador se abstém (`stage="abstain"`) e não produz BIS visível. O score não é chamado de confiança e não aciona conduta. A auditoria local encontrou o `case24` em escala incompatível (0–4095, 99,87% fora do limite observado nos demais casos); ele fica fora por esse gate, sem conversão inventada.

**Motivo:** um modelo pode produzir número mesmo quando a entrada está saturada ou plana. O score atual é apenas uma barreira conservadora de engenharia; o próximo ciclo deve definir uma política de abstention calibrada, com rótulos de SQI e taxa de falsos alertas, antes de exibir qualquer semáforo clínico.

No caminho online, `NaN` e `Inf` são rejeitados antes de avançar o estado do
filtro/resampler. A imputação existente é restrita à análise offline do dataset;
misturar os dois comportamentos poderia esconder uma falha de aquisição e manter
um valor de modelo enganoso.

## D11 — Benchmark de latência

**Escolha:** medir p50/p95 do processamento de chunks e da emissão de uma predição, comparando o p95 ao intervalo de emissão.

**Motivo:** “tempo real” é um requisito mensurável, não uma impressão visual da interface. O benchmark usa sinais sintéticos e o mesmo buffer causal, mas não substitui medida no equipamento, sistema operacional e carga de um centro cirúrgico.

**Comando:** `uv run brainsniffer benchmark-latency --iterations 30`.

## D12 — Validação cruzada agrupada

**Escolha:** a baseline pode rodar `--folds 5`, mantendo todos os segmentos de cada caso no mesmo fold.

**Motivo:** uma única partição pode depender demais de quais pacientes caíram no teste. Folds agrupados expõem a variação entre pacientes, mas não substituem validação externa em outro centro, aparelho ou população.

**Comando:** `uv run brainsniffer benchmark-baseline --folds 5`.

## D13 — Integridade do download

**Escolha:** o downloader confere tamanho e MD5 informado pelo manifesto oficial do Figshare antes de aceitar um arquivo. Um arquivo local inconsistente não é reutilizado silenciosamente; o operador precisa solicitar `--overwrite` para baixá-lo novamente.

**Motivo:** um arquivo truncado ou alterado pode produzir resultados científicos aparentemente válidos, mas não reproduzíveis. O checksum é uma verificação de integridade do transporte, não uma assinatura de autenticidade nem uma validação do conteúdo fisiológico.

**Referência:** [API oficial do artigo no Figshare](https://api.figshare.com/v2/articles/5589841).

## D14 — Segundo corpus para validação externa

**Escolha:** adicionar um cliente seletivo para o VitalDB Open Dataset, baixando somente os tracks `BIS/EEG1_WAV` e `BIS/BIS` de casos explicitamente escolhidos e gravando um `.npz` normalizado. O corpus VitalDB não entra automaticamente no treino do Figshare.

**Motivo:** o VitalDB oferece casos cirúrgicos e parâmetros do monitor BIS em escala maior, com ondas EEG a 128 Hz e tracks sincronizados. Isso é uma oportunidade de teste out-of-dataset, mas a escala, o dispositivo, os termos de uso, os intervalos do BIS e a população diferem do corpus inicial. Misturar as fontes antes de definir o protocolo esconderia justamente o efeito de domínio que precisamos medir.

**Limite de dados:** o download seletivo é intencional; não devemos baixar ou redistribuir o corpus completo sem conferir a licença CC BY-NC-SA 4.0 e os termos vigentes do provedor.

O catálogo oficial declara `BIS/EEG1_WAV` em µV a 128 Hz. O normalizador grava
essa proveniência no `.npz`, mas não transforma automaticamente a amplitude para
parecer com o Figshare: os arquivos observados têm caudas muito maiores que a
faixa central. A compatibilidade de ganho, referência, montagem e artefatos deve
ser demonstrada antes de interpretar a avaliação como comparação entre aparelhos.

**Referências:** [VitalDB Open Dataset — visão geral](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md), [API oficial de tracks](https://vitaldb.net/docs/?documentId=API/Web_API_OpenDataset.md) e [acordo de registro/uso](https://vitaldb.net/registration-agreement/).

## Decisões em aberto

- Expandir a validação externa para mais casos, centros e aparelhos, respeitando licença e variáveis disponíveis.
- Definir uma referência clínica complementar ao BIS: resposta a comando, anestesista, MAC, farmacocinética ou MOAA/S conforme o cenário.
- Medir atraso de rótulo e escolher um alinhamento pré-registrado.
- Validar SQI com anotação ou sinal de qualidade do monitor, em vez da heurística atual.
- Comparar modelos sob split por paciente e teste temporal/out-of-distribution.
- Definir requisito de latência, hardware e tratamento de abstention quando a qualidade for baixa.

## D15 — Avanço por gates e modo sombra

**Escolha:** não conectar o protótipo diretamente à rotina assistencial. O avanço
será dividido em bancada sem paciente, validação externa com checkpoint travado,
modo sombra prospectivo e somente depois uma avaliação clínica inicial desenhada
com a equipe responsável.

**Motivo:** desempenho retrospectivo não demonstra segurança, utilidade ou
adequação do conjunto software–hardware–usuário. Os gates preservam a
reprodutibilidade e permitem detectar mudança de domínio, falhas de aquisição e
problemas de fatores humanos antes de qualquer saída influenciar o cuidado.

**Referências:** [GMLP FDA/Health Canada/MHRA](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles), [DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) e [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378). O protocolo operacional está em [`docs/prospective_protocol.md`](prospective_protocol.md).

## D16 — Manifesto do sinal e invariância da taxa

**Escolha:** cada sessão de aquisição deve registrar unidade, posição do canal,
referência, montagem e origem; quando a taxa também aparecer no manifesto, ela
deve coincidir com a taxa efetiva dos chunks. O descritor XML LSL é aproveitado
automaticamente, e `--require-metadata` transforma a ausência dos campos mínimos
em bloqueio antes da inferência.

**Motivo:** filtro, resampling, janela e escala dependem da taxa e da montagem.
Aceitar uma declaração divergente cria uma saída numericamente plausível, mas
fisicamente ambígua. Separar transporte, metadados e algoritmo torna a sessão
reproduzível e deixa explícito o que ainda precisa ser confirmado no equipamento.

**Referências:** [introdução oficial do LSL](https://labstreaminglayer.readthedocs.io/info/intro.html), [GMLP FDA/Health Canada/MHRA](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles) e o [protocolo de avanço](prospective_protocol.md).

## D17 — Ficha técnica como gate de onboarding

**Escolha:** validar, antes do primeiro stream de um equipamento, um manifesto
versionado com fabricante, modelo, firmware/software, bridge, taxa nominal,
unidade em microvolt, canal, referência, montagem, faixa nominal/saturação e
processamento/ganho aplicado. O comando `validate-intake` não processa
EEG; `--require-intake` aplica a mesma exigência ao preflight e aos streams JSON/LSL.

**Motivo:** o modelo foi treinado em uma combinação específica de taxa, escala,
canal e montagem. Um número plausível produzido por uma fonte não identificada
não é evidência de compatibilidade. Separar a ficha de equipamento do relatório
de aquisição deixa o onboarding auditável e permite bloquear a inferência antes
de avançar o filtro quando a configuração essencial é desconhecida.

**Limite:** `ready_for_bench` significa somente que a documentação mínima está
completa. Ainda são obrigatórios o sinal sintético, o teste de perdas e
reconexões, a comparação de distribuição, a medição ponta a ponta e os gates
éticos/regulatórios do protocolo prospectivo.

**Referências:** [GMLP FDA/Health Canada/MHRA](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles), [DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) e o [protocolo de avanço](prospective_protocol.md).

## D19 — Sensibilidade de alinhamento antes de fixar o rótulo

**Escolha:** manter o checkpoint e a divisão por caso congelados e avaliar uma
grade explícita de `label_offset_seconds` no mesmo holdout, sem retreinamento.
O comando `evaluate-offset` salva os offsets, os casos, o manifesto SHA-256 e as
métricas em um relatório JSON.

**Motivo:** o BIS é uma referência processada e amostrada em intervalos longos;
comparar sempre a janela EEG ao ponto zero pode misturar erro de sincronização
com erro do modelo. Tornar o offset uma análise separada expõe essa hipótese sem
escolher pesos ou casos depois da avaliação.

**Limite:** a grade atual é exploratória e pós-hoc. O fato de offsets positivos
melhorarem este holdout não demonstra que o monitor tenha exatamente esse atraso,
nem autoriza usar o melhor valor em pacientes. O próximo estudo deve pré-registrar
o alinhamento, verificar relógios e repetir em dados independentes.

**Referências:** [Hajat, Ahmad & Andrzejowski (2017)](https://pubmed.ncbi.nlm.nih.gov/28044337/), [Li et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9160818/) e o [protocolo de avanço](prospective_protocol.md).

## D18 — Auditoria fail-closed durante o stream

**Escolha:** quando `--fail-on-audit` está ativo, o stream JSON/LSL reavalia a
auditoria depois de cada chunk e interrompe antes do resampler e do modelo ao
detectar amostras inválidas, lacuna temporal, timestamp inválido, taxa
inconsistente, qualidade abaixo do gate ou outra condição estrutural. A decisão
também é repetida no encerramento e o relatório parcial é preservado.

**Motivo:** verificar somente no fim de uma captura permitiria emitir predições
depois de uma falha já observada. Interromper antes de avançar o estado causal
reduz a chance de transformar uma perda de aquisição em uma saída aparentemente
válida. O modo sem a flag permanece disponível para caracterização exploratória.

**Limite:** essa barreira é de integridade do software e do stream; não é um
alarme clínico, não estima risco anestésico e não substitui testes de perda,
reconexão e latência ponta a ponta no hardware.

**Referências:** [GMLP FDA/Health Canada/MHRA](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles), [DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) e o [protocolo de avanço](prospective_protocol.md).

## D20 — Incerteza reamostrada por cirurgia

**Escolha:** o comando `evaluate` calcula, quando há pelo menos dois casos de
teste, intervalos exploratórios reamostrando casos inteiros. O relatório salva
`bootstrap_samples`, `bootstrap_seed` e os intervalos para as métricas contínuas
e por estágio; não reamostra janelas isoladas como se fossem pacientes
independentes.

**Motivo:** as janelas adjacentes de uma gravação longitudinal compartilham
estado fisiológico, aquisição e rótulo temporal. Reamostrar a unidade da
cirurgia expõe melhor a variabilidade entre pacientes, embora não corrija viés
de centro, aparelho, droga, atraso ou seleção do corpus.

**Limite:** o bootstrap do holdout atual usa somente cinco casos e é
exploratório; não é intervalo clínico nem substitui um plano estatístico
pré-especificado. A decisão deve ser repetida em uma coorte externa maior e
reportada junto com o número de pacientes, não apenas com o número de janelas.

**Referências:** [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378),
[DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) e a seção de
avaliação por caso em [`docs/prospective_protocol.md`](prospective_protocol.md).

## D21 — Expiração explícita de uma estimativa após silêncio

**Escolha:** o consumidor LSL tem `--stale-timeout` (padrão de engenharia: 2 s).
Se nenhum chunk chega nesse intervalo, `--fail-on-audit` encerra a sessão com
erro e preserva o relatório parcial. Sem essa flag, o estimador emite uma única
saída `ABSTAIN` com qualidade zero, limpa a suavização e exige uma janela nova
antes de voltar a estimar.

**Motivo:** ausência de dados é diferente de uma janela recebida com baixa
qualidade. Em ambos os casos a última estimativa não pode continuar visível como
se fosse atual; além disso, misturar amostras antes e depois de uma perda no
mesmo buffer atravessa uma lacuna física não observada.

**Limite:** o timeout é um critério de engenharia e deve ser medido novamente
com o bridge e o hardware alvo. Ele não detecta sozinho uma conexão que continua
transmitindo uma forma de onda errada ou congelada; a auditoria de timestamps,
finitude, saturação e linha plana continua necessária.

**Referências:** [Lab Streaming Layer — introdução oficial](https://labstreaminglayer.readthedocs.io/info/intro.html),
[GMLP FDA/Health Canada/MHRA](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)
e [`docs/real_eeg_intake.md`](real_eeg_intake.md).

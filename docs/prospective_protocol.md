# Protocolo de avanço: bancada, validação externa e modo sombra

Este documento é um plano de pesquisa para levar o BrainSniffer do replay para
uma aquisição EEG real. Ele não é protocolo assistencial, não autoriza pesquisa
em seres humanos e não define uma dose, alarme ou limiar clínico. Qualquer etapa
com dados de pacientes exige aprovação ética, revisão do anestesiologista,
proteção de dados, autorização do equipamento e a avaliação regulatória aplicável.

O desenho segue a lógica de boas práticas de desenvolvimento de ML para
dispositivos, avaliação clínica inicial centrada em segurança e fatores humanos,
e relato transparente de modelos preditivos: [GMLP FDA/Health Canada/MHRA](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles),
[DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) e
[TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378).

## Estado travado antes de qualquer captura

Registrar em um arquivo de estudo:

- hash do checkpoint, versão do código, ambiente Python e configuração de pré-processamento;
- fabricante/modelo, firmware, SDK ou bridge, taxa medida, unidade, canal,
  referência, montagem, filtros aplicados pelo equipamento e relógio dos timestamps;
- objetivo primário, casos de inclusão/exclusão, alinhamento EEG/BIS e plano de análise;
- responsáveis técnico, clínico e de proteção de dados;
- política de não intervenção: a saída não será mostrada para conduzir anestesia.

O checkpoint deve permanecer congelado durante cada conjunto de avaliação. Uma
mudança de peso, pré-processamento, alinhamento, limiar ou seleção de casos abre
uma nova versão e um novo protocolo de análise.

Na entrada ao vivo, validar primeiro a ficha com `validate-intake` e executar o
gate explícito `--require-metadata --require-intake` (ou manter as opções
equivalentes marcadas na interface) depois de preencher fabricante, modelo,
firmware/software, bridge, taxa, unidade em microvolt, faixa nominal, processamento/
ganho, posição do canal, referência e montagem. A ausência desses campos pode ser tolerada apenas em caracterização
técnica, nunca como evidência suficiente para um estudo com paciente.

## Etapa 0 — bancada sem paciente

Usar um publisher LSL ou bridge JSON com sinal sintético conhecido e, quando for
seguro e permitido, uma entrada de teste do fabricante. Executar:

1. cinco minutos de sinal com timestamps por amostra;
2. taxa declarada e taxa medida, inclusive uma fonte diferente de 128 Hz;
3. sinal saturado, linha plana, ruído e amostras ausentes;
4. perda de chunks, pausa, reconexão e encerramento;
5. `audit-json`, `stream-json --report --fail-on-audit` e o benchmark no host que
   será usado.

O gate é técnico, não clínico. Com `--fail-on-audit`, rejeitar a sessão durante a
captura se houver taxa desconhecida ou instável, unidade/montagem não documentada,
timestamps não finitos ou regressivos,
amostras não finitas aceitas pelo caminho online, reconexão silenciosa, previsão
stale após abstenção ou relatório que não identifique o checkpoint. Medir p50 e
p95 ponta a ponta, incluindo aquisição, bridge, resampling, inferência e
visualização; o benchmark interno do projeto mede somente a inferência.

## Etapa 1 — validação retrospectiva externa travada

Usar casos que não tenham participado do treino nem da escolha de hiperparâmetros.
O VitalDB já fornece um piloto exploratório de 15 casos compatíveis (1–10, 12–14,
16 e 17); ele ainda deve ser ampliado e pré-especificado antes de qualquer
conclusão. Para cada caso, conservar a proveniência de
`BIS/EEG1_WAV`, `BIS/BIS`, unidade, taxa, montagem disponível e exclusões.

Relatar, sem tratar janelas adjacentes como pacientes independentes:

- MAE, RMSE, viés e correlação do valor contínuo;
- acurácia e macro-F1 das faixas apenas como análise derivada;
- número de casos, janelas, duração, dados faltantes e fração de abstenção;
- métricas por caso, por qualidade e por distribuição de referência;
- bootstrap agrupado por caso e intervalo de incerteza;
- análise de erro e de discordância com o BIS, sem chamá-la de consciência medida.

Não adaptar escala, ganho, referência ou montagem para melhorar o resultado sem
pré-especificar a transformação e avaliá-la em dados separados. Se a unidade ou
o canal não forem comparáveis, declarar incompatibilidade em vez de converter
arbitrariamente.

## Etapa 2 — modo sombra prospectivo

Depois das aprovações, capturar o EEG e o BIS durante o cuidado usual, mas manter
a saída do BrainSniffer oculta para a equipe e sem qualquer conexão com bomba,
alarme ou decisão. O sistema deve registrar somente o necessário, de preferência
com IDs de estudo, e permitir auditoria posterior de:

- integridade do sinal, perdas, timestamps, latência e reconexões;
- valor bruto, valor suavizado, qualidade, abstenções e versão do modelo;
- BIS publicado, eventos de aquisição e anotações clínicas autorizadas;
- incidentes de privacidade, falhas de software e discrepâncias relevantes.

O objetivo desta etapa é descobrir mudança de domínio, problemas de usabilidade,
atraso e falhas de aquisição. Ela não mede benefício clínico. Qualquer alteração
do cuidado por causa de uma saída acidental deve ser tratada como incidente e
interromper a sessão para revisão.

## Etapa 3 — avaliação clínica inicial

Somente após a análise do modo sombra, revisão de segurança e decisão formal do
grupo clínico pode ser desenhada uma avaliação clínica inicial. O plano deve
descrever usuários, contexto, treinamento, fatores humanos, como discrepâncias
serão apresentadas e como o operador poderá ignorar a saída. O resultado deve
ser avaliado como uma intervenção de software + hardware + usuário, conforme a
orientação do DECIDE-AI, e reportado com o checklist TRIPOD+AI.

Não avançar se o modelo depender de um canal, aparelho, fármaco, população ou
alinhamento diferente do que foi validado. Não substituir o BIS ou o julgamento
do anestesiologista por uma classe da CNN. Uma eventual indicação regulatória é
um projeto separado, com análise de risco, verificação, validação e requisitos
de qualidade próprios.

## Critérios de parada e registro

Parar a captura e preservar o relatório parcial quando ocorrer qualquer um dos
seguintes eventos: unidade desconhecida, canal/montagem divergente, timestamps
inválidos, perda não explicada, saída stale, falha de anonimização, erro que possa
ocultar uma abstenção ou tentativa de usar a saída para comandar tratamento.
Retomar somente após revisão documentada; não apagar a sessão com erro.

Os comandos e a ficha de hardware estão em [`live_acquisition.md`](live_acquisition.md)
e [`real_eeg_intake.md`](real_eeg_intake.md). O registro de decisões científicas
fica em [`decisions.md`](decisions.md), e o estado verificável do protótipo em
[`project_status.md`](project_status.md).

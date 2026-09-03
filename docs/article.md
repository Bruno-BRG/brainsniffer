# BrainSniffer: um baseline reproduzível para estimativa de profundidade anestésica a partir de EEG

> Nota de versão: este arquivo permanece como rascunho técnico de apoio. A versão revisada para entrega é [`tcc_brainsniffer.tex`](tcc_brainsniffer.tex), com [`PDF final`](../output/pdf/brainsniffer_tcc_sbc.pdf), dez páginas, quatro figuras e citações nos parágrafos do artigo.

**Tipo:** rascunho de artigo técnico; os números abaixo são uma execução de engenharia reproduzível, não uma validação clínica.

## Resumo

Este trabalho apresenta o BrainSniffer, um protótipo aberto para pesquisa de estimativa de profundidade anestésica a partir de EEG frontal. O sistema lê o dataset público *EEG and BIS raw data*, divide o sinal em janelas de 5 s, aplica pré-processamento reprodutível e treina uma rede neural convolucional unidimensional para estimar um índice contínuo semelhante ao BIS. O mesmo valor é convertido em quatro estágios de pesquisa: profundo, anestesia geral, sedação leve e acordado. A avaliação é feita por separação de casos, evitando que segmentos do mesmo paciente apareçam em treino e teste. A interface Streamlit permite baixar os dados, explorar o sinal, executar o treino e reproduzir a inferência como fluxo. O sistema é um instrumento de engenharia e não uma decisão clínica: o BIS é tratado como rótulo de referência do monitor, sujeito a atrasos, artefatos e limitações farmacológicas. A execução documentada reporta MAE, RMSE, viés, correlação, F1 por estágio, latência e comportamento sob baixa qualidade de sinal, sempre como evidência preliminar.

**Palavras-chave:** EEG; anestesia; BIS; aprendizagem profunda; CNN; inferência em tempo real; validação por paciente.

## 1. Introdução

Avaliar o componente hipnótico da anestesia é difícil porque o EEG muda com o fármaco, a idade, a doença, a estimulação cirúrgica, artefatos musculares e o próprio sistema de aquisição. Monitores processados, como BIS, oferecem uma referência operacional útil, mas não devem ser interpretados como leitura infalível de consciência. A motivação do BrainSniffer é criar um pipeline transparente que possa ser reproduzido, auditado e comparado com baselines, mantendo explícita a distância entre estimar um índice de monitor e afirmar profundidade anestésica clínica.

Trabalhos anteriores investigaram CNNs, redes híbridas e sistemas de baixa latência para DoA. O presente baseline escolhe uma CNN 1-D sobre EEG cru para reduzir o número de decisões manuais na primeira etapa. O foco metodológico principal é a separação por paciente e a exposição de qualidade do sinal, duas condições necessárias antes de discutir desempenho.

## 2. Materiais e métodos

### 2.1 Dataset

O dataset de Ma contém 24 casos públicos em arquivos MATLAB e é disponibilizado no Figshare sob CC BY 4.0. A descrição associada ao corpus informa EEG frontal amostrado a 128 Hz e valores BIS a cada 5 s. No carregamento, os casos são identificados somente pelo nome anônimo do arquivo; o pipeline não solicita nem armazena identificadores diretos. Uma auditoria do arquivo bruto encontrou o `case24` em escala 0–4095, com 99,87% das amostras fora do limite de ±62,5 usado pelos demais casos; como não há metadado confiável para converter essa escala, o gate padrão de qualidade 0,20 exclui suas janelas do treino. O modo `--min-quality 0` fica reservado à auditoria exploratória.

### 2.2 Alinhamento e rótulos

Cada janela de 640 amostras é pareada ao valor BIS correspondente ao início da
janela (`label_offset_seconds=0` no checkpoint atual). O pipeline permite registrar
explicitamente um deslocamento em segundos; como a referência é amostrada em
intervalos discretos, o índice efetivo é quantizado pelo intervalo do caso. Valores
não finitos ou fora de 0–100 são excluídos da construção de exemplos, assim como
janelas abaixo do score heurístico mínimo 0,20. Para análise discreta, adotamos
intervalos semiabertos `[0,40)`, `[40,60)`, `[60,80)` e `[80,100]`. O valor contínuo
permanece disponível para evitar que a discretização oculte erros próximos às
fronteiras.

### 2.3 Pré-processamento

Cada janela é limitada a uma faixa de segurança, filtrada entre 0,5 e 45 Hz e submetida a notch de 50 Hz. A escala é registrada na configuração do checkpoint. Tanto o treino quanto o replay usam o `StreamingPreprocessor` causal, com estado carregado entre chunks e por caso; uma função zero-phase permanece disponível somente para análise offline explícita. Assim, o checkpoint demonstrativo não depende de amostras futuras no caminho online.

### 2.4 Modelo

A entrada tem forma `(batch, 1, 640)`. A CNN possui quatro convoluções com normalização, GELU e pooling, seguida por pooling adaptativo e duas camadas densas. A saída usa sigmoide escalada para 0–100. A perda inicial é Smooth L1. A arquitetura é deliberadamente compacta; não é apresentada como superior a TCN, atenção, decomposição tempo-frequência ou redes multitarefa.

### 2.5 Avaliação

Casos, não janelas, são embaralhados e divididos em treino, validação e teste. O protocolo deve registrar a semente, os IDs anônimos dos casos em cada partição, o número de janelas e a distribuição dos estágios. As métricas primárias são MAE e RMSE do BIS; as secundárias são viés, correlação, acurácia e macro-F1 dos estágios. A avaliação externa em pacientes e centros não é substituída por validação cruzada em segmentos do mesmo caso.

Como controle de complexidade, o repositório inclui uma baseline espectral com potência absoluta e relativa nas bandas delta, theta, alpha, beta e gamma, frequência de borda de 90%, entropia espectral, RMS e line length, seguida por Random Forest. A comparação usa exatamente as mesmas partições por caso da CNN.

### 2.6 Inferência em fluxo

O replay mantém uma janela circular de 5 s e emite uma estimativa a cada 1 s por padrão. Um EWMA estabiliza a visualização, mas os valores bruto e suavizado são mantidos. A qualidade atual é uma heurística de finitude, saturação e linha plana e deve ser substituída por um SQI validado. O sistema não emite dose nem aciona equipamento.

Para aquisição de pesquisa, o mesmo buffer pode receber chunks via LSL, preservando timestamps e selecionando um canal. O adaptador também aceita bridges JSON pela entrada padrão. Em taxas diferentes de 128 Hz, o adaptador usa resampling polifásico com sobreposição/estado entre chunks e atraso da cauda do filtro; a versão experimental não deve ser descrita como aquisição clínica até que taxa, estado, latência e fidelidade sejam validados no hardware.

## 3. Resultados da execução de engenharia

Executar:

```bash
uv run brainsniffer train --epochs 10 --min-quality 0.2 --checkpoint models/brainsniffer_cnn.pt
```

O protocolo e os números do corpus interno abaixo são um ponto de partida
reprodutível. Eles ainda não representam validação externa definitiva, clínica ou
regulatória:

| Item | Resultado | Protocolo |
|---|---:|---|
| Casos treino/validação/teste | 13 / 5 / 5 | split por caso, seed 42 |
| Janelas treino/validação/teste | 15.894 / 9.638 / 5.523 | 5 s, 128 Hz, filtro causal, qualidade ≥ 0,20 |
| CNN — MAE / RMSE | 7,03 / 11,08 | BIS contínuo, teste |
| CNN — viés / correlação | 0,79 / 0,784 | somente teste |
| CNN — acurácia / macro-F1 | 0,585 / 0,548 | quatro estágios, teste |
| Baseline espectral — MAE / RMSE | 8,86 / 13,23 | mesma partição da CNN |
| Baseline espectral — acurácia / macro-F1 | 0,540 / 0,480 | mesma partição da CNN |
| Streaming sintético — p50 / p95 | 0,57 / 0,79 ms | 30 iterações, CPU local, executado em 2026-09-02 |
| Falhas sob baixa qualidade | ABSTAIN | heurística com limiar padrão 0,2; sem calibração clínica |

Não inserir números produzidos por divisão aleatória de janelas como se fossem generalização a novos pacientes.

### 3.1 Execução de engenharia verificável (não resultado final)

Como execução de engenharia do pipeline completo, foram usados 23 casos efetivos (24 baixados; `case24` reprovado no gate padrão), 31.055 janelas válidas, seed 42 e dez épocas da CNN causal. O conjunto de teste por caso teve MAE 7,03, RMSE 11,08, correlação de Pearson 0,784, acurácia de estágio 0,585 e macro-F1 0,548. A baseline espectral, usando a mesma partição e janelas causais, teve MAE 8,86, RMSE 13,23, correlação 0,667, acurácia 0,540 e macro-F1 0,480. A execução reproduzível do holdout está registrada em `reports/figshare_holdout_evaluation.json`, com bootstrap agrupado por cirurgia, hashes dos cinco arquivos e sem sinais brutos.

Esses números demonstram reprodutibilidade de execução, mas não sustentam uma conclusão clínica ou de superioridade geral: não houve ajuste de hiperparâmetros, calibração, teste em outro centro ou comparação com um sistema clínico. A avaliação VitalDB descrita abaixo é exploratória e não substitui uma validação externa pré-especificada. A CNN melhorou algumas métricas contínuas neste split, enquanto a baseline teve acurácia de estágio ligeiramente maior; isso é uma hipótese para investigação, não uma prova. A tabela deve ser substituída por um protocolo experimental pré-registrado antes de qualquer divulgação científica.

No mesmo ambiente local, o benchmark causal com sinal sintético mediu p50 de 0,57 ms e p95 de 0,79 ms na emissão de uma predição, com passo de 1 s. Esse valor representa somente o custo do software nesta máquina; não inclui aquisição, rede, atraso do resampling, visualização, concorrência, sistema operacional em tempo real ou um dispositivo médico. Não deve ser apresentado como latência clínica.

Como análise de sensibilidade da baseline, uma validação cruzada de cinco folds agrupados por caso produziu MAE médio de 7,91 ± 0,79, RMSE de 11,37 ± 0,94, viés de 0,07 ± 2,10, correlação de Pearson de 0,745 ± 0,034, acurácia de estágio de 0,632 ± 0,043 e macro-F1 de 0,518 ± 0,030. Esses valores são da baseline, não da CNN, e não devem ser comparados diretamente com o holdout da CNN como se fossem o mesmo protocolo; servem para mostrar a variabilidade entre grupos de pacientes.

### 3.2 Sensibilidade exploratória ao alinhamento EEG-BIS

O comando `evaluate-offset` reaplica o checkpoint congelado aos mesmos cinco
casos de teste, sem retreinamento, variando somente o instante do rótulo BIS.
Um offset positivo associa a janela EEG a um valor BIS posterior. A execução
registrada em `reports/offset_sensitivity.json` produziu:

| Offset (s) | Janelas | MAE | RMSE | Pearson | Acurácia | Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| −20 | 5.503 | 7,22 | 11,47 | 0,766 | 0,584 | 0,544 |
| −15 | 5.508 | 7,19 | 11,39 | 0,770 | 0,584 | 0,544 |
| −10 | 5.512 | 7,13 | 11,28 | 0,775 | 0,583 | 0,545 |
| −5 | 5.518 | 7,08 | 11,18 | 0,779 | 0,583 | 0,545 |
| 0 | 5.523 | 7,03 | 11,08 | 0,784 | 0,585 | 0,548 |
| +5 | 5.524 | 6,96 | 10,99 | 0,788 | 0,588 | 0,552 |
| +10 | 5.522 | 6,90 | 10,89 | 0,791 | 0,589 | 0,555 |
| +15 | 5.523 | 6,83 | 10,81 | 0,795 | 0,591 | 0,558 |
| +20 | 5.525 | 6,76 | 10,69 | 0,799 | 0,590 | 0,559 |

O ganho gradual ao associar o EEG a BIS posterior é compatível com uma hipótese
de atraso/alinhamento do monitor, mas não prova a magnitude nem a causa do atraso.
Como a grade foi escolhida após observar o checkpoint, ela é pós-hoc e não deve
ser usada para selecionar o offset final. O próximo protocolo deve fixar o
offset antes da análise, documentar os relógios e avaliar a decisão em dados
separados; o resultado não transforma a saída em medição clínica de consciência.

Como verificação exploratória fora do corpus, 15 casos VitalDB compatíveis foram
normalizados sem retreino e avaliados com `evaluate-external`: 1–10, 12–14, 16 e
17. Em 38.730 janelas, a CNN obteve MAE 12,43, correlação 0,024 e macro-F1 0,398;
os resultados por caso foram heterogêneos. A expansão ocorreu depois do primeiro
piloto 1–5, portanto esta análise não é pré-registrada. O caso 11 não continha
simultaneamente os tracks `BIS/EEG1_WAV` e `BIS/BIS` e foi rejeitado pelo
downloader. A evidência é compatível com mudança de domínio entre aparelho/centro
e o Figshare; não é apresentada como validação clínica nem como conclusão sobre o
VitalDB inteiro. A análise detalhada e os termos do corpus estão em
`docs/vitaldb_external_validation.md`.
O bootstrap exploratório de 1.000 reamostragens por caso estimou Pearson 0,023
[−0,126–0,193], reforçando que a incerteza ainda é grande.
Os diagnósticos agregados encontraram amostras não finitas nos 15 arquivos; elas
foram tratadas somente na preparação offline das janelas. No caminho online, o
mesmo tipo de entrada é rejeitado antes do filtro causal, portanto essa avaliação
não demonstra tolerância a perda de amostras em aquisição ao vivo.

O catálogo VitalDB declara a unidade do waveform como µV, e a rotina de
normalização preserva essa proveniência nos novos arquivos normalizados. Ainda
assim, os 15 casos locais exibem caudas de amplitude aproximadamente entre
−1,477×10³ e 1,800×10³ µV; como o Figshare não fornece
no registro usado uma especificação equivalente de ganho/montagem, não fazemos
conversão arbitrária nem atribuímos a diferença exclusivamente ao domínio.

## 4. Limitações

O corpus é pequeno para sustentar uso clínico, possui um único sensor/canal e usa BIS como referência operacional. O BIS pode sofrer com artefatos e diferentes agentes anestésicos; a disponibilidade pública do sinal não resolve viés de centro, equipamento ou população. A arquitetura não modela explicitamente atraso do monitor, drogas, idade, EMG, hemodinâmica ou estímulo cirúrgico. A filtragem e a heurística de qualidade ainda requerem validação específica para aquisição contínua. Por fim, a demonstração de replay não equivale a segurança em um centro cirúrgico.

O modo LSL reduz o acoplamento a um fabricante, mas não é um driver de monitor nem uma garantia de sincronização clínica. Uma fonte pode declarar a taxa errada, perder amostras ou expor um canal que não corresponde ao sensor esperado; o sistema precisa registrar esses eventos e poder se abster.

## 5. Conclusão e próximos experimentos

O BrainSniffer fornece uma base reproduzível para medir, antes de ampliar a complexidade do modelo, se EEG cru contém informação suficiente para reproduzir uma referência BIS sob separação por paciente. Os próximos passos são baseline espectral/entropia, modelo multitarefa ordinal, alinhamento de atraso, validação externa definitiva, calibração/abstenção e estudo prospectivo supervisionado por equipe de anestesiologia.

## Referências selecionadas

- Ma, L. (2017). *EEG and BIS raw data*. Figshare. https://doi.org/10.6084/m9.figshare.5589841.v1
- Li et al. (2022). *Multiscale depth of anaesthesia prediction for surgery using frontal cortex electroencephalography*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9160818/
- Yang et al. (2020). *A Real-Time Depth of Anesthesia Monitoring System Based on Deep Neural Network...*. https://pubmed.ncbi.nlm.nih.gov/32746339/
- Shi et al. (2023). *Estimating the Depth of Anesthesia from EEG Signals Based on a Deep Residual Shrinkage Network*. https://pubmed.ncbi.nlm.nih.gov/36679805/
- Hajat, Ahmad & Andrzejowski (2017). *The role and limitations of EEG-based depth of anaesthesia monitoring*. https://pubmed.ncbi.nlm.nih.gov/28044337/
- *Open Reimplementation of the BIS Algorithms for Depth of Anesthesia* (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9481655/
- Lee et al. (2022). *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9178032/
- VitalDB. *Open Dataset overview, track units and data-use terms*. https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md
- Kothe et al. (2025). *The Lab Streaming Layer for Synchronized Multimodal Recording*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12434378/

# Validação externa com VitalDB

O VitalDB é uma fonte independente de dados perioperatórios. A documentação
oficial descreve o monitor BIS Vista com `BIS/EEG1_WAV`, `BIS/EEG2_WAV` e `BIS/BIS`;
o waveform EEG é amostrado a 128 Hz, os tracks da mesma cirurgia são
sincronizados e a tabela de parâmetros declara a unidade do EEG como µV. O
conjunto tem termos próprios, diferentes do Figshare, portanto esta integração é
deliberadamente seletiva.

## Download de um caso

```bash
uv run brainsniffer download-vitaldb \
  --case 1 \
  --out data/vitaldb
```

Repita `--case` para solicitar mais casos. O comando consulta o índice oficial,
baixa apenas `BIS/EEG1_WAV` e `BIS/BIS`, interpola o BIS numérico para uma grade de
1 segundo relativa ao começo do EEG e salva:

```text
data/vitaldb/vitaldb_case1.npz
```

O mesmo download está disponível na aba **Dados** da interface Streamlit. Informe
os números separados por vírgula, por exemplo `1,2,3`; depois use **Avaliar
VitalDB sem retreino** para ver as métricas por caso. Essa tela é apenas uma
conveniência para pesquisa e repete o mesmo caminho do comando CLI.

O `.npz` não é incluído automaticamente no treino Figshare. Antes de usá-lo,
devemos fechar um protocolo de validação externa: escolher casos sem consultar o
resultado, definir o alinhamento temporal, verificar a unidade/montagem, avaliar
qualidade e comparar as métricas por caso.

A rotina de normalização preserva a proveniência do dataset, a unidade declarada
(`uV`), os nomes dos tracks e seus IDs. Nos 15 arquivos já avaliados, contudo,
foram observadas caudas de amplitude aproximadamente entre −1,477×10³ e
1,800×10³ µV,
enquanto a maior parte do sinal está muito mais próxima de zero. Isso pode
representar artefato, saturação ou característica do caminho de aquisição; não é
seguro convertê-lo arbitrariamente. A heurística atual sinaliza/limita esses
valores, e a compatibilidade de escala/montagem com o Figshare continua sendo uma
questão aberta, mesmo com a unidade nominal documentada.
Arquivos gerados antes desta mudança precisam ser baixados novamente com
`--overwrite` para receber esses campos de proveniência.

Uma primeira avaliação sem retreino pode ser executada assim:

```bash
uv run brainsniffer evaluate-external \
  --checkpoint models/brainsniffer_cnn.pt \
  --case data/vitaldb/vitaldb_case1.npz \
  --case data/vitaldb/vitaldb_case2.npz \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42
```

O resultado é uma medida exploratória out-of-dataset. Ela não deve ser chamada de
validação clínica nem de validação externa definitiva até que casos, exclusões,
unidade, montagem, atraso e protocolo sejam pré-especificados.

Para avaliar todos os arquivos normalizados presentes no diretório sem montar a
lista manualmente, use:

```bash
uv run brainsniffer evaluate-external \
  --data-dir data/vitaldb \
  --checkpoint models/brainsniffer_cnn.pt \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --report reports/vitaldb_external_validation.json
```

O comando considera somente `vitaldb_case*.npz`, ordena os arquivos pelo nome e
inclui cada caminho, tamanho e SHA-256 no resultado. Com `--report`, o mesmo JSON
é salvo pelo programa para acompanhar o estudo. O campo `input_diagnostics`
registra, por caso, a quantidade de amostras não finitas, a fração finita, a
faixa bruta observada e se a preparação offline precisou imputar pontos.
O campo `data_handling` declara a política completa: a avaliação é offline, a
construção de janelas pode fazer interpolação linear e o caminho online rejeita
dados não finitos antes do filtro/resampler; o relatório não contém EEG bruto.

Além das métricas agregadas, a saída inclui uma linha por caso e intervalos
exploratórios obtidos por bootstrap de casos inteiros. Esses intervalos não
tratam janelas adjacentes como observações independentes e não substituem um
plano estatístico revisado. O relatório também registra tamanho e SHA-256 de cada
`.npz` externo avaliado, além de `bootstrap_samples` e `bootstrap_seed`.

## Primeira execução exploratória

Em 2026-09-02, o caso 1 foi baixado e normalizado com 1.477.268 amostras EEG a
128 Hz, 11.542 pontos BIS e 2.302 janelas aceitas pelo gate de qualidade. Sem
retreino ou adaptação, o checkpoint Figshare produziu:

| Métrica | Caso VitalDB 1 |
|---|---:|
| MAE BIS | 15,53 |
| RMSE BIS | 20,12 |
| Viés | −2,34 |
| Pearson | −0,139 |
| Macro-F1 | 0,275 |

Esse resultado é uma falha informativa de generalização, não uma validação
definitiva. A distribuição bruta também difere: no caso observado, os percentis
1–99 foram −27 a 75,2 e a mediana 25,1, com outliers entre −1.474,9 e 1.798,6.
Precisamos conferir unidade, referência, montagem, atraso do monitor e política
de artefatos antes de concluir que a causa é apenas escala. O caso permanece
separado do treino e do artigo principal até um protocolo externo pré-especificado.

### Piloto histórico: VitalDB 1–10

Uma avaliação ampliada dos casos VitalDB 1–10, feita sem retreino ou adaptação,
produziu 23.110 janelas. A inclusão dos casos 6–10 ocorreu depois do primeiro
piloto 1–5; por isso, esta expansão é exploratória e não deve ser descrita como
um protocolo pré-registrado.

| Métrica | Casos 1–10 |
|---|---:|
| Janelas | 23.110 |
| MAE BIS | 12,71 |
| RMSE BIS | 19,17 |
| Viés | 0,56 |
| Pearson | 0,054 |
| Acurácia de faixa | 0,409 |
| Macro-F1 | 0,388 |

Por caso, MAE/Pearson/macro-F1 foram: caso 1 = 15,53/−0,139/0,275; caso 2 =
11,22/0,164/0,375; caso 3 = 18,21/0,205/0,375; caso 4 = 11,22/0,466/0,356;
caso 5 = 9,79/−0,472/0,251; caso 6 = 13,25/0,070/0,305; caso 7 =
11,70/0,013/0,428; caso 8 = 18,47/−0,182/0,223; caso 9 =
21,43/0,038/0,338; caso 10 = 12,39/0,058/0,132. A heterogeneidade reforça
que a média agregada não é suficiente e que o próximo protocolo deve
pré-especificar análise por paciente.

Com 1.000 reamostragens de casos inteiros (seed 42), os intervalos exploratórios
de 95% foram: MAE 12,76 [11,31–14,70], RMSE 19,21 [17,20–22,07], viés 0,66
[−3,83–5,61], Pearson 0,047 [−0,155–0,258], acurácia de faixa 0,407
[0,216–0,600] e macro-F1 0,377 [0,277–0,471]. Mesmo com dez casos, esses
intervalos servem para expor a incerteza e não para sustentar desempenho clínico.

### Atualização exploratória: 15 casos compatíveis

Em 2026-09-02, a avaliação foi ampliada sem retreino ou adaptação para os casos
1–10, 12–14, 16 e 17. O caso 11 não possui simultaneamente `BIS/EEG1_WAV` e
`BIS/BIS` no índice oficial; ele foi rejeitado pelo downloader e não gerou um
arquivo parcial. O conjunto ampliado contém 38.730 janelas e permanece totalmente
fora do treino Figshare.

| Métrica | 15 casos |
|---|---:|
| Janelas | 38.730 |
| MAE BIS | 12,43 |
| RMSE BIS | 18,80 |
| Viés | 1,40 |
| Pearson | 0,024 |
| Acurácia de faixa | 0,420 |
| Macro-F1 | 0,398 |

Com 1.000 reamostragens agrupadas por caso (seed 42), os intervalos exploratórios
de 95% foram: MAE 12,52 [11,05–14,45], RMSE 18,87 [16,51–21,83], viés 1,53
[−2,66–6,60], Pearson 0,023 [−0,126–0,193], acurácia de faixa 0,415
[0,301–0,543] e macro-F1 0,391 [0,324–0,462]. A inclusão foi uma expansão
posterior, não um protocolo pré-registrado; o resultado continua sendo uma
falha informativa de generalização, não validação clínica.

## Termos e segurança

O VitalDB é um dataset para pesquisa com termos de uso e licença diferentes do
corpus inicial. Leia a [visão geral oficial do VitalDB](https://vitaldb.net/docs/?documentId=OpenDataset/Overview.md)
e a documentação da [API de tracks](https://vitaldb.net/docs/?documentId=API/Web_API_OpenDataset.md)
e o [acordo de registro/uso](https://vitaldb.net/registration-agreement/)
antes de baixar, armazenar ou compartilhar dados. O adaptador não solicita PHI,
não tenta reidentificar pacientes e não transforma o dataset em autorização para
uso clínico.

## O que ainda precisa ser medido

- Quantos casos contêm simultaneamente EEG1 e BIS utilizáveis.
- Se a unidade e a referência do EEG são compatíveis com o treino Figshare.
- Atraso, perdas, valores faltantes e qualidade do BIS.
- Desempenho sem retreino e, separadamente, desempenho após protocolo de adaptação.
- Diferenças de aparelho, centro, anestésicos e população.

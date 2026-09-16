# BrainSniffer — nota histórica e guia do artigo

Este Markdown é uma nota de contexto, não um segundo manuscrito. A fonte do TCC
é [`tcc_brainsniffer.tex`](tcc_brainsniffer.tex), com bibliografia em
[`referencias.bib`](referencias.bib) e receita em [`latex/README.md`](latex/README.md).
O estado real da compilação está em [`project_status.md`](project_status.md).

## Contribuição de graduação

O BrainSniffer implementa uma cadeia auditável EEG frontal → referência BIS:
janelas de 5 s a 128 Hz, CNN 1-D compacta, baseline espectral com Random Forest,
separação por grupos, relatórios e replay. O objetivo é comparar essas soluções
nos dados disponíveis, não medir consciência ou orientar anestesia. O BIS é a
saída de um monitor, sujeita a artefatos, fármacos e atrasos.

## Resultados históricos preservados

Métricas agregadas por janela aceita; casos longos têm mais peso. Os checkpoints
não foram retreinados nesta revisão.

| Benchmark | Modelo | Casos | MAE | RMSE | Pearson |
|---|---|---:|---:|---:|---:|
| Figshare | CNN ativa | 5 | 7,03 | 11,08 | 0,784 |
| Figshare | Baseline espectral | 5 | 8,86 | 13,23 | 0,667 |
| Figshare histórico | CNN mista | 5 | 6,69 | 10,67 | 0,818 |
| VitalDB histórico | CNN ativa | 15 | 12,43 | 18,80 | 0,024 |
| VitalDB histórico | CNN mista | 15 | 8,60 | 11,74 | 0,688 |

Fontes: `reports/figshare_holdout_evaluation.json`,
`reports/spectral_baseline_holdout.json`, `reports/mixed_fixed_figshare_holdout.json`,
`reports/vitaldb_external_validation.json` e `reports/mixed_vitaldb_external.json`.
Há 5.523 janelas no Figshare e 38.730 no VitalDB. A CNN supera o baseline apenas
nesta partição/execução; a perda de associação do ativo no VitalDB é um resultado
negativo central. O misto melhora o MAE em 14/15 casos VitalDB, mas piora no caso 16.
Isso é compatível com adaptação ao domínio, não prova de generalização externa.

## Como interpretar

- Figshare: 23 casos elegíveis (caso 24 em quarentena), split ativo 13/5/5,
  seed 42. Não há identificador de pessoa: separar cirurgias não comprova
  independência entre cirurgias da mesma pessoa.
- Misto: 18 Figshare + 10 VitalDB = 28 grupos após excluir o holdout Figshare;
  split 16/6/6. O ajuste usa 12 Figshare + 4 VitalDB; validação 2+4 e teste
  interno 4+2. VitalDB agrupa reoperações conhecidas por participante.
- Os 15 casos VitalDB estão fora do ajuste, mas o candidato já viu outros
  participantes dessa fonte e o benchmark já tinha sido inspecionado. A comparação
  é retrospectiva, histórica e exploratória. Não promover o candidato ao dashboard
  não recupera independência estatística.
- Bootstrap por caso: 1.000 réplicas, seed 42. Para a CNN ativa no Figshare,
  MAE observado 7,03 tem intervalo percentil 95% 6,38–8,24 e Pearson observado
  0,784 tem 0,703–0,881. Cinco casos não se tornam mil casos independentes.
  Pearson mede associação, não acordo ou calibração.
- Offset padrão zero associa o alvo ao início da janela; a emissão ocorre após
  seus 5 s, além do processamento. A grade histórica −20…+20 s é pós-hoc,
  sem retreino, e não autoriza mudar o offset (`reports/offset_sensitivity.json`).
- Filtragem causal não torna toda a preparação offline causal: EEG não finito
  pode ser interpolado com amostras futuras, sem limite de lacuna nessa rotina,
  com extensão dos extremos e zeros se todo o vetor for não finito. Os gates
  de desenvolvimento (incluindo lacuna EEG máxima de 2 s) não equivalem à
  avaliação histórica dos 15 VitalDB nem ao live, que rejeita não finitos.
- Na normalização VitalDB, `_align_numeric_to_eeg` em
  `src/brainsniffer/data/vitaldb.py` ordena e deduplica tempos, interpola BIS
  finito numa grade de 1 s entre o primeiro e último ponto válido e deixa NaN
  fora desse intervalo. Não há limite de lacuna nem máscara de imputação
  persistida. Finitude nessa etapa não significa faixa 0–100; alvos fora dessa
  faixa ou não finitos são descartados depois na construção das janelas.
  A política não foi modificada nem seu efeito reavaliado.

## Limites e próximos passos

A CNN histórica e seu baseline bastam ao escopo do TCC; não são exigidos novas
arquiteturas, ablações extensas ou estudo clínico. Ensaiar o replay e investigar
lacunas são extensões possíveis. Aquisição física, latência ponta a ponta,
acordo/calibração e generalização independente não foram demonstrados.
Medições antigas de latência sintética e de validação cruzada da baseline,
sem rechecagem de relatório bruto nesta revisão, não são usadas como evidência
principal; seu histórico operacional permanece em `project_status.md`.

O protótipo não mede consciência, não recomenda dose e não deve acionar alarme,
conduta ou equipamento. Dados públicos não implicam dispensa ética automática.
Autor e instituição precisam confirmar termos de uso, enquadramento ético,
curso, instituição e cidade/estado. Compilar o artigo não aprova o sistema.

# Mestrado: BrainSniffer Embarcado

Proposta de dissertação para levar a CNN de estimativa de BIS do BrainSniffer a microcontroladores ESP32 e ESP32-S3, medindo a preservação de desempenho.

## Arquivos

| Arquivo | O que é |
|---|---|
| `proposta/proposta_mestrado_brainsniffer_embarcado.docx` | Proposta completa: justificativa, questões e hipóteses, fundamentação, trabalhos relacionados, ponto de partida, metodologia, cronograma, riscos, ética e referências. Inclui anexos com o cálculo de recursos da CNN e um glossário. |
| `apresentacao/apresentacao_mestrado_brainsniffer_embarcado.pptx` | Apresentação de 20 slides em 16:9, com notas do orador em cada slide. |
| `referencias/referencias.bib` | 50 referências em BibTeX, para importar no Zotero, Mendeley ou LaTeX. O campo `keywords` agrupa por tema e o campo `note` marca o que ainda precisa ser conferido. |

## Antes de submeter

- Preencher os campos entre colchetes: `[Instituição]`, `[nome do programa]` e o orientador (`[a definir]`), na proposta e na apresentação.
- Conferir as 8 referências marcadas com `[verificar antes de citar]` (campo `note` no `.bib`).
- Levantar cotações de hardware (seção 15 da proposta).
- Validar com o orientador os limiares de H1 a H4 (seção 6 da proposta) antes de rodar os experimentos.
- Consultar o CEP sobre a necessidade de análise para as bases públicas e sobre a Fase C, que envolve voluntários.

## Origem dos números

- Parâmetros (146.817) e operações por janela (17,85 M MAC) foram contados diretamente na classe `Conv1DDepthEstimator` (`src/brainsniffer/nets/cnn.py`).
- Métricas de desempenho vêm dos relatórios e dos `models/*.json` do repositório, descritos na seção 9 da proposta.

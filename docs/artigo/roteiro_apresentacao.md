# Roteiro de apresentação — versão enxuta

**Slides:** 24 (21 de fala + 3 de referências em anexo) · **Fala:** 13:55 · **Alvo:** 17 min · **Limite:** 20 min

Com respiração, transições e ritmo natural, a fala real tende a ficar entre 15 e 16 min. Se terminar antes do alvo, use a lista "Se sobrar tempo" no fim.

**O que saiu em relação à versão completa** (guardada em `docs/versao_completa/`): o baseline espectral (Random Forest com 14 atributos), o experimento fatorial (quatro retreinos), o slide de ritmos do EEG e as figuras que dependiam deles.

---

## Mapa de tempo

| # | Slide | Tempo | Acumulado |
|---|-------|-------|-----------|
| 1 | Capa | 20 s | 0:20 |
| 2 | Identificação | 15 s | 0:35 |
| 3 | Sumário | 30 s | 1:05 |
| 4 | Problema | 60 s | 2:05 |
| 5 | Perguntas de pesquisa | 40 s | 2:45 |
| 6 | Como o EEG funciona | 55 s | 3:40 |
| 7 | EEG sob anestesia | 50 s | 4:30 |
| 8 | BIS | 50 s | 5:20 |
| 9 | Caso real | 35 s | 5:55 |
| 10 | Dados | 45 s | 6:40 |
| 11 | Pré-processamento | 40 s | 7:20 |
| 12 | A rede | 50 s | 8:10 |
| 13 | Treino e checkpoints | 30 s | 8:40 |
| 14 | Resultado principal | 35 s | 9:15 |
| 15 | Trajetória | 35 s | 9:50 |
| 16 | Comparação | 45 s | 10:35 |
| 17 | VitalDB | 40 s | 11:15 |
| 18 | Implantabilidade | 45 s | 12:00 |
| 19 | Conclusão | 55 s | 12:55 |
| 20 | Limitações | 45 s | 13:40 |
| 21 | Obrigado | 15 s | 13:55 |
| 22–24 | Referências | — | anexo, não passar |

---

## Bloco 1 — Abertura e problema (0:00–2:45)

### 1. Capa — 20 s
> Boa tarde. Meu nome é Bruno Rocha Guimarães e vou apresentar o BrainSniffer: uma CNN compacta e causal para estimar a referência BIS a partir de EEG frontal. É meu trabalho de conclusão de curso em Ciência da Computação, orientado por Igor Gonzalez Pimenta.

### 2. Identificação — 15 s
> Desenvolvido no Centro Universitário Jorge Amado, em Salvador.

Slide de cortesia: vire rápido.

### 3. Sumário — 30 s
> O roteiro é curto: o problema e as três perguntas; o mínimo de EEG e anestesia; os dados, a rede e os resultados; e o que o trabalho entrega e o que não entrega.

Não leia os oito itens.

### 4. Problema — 60 s
> Na anestesia geral o paciente precisa estar hipnotizado o suficiente, sem excesso de fármaco. O índice mais usado à beira do leito é o BIS, mas ele é proprietário: a regra que o calcula não é publicada.
>
> Há ainda um risco metodológico: janelas vizinhas da mesma cirurgia são quase idênticas. Se eu dividir treino e teste ao acaso, o modelo decora a cirurgia e eu meço memória, não generalização.
>
> A lacuna é essa: **um estimador compacto e causal, avaliado por caso e em mais de uma fonte de dados.**

Frase-âncora: *"mede memória, não generalização"*.

### 5. Perguntas — 40 s
> São três perguntas. Primeira: uma CNN 1-D compacta, sob validação agrupada por caso, aproxima o BIS na faixa de variabilidade do próprio monitor? Segunda: ela é competitiva com um modelo publicado cerca de sete vezes maior, mantendo uma cadeia que pode rodar ao vivo? Terceira: o desempenho se mantém ao trocar a fonte, do Figshare para o VitalDB?
>
> As métricas primárias foram definidas antes: MAE por janela, em pontos BIS, e a probabilidade de predição PK.

---

## Bloco 2 — Fundamentos (2:45–5:55)

### 6. Como o EEG é gerado e medido — 55 s
> O EEG mede a atividade elétrica do córtex. Os neurônios piramidais estão alinhados; quando milhões recebem entrada sináptica ao mesmo tempo, os potenciais somam e formam um campo que atravessa meninges, crânio e couro cabeludo.
>
> Ele chega muito atenuado, em dezenas de microvolts, e por isso é amplificado, filtrado e digitalizado. O que interessa aqui: resolução temporal de milissegundos e um único canal frontal a 128 Hz, o mesmo sinal que o monitor BIS usa.

Se estiver apertado, diga só "soma de potenciais, atenuada pelo crânio".

### 7. EEG sob anestesia — 50 s
> Com o aprofundamento, o EEG muda de forma característica. Acordado, predominam ritmos rápidos e de baixa amplitude. Na sedação, há um aumento paradoxal de beta e o sinal começa a lentificar.
>
> Em anestesia geral aparece o alfa frontal em torno de 10 Hz com ondas lentas de grande amplitude. No plano profundo, surto-supressão: rajadas entre trechos quase planos.
>
> Essas quatro faixas são classes que uso para relatar acurácia. **Não são limiares clínicos validados.**

### 8. BIS — 50 s
> O BIS resume o EEG frontal numa escala de 0 a 100: perto de 100 acordado, 40 a 60 anestesia geral, perto de zero o EEG isoelétrico.
>
> Mas tem limites: é proprietário, sofre interferência de músculo, bloqueio neuromuscular e cetamina, e chega suavizado e com atraso.
>
> Por isso o vocabulário é cuidadoso: eu estimo **um rótulo de referência BIS** — não o BIS como verdade, e muito menos consciência.

Frase-âncora: *"um rótulo, não a verdade"*.

---

## Bloco 3 — Dados e método (5:55–8:40)

### 9. Caso real — 35 s
> Este é o caso 19 do holdout. Em cima, duas janelas de 5 segundos: uma limpa, que o pipeline quase não altera, e uma com pico de 125 microvolts, contido para 88. Embaixo, o espectrograma dos 72 minutos com o BIS na mesma base de tempo: quando o BIS sobe, cai o teta e sobem beta e gama. É essa relação que a rede aprende.

Primeiro slide a cortar se o tempo apertar.

### 10. Dados — 45 s
> Duas fontes públicas. O Figshare tem 24 casos anônimos, com EEG frontal e BIS a cada 5 segundos; sem identificador de pessoa, então o grupo é o caso. O VitalDB é muito maior, de Seul, com BIS a cada segundo; ali agrupo pelo identificador do participante.
>
> São 59 arquivos; depois dos gates de qualidade, 33 casos elegíveis e 55.471 janelas. A regra central: **nenhum caso cruza treino, validação e teste.**

### 11. Pré-processamento — 40 s
> O pipeline tem cinco passos: EEG frontal, gates de qualidade, filtro causal de 0,5 a 45 Hz com notch de 50 Hz, corte de amplitude e janelas de 5 segundos, 640 amostras a 128 Hz.
>
> O ponto importante é a causalidade: o filtro usa só passado e presente, com estado por caso. A mesma cadeia roda em replay ao vivo.

Não leia os limiares dos gates.

### 12. A rede — 50 s
> A rede é pequena de propósito. Recebe 640 amostras em um canal e passa por quatro blocos convolucionais, com batch norm e GELU, e pooling que reduz o tempo de 640 para 80. Depois, média global, cabeça densa e uma sigmoide escalada de 0 a 100.
>
> São **0,15 milhão de parâmetros**: roda em CPU.

### 13. Treino e checkpoints — 30 s
> Treino curto: dez épocas. Dois checkpoints: o ativo, treinado só no Figshare, que mede o limite interno e a queda ao mudar de fonte; e o misto, que viu também parte do VitalDB.
>
> E, como o próprio BIS chega suavizado, aplico uma média móvel exponencial causal sobre as predições congeladas, sem retreino: uma recursão de uma linha que só usa o passado.

---

## Bloco 4 — Resultados (8:40–12:00)

### 14. Resultado principal — 35 s
> No holdout Figshare, cinco casos e 5.523 janelas, o checkpoint ativo teve **MAE de 7,03 pontos BIS**, abaixo da escala de leitura de 10 pontos.
>
> A ordenação, medida pelo PK, foi 0,773: sorteados dois momentos, a rede os ordena como o BIS em 77% das vezes. Com cinco casos, é exploratório.
>
> Esses números vêm da avaliação offline. Reexecutei o mesmo checkpoint em replay, em fluxo, nos mesmos cinco casos, com passo de 5 s: as predições bateram até a quinta casa decimal (diferença máxima de 1,5·10⁻⁵). No VitalDB eu não medi o replay.

### 15. Trajetória — 35 s
> Aqui a predição ao longo de toda a cirurgia do caso 19, escolhido antes da inferência. A rede acompanha a queda na indução e a subida no despertar, mas **superestima o patamar do meio**: viés de mais 4,79.
>
> Em cima, a linha contínua é o BIS, a tracejada é a predição bruta e a pontilhada é a suavizada (EWMA, span 10); embaixo, o erro. Entre os minutos 20 e 58 a rede fica em 40–50 e a referência em 25–40. Esta é a inferência offline, posicionada no início da janela: a estimativa só existe 5 s depois.
>
> A correlação dentro do caso é alta, 0,91: associação alta não elimina erro sistemático.

### 16. Comparação — 45 s
> Os dois checkpoints nos dois benchmarks, com intervalos de 95% por bootstrap de casos inteiros. Três dos quatro MAE ficam abaixo de 10 pontos. Esse 10 é uma régua de leitura que tirei de estudos que comparam montagens do próprio BIS; não é a variabilidade geral do monitor nem um critério clínico.
>
> No Figshare, o misto não é significativamente melhor que o ativo.

### 17. VitalDB — 40 s
> A fronteira do trabalho é o transporte. O checkpoint que fica em 7 no Figshare vai a **12,43 no VitalDB**, e o PK cai para 0,515: a rede deixa de ordenar os momentos da cirurgia. Mudam equipamento, montagem e população, e o experimento não separa a causa.
>
> O misto melhora 14 dos 15 casos, mas já tinha visto essa fonte: é adaptação ao domínio, não generalização.

### 18. Implantabilidade — 45 s
> A segunda pergunta era sobre o estado da arte. O melhor resultado publicado neste corpus é um TCN-BiLSTM de 2026, com 1,07 milhão de parâmetros e MAE 6,03 após suavização.
>
> Com a minha suavização causal, o misto chega a 6,11 com **10 segundos de memória**, contra 45 a 95 do deles. E o meu é causal: o deles usa BiLSTM bidirecional e filtro de fase zero, que não existem ao vivo. Não é um ranking, porque os protocolos diferem: é ordem de grandeza.

---

## Bloco 5 — Fechamento (12:00–13:55)

### 19. Conclusão — 55 s
> Respondendo às três perguntas. Um: sim, a rede compacta aproxima o rótulo BIS com erro abaixo da escala de leitura, sob validação por caso, e ordena os momentos da cirurgia. Dois: com a suavização causal ela entra na faixa do melhor modelo publicado, sendo sete vezes menor e rodando ao vivo em CPU. Três: não, o desempenho não transporta para outra fonte; a prioridade passa de "modelo maior" para "dados da fonte".
>
> A contribuição é uma cadeia compacta, causal e auditável para testar hipóteses — não um sistema clínico.

### 20. Limitações — 45 s
> As limitações estão declaradas: estudo retrospectivo e exploratório, com cinco casos de teste; o benchmark VitalDB foi reutilizado; a imputação offline pode usar amostras futuras; o alvo é o próprio BIS; e o EWMA foi escolhido após ver os resultados.
>
> O próximo passo é uma coorte travada de outra fonte, com partição e métricas definidas antes, e treino com mais fontes.
>
> E deixo claro: **o BrainSniffer não mede consciência, não prescreve dose e não deve acionar conduta clínica.**

### 21. Obrigado — 15 s
> Obrigado. Fico à disposição para perguntas. O código está no GitHub, sob licença MIT, sem redistribuir os dados.

---

## Se atrasar — ordem de corte

1. **Slide 9 (Caso real):** mostre a figura e diga só a última frase. **−25 s**
2. **Slide 15 (Trajetória):** só a frase do viés. **−20 s**
3. **Slide 6 (EEG):** "soma de potenciais, atenuada pelo crânio, canal frontal a 128 Hz". **−25 s**
4. **Slide 13 (Treino):** "dez épocas, dois checkpoints, EWMA causal". **−10 s**

Total possível: **−1:20**.

## Se sobrar tempo — onde ganhar

- Slide 17: detalhe o caso 16, o único que piora com o misto.
- Slide 18: explique filtro causal versus filtro de fase zero.
- Slide 14: explique o PK com um exemplo de dois momentos da cirurgia.
- Slide 19: repita a contribuição em uma frase própria.

---

## Perguntas prováveis

**Por que não usar o BIS como verdade absoluta?**
> Porque ele é proprietário, sensível a EMG e cetamina, e chega com atraso. Eu o trato como rótulo de referência, não como medida de consciência.

**Por que o VitalDB piora tanto?**
> Mudança de domínio: outro monitor, outra montagem, outra população e outra taxa de BIS. O experimento não separa qual dessas causas pesa mais.

**O que é PK?**
> Probabilidade de predição: sorteio dois momentos da cirurgia; o PK é a chance de o modelo ordená-los como o BIS. 0,5 é acaso; 1 é ordem perfeita.

**Por que a rede é tão pequena?**
> Porque o objetivo é implantabilidade: 0,15 milhão de parâmetros roda em CPU e permite uma cadeia causal ao vivo.

**Como rodaria ao vivo?**
> Buffer de 5 segundos, atualização a cada segundo, filtro causal com estado por caso e EWMA, a mesma cadeia usada offline.

**Você não comparou com um modelo mais simples?**
> Explorei um controle espectral, mas não o validei com o mesmo rigor da rede, então não o uso como evidência. A comparação que sustenta o trabalho é com o modelo publicado e com o próprio limite de variabilidade do monitor.

**Os números são do sistema ao vivo ou offline?**
> Offline, em janelas de 5 s. No Figshare conferi em replay causal, com passo de 5 s: as predições coincidem até 1,5·10⁻⁵ ponto BIS, então o número é o da cadeia em replay. No VitalDB não medi: lá o offline interpola amostras que o replay rejeitaria, e os números tendem a ser otimistas.

**Por que não um Transformer?**
> Não testei. Com 24 casos e a meta de rodar ao vivo em CPU, a CNN era a escolha mais defensável: tem o viés certo para sinal local e poucos parâmetros. Transformer é extensão natural quando houver mais fontes.

**Isso pode ser usado clinicamente?**
> Não. É uma base auditável para testar hipóteses; não mede consciência, não prescreve dose e não deve acionar alarme.

---

## Checklist de ensaio

- [ ] Cronometrar uma passada completa e anotar o tempo real por bloco.
- [ ] Ensaiar as transições 4 → 5, 8 → 9 e 16 → 17.
- [ ] Decorar só três números: **MAE 7,03**, **PK 0,773** e **12,43 / 0,515**.
- [ ] Testar o controle remoto nos slides 16 e 17 (tabela e figura).
- [ ] Deixar as referências como anexo: só abrir se pedirem a fonte.

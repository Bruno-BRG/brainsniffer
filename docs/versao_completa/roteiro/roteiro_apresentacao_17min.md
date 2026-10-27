# Roteiro de apresentação — alvo 17 min

**Slides:** 26 · **Fala:** 15:30 · **Folga:** 1:30 (transições, tropeços, respiração)
**Regra:** cada slide = uma ideia. Se estiver atrasado, corte pela lista do fim.

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
| 7 | Ritmos | 35 s | 4:15 |
| 8 | EEG sob anestesia | 50 s | 5:05 |
| 9 | BIS | 50 s | 5:55 |
| 10 | Caso real | 35 s | 6:30 |
| 11 | Dados | 45 s | 7:15 |
| 12 | Pré-processamento | 40 s | 7:55 |
| 13 | A rede | 50 s | 8:45 |
| 14 | Treino e controles | 40 s | 9:25 |
| 15 | Resultado principal | 40 s | 10:05 |
| 16 | Trajetória | 35 s | 10:40 |
| 17 | Comparação | 45 s | 11:25 |
| 18 | VitalDB | 45 s | 12:10 |
| 19 | Implantabilidade | 45 s | 12:55 |
| 20 | Fatorial | 35 s | 13:30 |
| 21 | Conclusão | 60 s | 14:30 |
| 22 | Limitações | 45 s | 15:15 |
| 23 | Obrigado | 15 s | 15:30 |
| 24–26 | Referências | — | anexo, não passar |

---

## Bloco 1 — Abertura e problema (0:00–2:45)

### 1. Capa — 20 s
> Boa tarde. Meu nome é Bruno Rocha Guimarães e vou apresentar o BrainSniffer: uma CNN compacta e causal para estimar a referência BIS a partir de EEG frontal. É um trabalho de conclusão de curso em Ciência da Computação, orientado por Igor Gonzalez Pimenta.

Não leia o subtítulo inteiro: o slide já diz.

### 2. Identificação — 15 s
> O trabalho foi desenvolvido no Centro Universitário Jorge Amado, em Salvador.

Slide de cortesia. Vire rápido.

### 3. Sumário — 30 s
> O roteiro é curto. Começo pelo problema e pelas três perguntas; explico o mínimo de EEG e anestesia; mostro os dados, a rede e os resultados; e fecho com o que o trabalho entrega e o que ele não entrega.

Não leia os oito itens. Só aponte o caminho.

### 4. Problema — 60 s
> Na anestesia geral existe um problema prático: o paciente precisa estar hipnotizado o suficiente, sem excesso de fármaco. O índice mais usado à beira do leito é o BIS — mas ele é proprietário, a regra que calcula o número não é publicada.
>
> Há ainda um risco metodológico que passa batido: janelas vizinhas da mesma cirurgia são quase idênticas. Se eu dividir treino e teste ao acaso, o modelo decora a cirurgia e eu meço memória, não generalização.
>
> A lacuna que ataco é essa: **um estimador compacto e causal, avaliado por caso e contra um controle espectral, em mais de uma fonte de dados.**

Frase-âncora: *"mede memória, não generalização"*. É o que a banca vai lembrar.

### 5. Perguntas — 40 s
> São três perguntas.
> Primeiro: uma CNN 1-D compacta, sob validação agrupada por caso, aproxima o rótulo BIS e supera um baseline espectral?
> Segundo: ela é competitiva com um modelo publicado cerca de sete vezes maior, mantendo uma cadeia exportável para uso ao vivo?
> Terceiro: o desempenho se mantém ao trocar a fonte dos dados, do Figshare para o VitalDB?
>
> As métricas primárias foram definidas antes: MAE por janela, em pontos BIS, e a probabilidade de predição PK.

---

## Bloco 2 — Fundamentos (2:45–5:55)

### 6. Como o EEG é gerado e medido — 55 s
> O EEG mede atividade elétrica do córtex. Os neurônios piramidais estão alinhados perpendicularmente à superfície; quando milhões recebem entrada sináptica ao mesmo tempo, os potenciais pós-sinápticos somam e formam um campo.
>
> Esse campo atravessa meninges, crânio e couro cabeludo e chega muito atenuado — dezenas de microvolts. Por isso é amplificado, filtrado e digitalizado.
>
> O que interessa aqui: resolução temporal de milissegundos, e um único canal frontal a 128 Hz. É o mesmo sinal que o monitor BIS usa.

Pule a lista numerada se estiver atrasado: diga só "soma de potenciais, atenuada pelo crânio".

### 7. Ritmos — 35 s
> O EEG é descrito por bandas. Delta são ondas lentas, de sono profundo e anestesia profunda; teta aparece em sonolência; alfa na vigília relaxada e, sob propofol, como um alfa frontal característico; beta na vigília ativa; gama é rápido e, na testa, muito contaminado por músculo.
>
> Essas mesmas bandas são a base do meu controle espectral: 14 atributos clássicos por janela.

Não leia a tabela linha a linha. Aponte delta e alfa, que são os que voltam depois.

### 8. EEG sob anestesia — 50 s
> Com o aprofundamento, o EEG muda de forma característica. Acordado, predominam ritmos rápidos e de baixa amplitude. Na sedação leve há um aumento paradoxal de beta e o sinal começa a lentificar.
>
> Em anestesia geral surge o alfa frontal em torno de 10 Hz somado a ondas lentas de grande amplitude. No plano profundo aparece surto-supressão: rajadas entre trechos quase planos.
>
> Essas quatro faixas são as classes que uso para relatar acurácia. **Não são limiares clínicos validados** — são vocabulário.

### 9. BIS — 50 s
> O BIS resume o EEG frontal numa escala de 0 a 100: perto de 100 acordado, 40 a 60 na anestesia geral, perto de zero o EEG isoelétrico.
>
> Mas ele tem limites que importam para o meu desenho. É proprietário. Sofre interferência de EMG, bloqueio neuromuscular e cetamina. E chega suavizado e com atraso de cálculo.
>
> Por isso o vocabulário do trabalho é cuidadoso: eu estimo **um rótulo de referência BIS** — não o BIS como verdade, e muito menos consciência.

Frase-âncora: *"um rótulo, não a verdade"*.

---

## Bloco 3 — Dados e método (5:55–9:25)

### 10. Caso real — 35 s
> Este é o caso 19 do holdout. Em cima, duas janelas de 5 segundos: uma limpa, que o pipeline quase não altera, e uma com pico de 125 microvolts, contido para 88.
>
> Embaixo, o espectrograma dos 72 minutos com o BIS na mesma base de tempo. Quando o BIS sobe, cai a potência em teta e sobe a de beta e gama. É essa relação que a rede aprende.
>
> As faixas brancas são quedas de eletrodo, rejeitadas pelos gates.

Slide de intuição. Se estiver atrasado, mostre e diga só a última frase.

### 11. Dados — 45 s
> Duas fontes públicas. O Figshare tem 24 casos anônimos com EEG frontal e BIS a cada 5 segundos; como não há identificador de pessoa, o grupo é o caso. O VitalDB é muito maior, de Seul, com monitor BIS Vista e BIS a cada segundo; ali agrupo pelo identificador do participante.
>
> São 59 arquivos; depois dos gates de qualidade, 33 casos elegíveis e 55.471 janelas de 5 segundos.
>
> A regra central: **nenhum caso cruza treino, validação e teste.**

### 12. Pré-processamento — 40 s
> O pipeline tem cinco passos: EEG frontal em microvolts, gates de qualidade, filtro causal de 0,5 a 45 Hz com notch de 50 Hz, corte de amplitude em ±100 microvolts e janelas de 5 segundos, 640 amostras a 128 Hz.
>
> O ponto importante é a causalidade: o filtro usa só passado e presente, com estado por caso. A mesma cadeia roda em replay ao vivo, com buffer de 5 segundos e atualização a cada segundo.

Não leia os limiares dos gates. Diga "três gates heurísticos de qualidade".

### 13. A rede — 50 s
> A rede é propositalmente pequena. Recebe 640 amostras em um canal e passa por quatro blocos convolucionais: canais de 1 para 32, 64, 128 e 128; kernels de 7 e 5; batch norm e GELU; max-pooling nos três primeiros, reduzindo o tempo de 640 para 80.
>
> Depois, média global, cabeça densa de 128 para 64 para 1 com dropout, e uma sigmoide escalada para 0 a 100.
>
> São **0,15 milhão de parâmetros**, o que permite rodar em CPU.

### 14. Treino e controles — 40 s
> O treino é curto: Smooth L1, dez épocas, batch 128. Dois checkpoints: o ativo, treinado só no Figshare, que testa o limite interno e a queda ao mudar de fonte; e o misto, que viu também parte do VitalDB.
>
> Como controle, uma Random Forest sobre os 14 atributos espectrais, nas mesmas janelas.
>
> E, como o próprio BIS chega suavizado, aplico uma média móvel exponencial causal sobre as predições congeladas — uma recursão de uma linha que só usa o passado.

---

## Bloco 4 — Resultados (9:25–13:30)

### 15. Resultado principal — 40 s
> No holdout Figshare, cinco casos e 5.523 janelas, a CNN teve **MAE de 7,03 pontos BIS contra 8,86 do controle espectral**.
>
> No bootstrap pareado por caso, a diferença é de menos 1,84 e o intervalo exclui zero.
>
> A ordenação, medida pelo PK, foi 0,773: sorteados dois momentos, a rede os ordena como o BIS em 77% das vezes. Com cinco casos, isso é exploratório.

### 16. Trajetória — 35 s
> Aqui a predição bruta ao longo de toda a cirurgia do caso 19, escolhido antes da inferência. A rede acompanha a queda na indução e a subida no despertar, mas **superestima o patamar do meio**: viés de mais 4,79.
>
> A correlação dentro do caso é alta, 0,91 — o que mostra que associação alta não elimina erro sistemático.

### 17. Comparação — 45 s
> Esta tabela junta os dois checkpoints nos dois benchmarks, com intervalos de 95% por bootstrap de casos inteiros.
>
> Três dos quatro MAE da rede ficam abaixo de 10 pontos — a diferença considerada aceitável entre duas medições do próprio BIS. É a minha régua de leitura, não um critério clínico.
>
> No Figshare, o misto não é significativamente melhor que o ativo: delta de 0,34, com intervalo cruzando zero.

### 18. VitalDB — 45 s
> A fronteira do trabalho está no transporte. O checkpoint que fica em 7 no Figshare vai a **12,43 no VitalDB**, e o PK cai para 0,515 — a rede deixa de ordenar os momentos da cirurgia.
>
> O controle espectral colapsa do mesmo jeito, então a perda não é específica da CNN.
>
> O misto melhora 14 dos 15 casos, piorando só o caso 16 — mas ele já tinha visto essa fonte. Isso é adaptação ao domínio, não generalização.

### 19. Implantabilidade — 45 s
> A segunda pergunta era sobre o estado da arte. O melhor resultado publicado neste corpus é um TCN-BiLSTM com atenção, de 2026: 1,07 milhão de parâmetros, MAE 6,03 após suavização.
>
> Com a nossa suavização causal de uma linha, o misto chega a 6,11 com **10 segundos de memória**, contra 45 a 95 segundos do deles.
>
> E o nosso é causal: o deles usa BiLSTM bidirecional e filtro de fase zero, que não existem ao vivo. Não é um ranking, porque os protocolos diferem — é uma comparação de ordem de grandeza.

### 20. Fatorial — 35 s
> Para fechar o diagnóstico, treinei quatro braços do zero só no Figshare, testando as escolhas do modelo publicado: reprodução, winsorização, Huber com treino longo e z-score por sujeito.
>
> O treino longo foi o único ganho interno: 6,63. Mas no VitalDB todos ficaram entre 12 e 12,6, contra 8,60 do misto, que viu a fonte.
>
> **O que separa os resultados é a exposição ao domínio** — não arquitetura, perda ou normalização.

---

## Bloco 5 — Fechamento (13:30–15:30)

### 21. Conclusão — 60 s
> Respondendo às três perguntas.
>
> Um: sim, a rede compacta supera o controle espectral sob validação por caso e ordena os momentos da cirurgia como a referência.
>
> Dois: com a suavização causal ela entra na faixa do melhor modelo publicado, sendo sete vezes menor e rodando ao vivo em CPU.
>
> Três: não, o desempenho não transporta para outra fonte — e o experimento fatorial mostra que a solução é expor o treino a mais fontes.
>
> A contribuição não é um sistema clínico. É uma cadeia compacta, causal e auditável — hashes, partições e replay — para testar hipóteses.

### 22. Limitações — 45 s
> As limitações estão declaradas. O estudo é retrospectivo e exploratório, com cinco casos de teste. O benchmark VitalDB foi reutilizado depois da exposição ao domínio. A imputação offline pode usar amostras futuras. O alvo é o próprio BIS, sensível a EMG, cetamina e atraso. E o EWMA foi escolhido após ver os resultados.
>
> O próximo passo é uma coorte travada de outra fonte, com partição e métricas definidas antes de qualquer inspeção — e treino com mais fontes.
>
> E deixo claro: **o BrainSniffer não mede consciência, não prescreve dose e não deve acionar conduta clínica.**

### 23. Obrigado — 15 s
> Obrigado. Fico à disposição para perguntas. O código está no GitHub, sob licença MIT, sem redistribuir os dados.

---

## Se estiver atrasado — ordem de corte

Corte nesta ordem, sem culpa:

1. **Slide 7 (Ritmos)** — diz só "as bandas clássicas são a base do controle espectral" e passa. **−35 s**
2. **Slide 10 (Caso real)** — mostra a figura e diz "é um caso do holdout". **−25 s**
3. **Slide 14 (Treino)** — "treino curto, dois checkpoints e um controle espectral". **−25 s**
4. **Slide 16 (Trajetória)** — só a frase do viés. **−20 s**
5. **Slide 20 (Fatorial)** — só a conclusão: "o que resolve é dado da fonte". **−20 s**

Total possível: **−2:05**, levando a fala para 13:25 — sobra folga para perguntas.

## Se estiver adiantado — onde ganhar tempo

- Slide 18: detalhe os 14/15 casos e o caso 16.
- Slide 19: explique a diferença entre filtro causal e filtro de fase zero.
- Slide 21: repita a contribuição em uma frase própria.

---

## Perguntas prováveis e respostas curtas

**"Por que não usar o BIS como verdade absoluta?"**
> Porque ele é proprietário, sensível a EMG e cetamina, e chega com atraso. Eu o trato como rótulo de referência, não como medida de consciência.

**"Por que o VitalDB piora tanto?"**
> Mudança de domínio: outro monitor, outra população, outra taxa de BIS. O controle espectral colapsa igual, então não é falha da CNN.

**"O que é PK?"**
> Probabilidade de concórdia: sorteio dois momentos da cirurgia; o PK é a chance de o modelo ordená-los na mesma direção que o BIS. 0,5 é acaso; 1 é ordem perfeita.

**"Por que a rede é tão pequena?"**
> Porque o objetivo é implantabilidade: 0,15 milhão de parâmetros roda em CPU e permite uma cadeia causal ao vivo.

**"Como rodaria ao vivo?"**
> Buffer de 5 segundos, atualização a cada segundo, filtro causal com estado por caso e EWMA — a mesma cadeia usada offline.

**"Isso pode ser usado clinicamente?"**
> Não. É uma base auditável para testar hipóteses; não mede consciência, não prescreve dose e não deve acionar alarme.

---

## Checklist de ensaio

- [ ] Cronometrar uma passada completa e anotar o tempo real por bloco.
- [ ] Ensaiar as transições 4 → 5, 9 → 10 e 20 → 21 (são as que mais travam).
- [ ] Decorar apenas três números: **7,03 vs 8,86**, **PK 0,773**, **12,43 / 0,515**.
- [ ] Testar o clique do controle remoto nos slides 17 e 18 (tabela e figura).
- [ ] Deixar as referências como anexo: só abrir se alguém pedir a fonte.

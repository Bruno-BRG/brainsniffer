# Roteiro falado — duração estimada de 5 a 7 minutos

> Texto para ensaio, não duração medida. Estimativa a 120–150 palavras por minuto, sem demonstração longa. A fonte científica é `tcc_brainsniffer.tex`; o estado do build está em `project_status.md`. As notas não fazem parte da fala.

## Abertura e pergunta

O BrainSniffer é um protótipo retrospectivo que lê gravações existentes de EEG frontal e estima um rótulo de referência BIS. O TCC não terá conexão com EEG físico. Ele não mede consciência, não controla anestésico, não aciona conduta e não substitui o anestesiologista. O BIS também não é uma verdade absoluta: artefatos, fármacos e atrasos podem alterar sua leitura.

A pergunta é: quanto uma rede compacta consegue aproximar essa referência em casos que não participaram do ajuste, comparada a um baseline espectral, e como o desempenho muda entre fontes? A contribuição é uma trilha auditável de dados, partições, configurações, relatórios e replay, não uma alegação de segurança clínica.

## Dados e separação

O Figshare oferece 24 casos anônimos. Vinte e três ficaram elegíveis; o caso 24 tem escala incompatível e permanece em quarentena. No checkpoint ativo, a divisão é de treze casos para treino, cinco para validação e cinco para teste. Não temos identificador de pessoa no Figshare: separar casos impede compartilhar a mesma cirurgia, mas não exclui cirurgias distintas da mesma pessoa.

No VitalDB, usamos EEG frontal e BIS, agrupando reoperações pelo identificador de participante. O pool elegível misto contém 23 casos Figshare e dez VitalDB. Excluindo os cinco casos do teste Figshare histórico, restam 28 grupos. Eles são divididos em dezesseis, seis e seis: o ajuste propriamente dito usa doze Figshare e quatro VitalDB. Portanto, 28 não é o número de grupos usados para ajustar os pesos.

## Método e tempo

Cada janela contém cinco segundos, ou 640 amostras a 128 hertz. Aplicamos controle de amplitude, banda de 0,5 a 45 hertz, notch de 50 hertz e filtragem causal com estado. A CNN histórica tem quatro convoluções, normalização, GELU, redução temporal e saída entre zero e cem. Os JSON registram dez épocas, batch 128 e semente 42. Melhorias recentes do treinador não podem ser atribuídas retroativamente a esses pesos.

Há uma distinção temporal importante: o alvo é o BIS do início da janela mais um offset, zero no padrão. A estimativa só fica disponível ao terminar os cinco segundos, além do processamento. Não é uma previsão de BIS futuro. No replay, depois desse buffer inicial, a atualização ocorre a cada segundo. Replay é percorrer um arquivo, não adquirir um sinal novo.

A preparação offline também tem limites. Na preparação offline, EEG não finito pode ser interpolado usando amostras futuras, sem limite de duração da lacuna nessa rotina. O gate de dois segundos do corpus de desenvolvimento não equivale à avaliação histórica dos quinze VitalDB. Na normalização VitalDB, o BIS é interpolado entre pontos finitos numa grade de um segundo, sem limite de lacuna nem máscara de imputação salva; fora do intervalo fica ausente. Depois, a construção das janelas descarta alvos não finitos ou fora de zero a cem. No replay, a política é rejeitar não finitos antes do filtro. Portanto, filtro causal não elimina o uso de amostras futuras na imputação offline.

## Resultados e incerteza

Nos cinco casos Figshare de teste, a CNN ativa teve MAE 7,03 e correlação 0,784. O baseline espectral com Random Forest teve MAE 8,86. A CNN também teve acurácia por faixa maior: 0,585 contra 0,540. Isso descreve uma partição e uma execução, não superioridade geral. As quatro faixas são heurísticas, não diagnósticos clínicos.

A nova figura mostra toda a gravação do caso dezenove, fixado antes da inferência por consistência da demonstração, não pelo desempenho. É a CNN bruta, sem suavização, comparada no tempo do alvo offline: início da janela, não instante de emissão. A queda inicial e a subida final aparecem nas duas curvas, mas a CNN superestima boa parte do patamar intermediário. Neste caso ilustrativo, o MAE foi 8,72 e o viés positivo 4,79 pontos. As lacunas de previsão foram mantidas, e a referência BIS continua um pouco além do EEG disponível. Um caso não sustenta generalização.

Reamostramos casos inteiros mil vezes. Para o MAE observado de 7,03, o intervalo exploratório ficou entre 6,38 e 8,24; para a correlação observada de 0,784, entre 0,703 e 0,881. Médias das réplicas não substituem esses pontos. Mil reamostragens de cinco casos continuam representando apenas cinco casos. Além disso, as métricas agrupam janelas, dando mais peso a cirurgias longas. Correlação mede associação, não acordo.

Nos quinze casos VitalDB históricos, o ativo perdeu desempenho: MAE 12,43 e correlação 0,024. O candidato misto obteve 8,60 e 0,688; no Figshare histórico, obteve 6,69 e 0,818. No VitalDB, o MAE melhorou em quatorze dos quinze casos, mas piorou no caso dezesseis. Isso é compatível com adaptação ao domínio, não prova de transporte para um novo centro.

O candidato já foi desenvolvido com outros participantes VitalDB, e os benchmarks tinham sido inspecionados. São comparações retrospectivas exploratórias. Não promover o candidato ao dashboard não recupera a independência do teste. Uma alegação futura de generalização exigiria dados realmente independentes; isso está fora da entrega deste TCC.

## Limitações e fechamento

O TCC entrega o protótipo retrospectivo documentado, a comparação CNN versus baseline, a visualização temporal e os limites encontrados. Investigar o efeito das lacunas e o acordo em dados secundários são extensões metodológicas possíveis, não requisitos adicionais desta graduação.

Usamos dados secundários públicos, mas isso não implica dispensa ética automática. Parecer ou dispensa, termos de uso e metadados institucionais precisam de confirmação pelo autor e pela instituição. A revisão do artigo e sua compilação não validam alterações simultâneas de código. O resultado defensável é uma base auditável para testar hipóteses, mantendo explícita a distância entre reproduzir BIS e cuidar de pacientes.

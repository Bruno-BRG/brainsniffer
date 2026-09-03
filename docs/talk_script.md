# Roteiro falado — apresentação de 5 a 7 minutos

## Abertura

“O BrainSniffer é um protótipo de pesquisa que recebe EEG frontal e estima, em fluxo, um índice relacionado ao BIS. A palavra importante aqui é protótipo: ele não controla anestésico, não substitui o anestesiologista e não foi validado para uso em pacientes.”

## Problema

“Durante uma cirurgia, o cérebro muda com o anestésico, o estímulo cirúrgico, os opioides, a idade e os artefatos. O EEG contém essa informação, mas o sinal é ruidoso e o BIS, embora útil como monitor, é um índice processado com limitações. Então a pergunta científica não é ‘a rede sabe a consciência?’. A pergunta inicial é mais honesta: ‘com um dataset público, conseguimos reproduzir de forma auditável uma referência BIS em pacientes não vistos no treinamento?’”

## Dados

“A primeira versão usa o dataset público EEG and BIS raw data, do Figshare, com 24 casos. O EEG é frontal e amostrado a 128 hertz; o BIS é registrado a cada cinco segundos. O sistema baixa os arquivos MATLAB diretamente pela interface ou pela linha de comando. Cada caso continua anônimo.”

## Método

“Nós cortamos o EEG em janelas de cinco segundos, portanto 640 amostras. O pipeline controla a amplitude, filtra de 0,5 a 45 hertz, remove a rede de 50 hertz e registra uma qualidade heurística baseada em valores ausentes, saturação e linha plana. Uma CNN de uma dimensão recebe a onda e retorna um valor entre zero e cem. Depois mapeamos esse valor para quatro faixas: profundo, anestesia geral, sedação leve e acordado.”

## Decisão metodológica principal

“A proteção mais importante contra uma métrica enganosa é a divisão por caso. Todas as janelas de um paciente ficam em apenas um conjunto. Se misturarmos janelas vizinhas entre treino e teste, a rede pode memorizar a gravação em vez de generalizar para um paciente novo.”

## Tempo quase real

“Na interface, o replay alimenta um buffer circular de cinco segundos. A cada segundo sai uma estimativa, com valor bruto, valor suavizado e qualidade. Isso demonstra o caminho de engenharia e permite medir latência. A filtragem causal mantém estado entre chunks; quando a fonte tem outra taxa, o resampler também mantém sobreposição e atraso conhecido. Ainda não é um adaptador para um equipamento clínico: taxa, latência e fidelidade precisam ser conferidas no hardware alvo.”

“Para aproximar o uso de pesquisa, o mesmo núcleo aceita Lab Streaming Layer, uma camada aberta que descobre streams, transporta timestamps e sincroniza sinais. Assim não prendemos a aplicação a uma marca de EEG. Também existe uma entrada JSON simples para um bridge do fabricante. LSL resolve transporte e sincronização; não resolve qualidade, validação ou autorização clínica.”

## Avaliação

“As métricas são erro absoluto médio, raiz do erro quadrático médio, viés e correlação para o valor contínuo. Para as faixas, usamos acurácia e macro-F1. Também devemos olhar por paciente, por droga, por qualidade e por transição. O resultado não pode ser apenas uma média global.”

“Na execução de engenharia, 24 casos foram baixados e 23 passaram pelo gate de qualidade, totalizando 31.055 janelas. A CNN obteve no teste separado por caso MAE de 7,03, RMSE de 11,08, correlação de 0,784 e macro-F1 de 0,548. A baseline espectral teve MAE de 8,86 e macro-F1 de 0,480 na mesma partição. Isso é uma demonstração reproduzível do pipeline, não uma prova de superioridade ou de segurança clínica.”

“Para não fingir que milhares de janelas são milhares de pacientes, reamostramos as cinco cirurgias de teste mil vezes. O Pearson médio foi 0,789, com intervalo exploratório de 95% entre 0,703 e 0,881; o MAE médio foi 7,11, entre 6,38 e 8,24. São apenas cinco casos, portanto isso mede incerteza do experimento e não eficácia clínica.”

“Também ampliamos um teste externo exploratório, sem retreinar, para 15 casos compatíveis do VitalDB: 1 a 10, 12 a 14, 16 e 17. Foram 38.730 janelas, com MAE de 12,43, correlação de 0,024 e macro-F1 de 0,398. O caso 11 não tinha os dois tracks necessários. A expansão ocorreu depois do primeiro piloto, então não é um resultado pré-registrado. O resultado é pior e varia muito entre pacientes; isso mostra que um checkpoint não pode ser levado para outro aparelho ou centro sem verificar unidade, montagem, atraso e população.”

“Também encontramos pontos não finitos nos 15 arquivos. O relatório os contabiliza e a preparação offline pode interpolá-los para inspeção do dataset, mas o caminho ao vivo rejeita NaN e Inf antes do filtro. Isso é uma decisão de segurança de aquisição, não uma alegação de robustez clínica.”

“Há uma segunda falha diferente: silêncio da fonte. Se nenhum chunk chega dentro do timeout, o modo de bancada encerra com erro e guarda o relatório parcial. No modo exploratório, a última estimativa é invalidada como ABSTAIN, o estado causal é limpo e o sistema só volta a estimar depois de preencher uma janela nova. Assim não deixamos um valor antigo parecer atual.”

“Fizemos ainda uma análise de sensibilidade do alinhamento. Mantendo o checkpoint
e os mesmos pacientes de teste, associamos a janela EEG a valores BIS de menos
20 até mais 20 segundos. O MAE caiu de 7,22 para 6,76 e a correlação subiu de
0,766 para 0,799 na grade. Isso é uma pista de atraso do monitor, não uma licença
para escolher o melhor número depois de olhar os dados. O offset precisa ser
pré-especificado e confirmado com relógios e dados independentes.”

“Ao reamostrar pacientes inteiros mil vezes, o intervalo exploratório de 95% para a correlação foi de menos 0,126 a 0,193. Ou seja, mesmo com 15 casos não temos evidência de correlação confiável; temos um sinal claro de que a validação precisa continuar.”

“Antes de aumentar a rede, comparamos com uma baseline espectral que usa potência por banda, entropia e frequência de borda. Se a CNN não superar essa referência sob pacientes novos, a complexidade não está justificada.”

## Limitações e segurança

“Este conjunto tem poucos pacientes, um canal e um rótulo derivado de monitor. Artefatos, atraso do BIS e agentes como cetamina podem mudar a interpretação. A rede não pode decidir dose. Antes de pensar em uso clínico seriam necessários protocolo ético, validação externa multicêntrica, estudo de segurança, calibração, abstention em sinal ruim e avaliação regulatória.”

## Fechamento

“O valor do BrainSniffer neste estágio é criar uma trilha reproduzível: dataset identificável, decisões registradas, baseline simples, divisão correta e replay auditável. A próxima pergunta será se uma CNN crua realmente supera baselines espectrais e se mantém desempenho em outro centro. Só depois disso faz sentido aumentar a arquitetura.”

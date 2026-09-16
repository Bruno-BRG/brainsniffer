# Checklist de aceite da demonstração

**Estado: cenários integrados abaixo NÃO EXECUTADOS nesta revisão.** Este é um
plano, não um relatório de aprovação. Testes automatizados, cliente Flask/Node
ou HTTP 200 isolado não comprovam renderização e interação no browser.

Uso somente em pesquisa. Replay não é aquisição ao vivo e BIS não é medida
absoluta de consciência. Este checklist não autoriza testes com pacientes,
acesso à rede, build, deploy ou mudanças em produção.

## Obrigatórios para a demonstração local do TCC

Cada item permanece **NÃO EXECUTADO** até haver evidência específica.

- [ ] **Preparar e registrar:** revisão Git/diff, Python/dependências, SO/browser,
  comando/configuração, casos e hashes SHA-256 de modelo e relatórios; sem
  credenciais ou EEG identificável. Confirmar checkout completo e artefatos
  congelados/somente leitura, responsável e versão de retorno.
- [ ] **Escopo:** Dash como única UI, arquivos existentes e replay retrospectivo.
  LSL foi removido. Download, treino e validação de ficha permanecem na CLI, fora
  das ações do visitante; JSONL é opcional/local, não captura de hardware.
  Não substituir modelo nem baixar dados durante a apresentação.
- [ ] **Roteiro no browser:** problema → caso → EEG/BIS → replay (play/pause,
  cursor e velocidade) → resultados → limitações. Conferir gráficos, benchmark,
  legendas, erros e resultados negativos, sem confundir projeção com medição.
- [ ] **Offline/rede:** com dependências e casos preparados, verificar assets,
  abas e replay sem acesso externo; registrar console e requisições. Se houver
  acesso à rede autorizado, comparar falha/disponibilidade dos recursos externos,
  sem download silencioso, dados inventados ou sessão travada sem diagnóstico.
- [ ] **Abstenção:** em fixture descartável, testar válido → inválido → válido;
  verificar cartões, curvas, cursor e mensagens. Valor antigo não pode parecer
  estimativa atual; recuperação respeita janela e qualidade, não confiança clínica.
- [ ] **Integridade e prontidão:** usar cópias descartáveis, nunca artefatos da
  demo. Testar caminho absoluto, `../`, symlink externo e arquivo fora da lista
  permitida com sentinelas inofensivas; rejeição antes de leitura indevida.
  Testar modelo/caso ausente, corrompido ou incompatível e troca do checkpoint
  no mesmo caminho (sem pickle malicioso). Registrar `/healthz`, corpo/código,
  hash e diagnóstico da UI; não aceitar cache antigo como novo resultado.
- [ ] **Duas sessões:** browsers/perfis independentes com casos, cursores e
  velocidades distintos, sem vazamento de estado; artefatos compartilhados
  permanecem imutáveis. Comparar hashes antes/depois e registrar permissões.

## Etapas separadas — somente se fizerem parte de pesquisa futura

Não são condições para apresentar o replay gravado da graduação. Permanecem
**NÃO EXECUTADAS**; exigem ambiente, escopo e autorização próprios.

| Etapa | Evidência necessária | Estado |
| --- | --- | --- |
| Gates JSONL locais na CLI (não captura) | NaN/Inf, saturação, linha plana, baixa qualidade, lacunas, timestamps inválidos/regressivos e silêncio; erro/ABSTAIN, invalidação da saída, recuperação e relatório parcial. | NÃO EXECUTADO |
| Metadados/intake na CLI | Omissões e conflitos de unidade, taxa, canal, referência e montagem; modo estrito rejeita antes da inferência, exploração identificada. | NÃO EXECUTADO |
| LSL sintético (histórico) | Adaptador, extra `live`/liblsl e publisher removidos. Relatórios passados preservados para auditoria, sem reexecução ou atribuição ao JSONL. | REMOVIDO / NÃO APLICÁVEL |
| EEG físico | Ficha real completa, autorização e protocolo específico; driver/bridge, unidade, ganho, montagem, relógios e falhas reais. `ready_for_bench=true` não substitui ensaio nem autoriza uso clínico. | NÃO EXECUTADO |
| Instalação/build ou publicação | Validar separadamente ambiente locked, build, rede e infraestrutura escolhida; nenhum deploy é exigido para demo local. | NÃO EXECUTADO |

## Encerramento (NÃO EXECUTADO)

- [ ] Registrar responsável, data, comandos/ações, exit code, resultado e local
  das evidências por item. Distinguir teste unitário, Flask/Node, browser e hardware.
  Manter `NÃO EXECUTADO` onde faltar ambiente/permissão; ausência de falha não é aceite.
- [ ] Obter decisão humana de prontidão da demo; registrar falhas e restrições.
  Falhas de isolamento, abstenção, integridade ou gates impedem declarar aprovação.
  Não alterar produção nem alegar aprovação do TCC por banca a partir deste checklist.

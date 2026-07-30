## Apresentação & Contexto do Projeto

Este repositório abriga o código-fonte, os experimentos e a documentação científica da pesquisa intitulada **"Análise de Eficiência e Resiliência em Grafos Dinâmicos: Um Estudo Bioinspirado em Redes Miceliais para Otimização de Redes Neurais Artificiais"**.

O projeto foi selecionado e financiado pelo **Edital de Iniciação Científica e Tecnológica (2026/2027)** da **Fundação de Apoio à Pesquisa do Distrito Federal (FAPDF)**. A execução técnica e a condução das simulações são realizadas por mim, pesquisadora bolsista FAPDF, visando contribuir para o ecossistema de computação sustentável, autonomia tecnológica nacional e inovação algorítmica no Brasil.

---

## Fundamentação Teórica

### O Paradigma da IA Verde (Green AI)
O avanço contemporâneo do Aprendizado de Máquina é marcado pela predominância do paradigma ***Red AI***, no qual ganhos marginais de precisão são alcançados por meio do aumento maciço do número de parâmetros e da infraestrutura computacional. Essa abordagem implica custos ambientais e hídricos críticos: relatórios da *International Energy Agency* (IEA, 2024) apontam que as cargas de trabalho de IA utilizam dezenas de gigawatts globalmente em data centers, exigindo bilhões de litros de água para resfriamento e emitindo milhões de toneladas de $CO_2$.

Em contraposição, a **Inteligência Artificial Verde (*Green AI*)** (Schwartz et al., 2020) defende a eficiência algorítmica e a economia estrutural como métricas primárias de desempenho (*Green-in-AI*). O objetivo é romper com arquiteturas estáticas, densas e redundantes, promovendo a esparsidade funcional desde a concepção do modelo.

### Morfologia e Biofísica das Redes Miceliais
Diferentemente dos sistemas biológicos nos quais a rede de transporte é dissociada do organismo, em redes fúngicas miceliais a infraestrutura lógica **é o próprio organismo** ("*The network is the organism*", Heaton et al., 2012). O micélio do fungo *Phanerochaete velutina* desenvolve-se explorando o substrato de forma descentralizada e adaptativa. 

Três processos fundamentais orquestram essa dinâmica:
1. **Crescimento Apical e o *Spitzenkörper*:** A extensão dos filamentos (hifas) ocorre na extremidade distal, coordenada pelo *Spitzenkörper*, um complexo organizador de vesículas que orienta a expansão espacial em resposta a gradientes ambientais.
2. **Anastomose (Fusão Lateral):** À medida que a exploração avança, hifas conectam-se lateralmente. A arquitetura evolui de uma árvore acíclica para um grafo cíclico e complexo, estabelecendo rotas alternativas de transporte e elevando a resiliência a danos locais.
3. **Fluxo de Massa Biofísico:** O transporte nutricional ocorre via fluxo de massa impulsionado por pressão de turgor ($q = \Delta V / t$). A condutância de cada caminho é reforçada pelo volume de sinal que trafega por ele, transmitindo informações globais de relevância sem a presença de um sistema nervoso central.

### O Modelamento Mycelium-NAS e a Poda Darwiniana
Inspirado pelo "Modelo Darwiniano de Redes" e consolidado por Islam (2025) na abordagem **Mycelium-NAS**, o crescimento micelial segue a lógica de superprodução inicial (*overproduction*) seguida de **Autofagia** (poda seletiva e reciclagem de biomassa). Caminhos com baixo fluxo informacional regridem, e a biomassa é reabsorvida para sustentar frentes de crescimento mais promissoras. Em Redes Neurais Artificiais, esse mecanismo traduz-se em algoritmos de *pruning* dinâmico e proativo em tempo de execução.

---

## A Questão de Pesquisa & Hipóteses

* **Pergunta Central:** *De que forma a modelagem de Redes Neurais Artificiais como grafos dinâmicos bioinspirados em redes miceliais influencia a eficiência estrutural, o consumo computacional e a resiliência a falhas em comparação com arquiteturas neurais estáticas tradicionais?*

* **Hipótese 1 (Eficiência):** A aplicação de operadores de expansão e autofagia reduzirá em pelo menos **25%** o número de conexões ativas necessárias para manter o fluxo informacional em comparação a redes densas tradicionais.
* **Hipótese 2 (Resiliência):** A emergência de ciclos decorrente do processo de anastomose garantirá a manutenção da conectividade global mesmo após testes de estresse com remoção aleatória de até **50%** dos nós ativos.

---

## Objetivos

### Objetivo Geral
Desenvolver e validar um modelo computacional baseado em grafos dinâmicos bioinspirados em redes miceliais, implementado em ambiente Python, comparando seu desempenho, eficiência estrutural e resiliência a falhas com arquiteturas neurais estáticas tradicionais.

### Objetivos Específicos
1. Formalizar matemática e algorítmicamente os processos de crescimento apical (*Spitzenkörper*), anastomose e autofagia do fungo *Phanerochaete velutina*.
2. Construir um **Grupo de Controle (MVP 1 - Estático)** com matrizes neurais densas e fixas em `PyTorch`.
3. Desenvolver um **Grupo Experimental (MVP 2 - Dinâmico)** através de simulador de grafos adaptativos utilizando `NetworkX`.
4. Auditar o consumo energético e a pegada de carbono das simulações empregando ferramentas de monitoramento em tempo real (`CodeCarbon`/`CarbonTracker`).
5. Avaliar a robustez dos modelos submetendo-os a baterias de testes de estresse por remoção aleatória de 10%, 30% e 50% de seus nós ativos.

---

## Procedimentos Metodológicos

### Mapeamento dos Operadores Biológicos

| Fenômeno Biológico | Abstração Matemática / Algorítmica | Operador Computacional |
| :--- | :--- | :--- |
| **Spitzenkörper** | Expansão direcional condicional baseada em gradientes de erro/sinal. | **Expansão Adaptativa:** Brotamento de novos nós/arestas. |
| **Anastomose** | Cruzamento de hifas e elevação do *Meshedness Alpha Index*. | **Fusão e Triangulação:** Adição de arestas cíclicas para redundância. |
| **Fluxo de Massa** | Condutância proporcional à variação de volume ($q = \Delta V / t$). | **Atualização Dinâmica de Pesos:** Modulação baseada na vazão de dados. |
| **Autofagia** | Reciclagem de biomassa com base em limiares mínimos de fluxo. | **Poda Seletiva (*Pruning*):** Remoção contínua de conexões ineficientes. |

### Arquitetura dos Protótipos (MVPs)

```
                       +-----------------------------------+
                       |      Massa de Dados de Entrada    |
                       +-----------------+-----------------+
                                         |
             +---------------------------+---------------------------+
             |                                                       |
             v                                                       v
 +-----------------------+                               +-----------------------+
 |  Grupo de Controle    |                               |  Grupo Experimental   |
 |  (MVP 1: Estático)    |                               |  (MVP 2: Dinâmico)    |
 +-----------------------+                               +-----------------------+
 | - Topologia Densa     |                               | - Expansão Apical     |
 | - Conexões Fixas      |                               | - Anastomose (Loops)  |
 | - Baseline PyTorch    |                               | - Autofagia (Pruning) |
 +-----------------------+                               +-----------------------+
             |                                                       |
             +---------------------------+---------------------------+
                                         |
                                         v
                       +-----------------------------------+
                       |  Avaliação Estatística & Auditoria|
                       |  - Teste de Mann-Whitney U        |
                       |  - Route Factor & Betweenness     |
                       |  - Pegada de Carbono (CodeCarbon) |
                       +-----------------------------------+
```

### Métricas de Avaliação e Testes de Estresse
- **Economia Estrutural:** Percentual de redução de arestas ativas sem perda da capacidade de fluxo.
- **Fator de Rota (*Route Factor* - $RF$):** Razão entre a menor distância no grafo e a distância euclidiana direta ($RF = d_{net} / d_{euc}$).
- **Coeficiente de Meshedness ($ lpha$):** Proporção de ciclos mantida na fase de maturação do grafo ($10\% - 20\%$).
- **Centralidade de Intermediação (*Betweenness Centrality*):** Identificação de *backbones* informacionais (análogos aos cordões miceliais).
- **Resiliência a Falhas:** Conectividade residual e variação do diâmetro da rede após remoção aleatória de 10%, 30% e 50% dos nós.
- **Tratamento Estatístico:** Aplicação do teste não-paramétrico de **Mann-Whitney U** para validação das distribuições de eficiência.

---

## Alinhamento Estratégico (PBIA 2024-2028 & ODS/ONU)

A realização deste projeto financiado pela **FAPDF** reforça a autonomia tecnológica e científica regional e nacional:
* **Plano Brasileiro de Inteligência Artificial (PBIA 2024–2028):**
  * *Eixo 1 (Infraestrutura e Desenvolvimento):* Foco na redução da demanda energética e criação de algoritmos sustentáveis.
  * *Eixo 4 (IA para Inovação Empresarial e Soberania):* Estudo de tecnologias abertas que reduzem a dependência de aceleradores de hardware de alto custo.
* **Objetivos de Desenvolvimento Sustentável (ODS/ONU):**
  * **ODS 9:** Indústria, Inovação e Infraestrutura (infraestruturas digitais resilientes).
  * **ODS 12:** Consumo e Produção Responsáveis (computação de baixo consumo de recursos).
  * **ODS 13:** Ação Contra a Mudança Global do Clima (mitigação da pegada de carbono da IA).

## Principais Referências Bibliográficas

1. **BRASIL. Ministério da Ciência, Tecnologia e Inovação.** *Plano Brasileiro de Inteligência Artificial: IA para o Bem de Todos (PBIA 2024-2028)*. Brasília: MCTI, 2024.
2. **FRICKER, M. D. et al.** The Mycelium as a Network. *Microbiology Spectrum*, v. 5, n. 3, 2017.
3. **HEATON, L. L. M. et al.** Growth-induced mass flows in fungal networks. *Proceedings of the Royal Society B: Biological Sciences*, v. 277, n. 1698, p. 3265-3274, 2010.
4. **HEATON, L. L. M. et al.** Analysis of fungal networks. *Fungal Biology Reviews*, v. 26, n. 1, p. 12-29, 2012.
5. **INTERNATIONAL ENERGY AGENCY (IEA).** *Electricity 2024: Analysis and forecast to 2026*. Paris: IEA, 2024.
6. **ISLAM, T.** Mycelium neural architecture search. *Evolutionary Intelligence*, v. 18, n. 4, p. 1-11, 2025.
7. **MUNDIM, G. P.** *Aplicação de heurísticas computacionais bioinspiradas no jogo do dinossauro*. Trabalho de Conclusão de Curso (Graduação) - Universidade Federal de Uberlândia, 2025.
8. **SCHWARTZ, R. et al.** Green AI. *Communications of the ACM*, v. 63, n. 12, p. 54-63, 2020.

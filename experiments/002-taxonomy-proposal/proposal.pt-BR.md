# Proposta de taxonomia: categorias e subcategorias

**Data:** 2026-09-28 · **Ambiente:** banco e Qdrant de desenvolvimento (somente leitura) ·
**Base:** experimento 002 ([README](README.md)) · **Definição completa:**
[`src/lab/taxonomy_v2.py`](../../src/lab/taxonomy_v2.py)

---

## 1. Resumo

- **16 categorias master** (hoje são 12) e **113 subcategorias**.
- **4 masters novas:** **Geopolítica** (~10% das notícias), **Entretenimento** (~4%),
  **Loterias** (1,3%) e **Estilo de Vida** (~1%).
- **Uma notícia pode ter mais de uma subcategoria**, por exemplo "Futebol" e "Transferências".
- **Nenhuma subcategoria usa geografia.** Não existe "futebol brasileiro" ou "política dos EUA".
  A localização do leitor vira um filtro separado.
- **Falta um passo:** validar a taxonomia em notícias que o processo ainda não viu (seção 6).

---

## 2. Como foi feito

1. **Categorias master:** olhamos as **85.586 notícias** do dev, agrupadas por assunto dentro
   das 12 categorias atuais, e procuramos grupos grandes que não pertenciam à categoria onde
   estavam. Foi assim que Geopolítica apareceu: 39% do que hoje é Política são guerras e
   diplomacia.
2. **Subcategorias:** um LLM deu etiquetas genéricas de assunto para **3.825 notícias** (sem
   nomes de país, clube, pessoa ou evento), outro LLM agrupou as etiquetas em subcategorias por
   categoria, e a versão final foi **revisada à mão**. A revisão resolveu o que o processo
   automático não decide sozinho, como de que lado fica uma fronteira.

---

## 3. Categorias master

| Categoria | O que muda |
|---|---|
| Política | Perde guerras e diplomacia para Geopolítica. Fica com a política **interna** de qualquer país |
| **Geopolítica** (nova) | Relações entre países: guerras, diplomacia, alianças militares, sanções, ONU, BRICS |
| Economia | Perde os resultados de loteria |
| **Loterias** (nova) | Sorteios e prêmios (Mega-Sena, Lotofácil…). Sem subcategorias |
| Esportes | Perde a vida pessoal de atletas para Entretenimento |
| Tecnologia | — |
| Segurança | Passa a incluir explicitamente **acidentes e tragédias** |
| Saúde | — |
| Cultura | Perde celebridades, TV, filmes e séries para Entretenimento |
| **Entretenimento** (nova) | Famosos e influenciadores, TV, novelas, reality shows, filmes e séries |
| **Estilo de Vida** (nova) | Viagens, gastronomia, moda, casa, pets, horóscopo, comportamento |
| Transporte | Passa a incluir carros e o mercado automotivo |
| Meio Ambiente | Passa a incluir **clima, previsão do tempo e desastres naturais** |
| Educação | — |
| Moradia | — |
| Direitos Humanos | — |

---

## 4. Subcategorias

### 4.1 Como funcionam

Cada categoria tem até **duas facetas**:

- **Principal:** divide a categoria por um único critério, e quase toda notícia cai em uma.
  Exemplos: em Esportes, a **modalidade**; em Política, o **assunto**.
- **Transversal** (opcional): aspectos que se repetem em várias subcategorias principais, e a
  notícia pode ter ou não. Exemplo em Esportes: Transferências, Gestão, Esporte feminino.

Uma notícia recebe todas as subcategorias que se aplicam, das duas facetas. Exemplos:

- Contratação de uma jogadora → **Futebol + Transferências + Esporte feminino**
- Chuva forte que derruba casas → **Desastres naturais**
- Reforma tributária votada no Congresso → **Projetos de lei e reformas** (Política), com
  Economia como categoria secundária

### 4.2 Lista

| Categoria | Faceta principal | Faceta transversal |
|---|---|---|
| **Política** | Eleições e campanhas · Corrupção e investigações · Democracia e instituições · Governo e gestão pública · Projetos de lei e reformas · Partidos e alianças · Polarização, protestos e extremismo | — |
| **Geopolítica** | Guerras e conflitos armados · Negociações de paz e cessar-fogo · Diplomacia e relações entre países · Defesa, alianças militares e armamentos · Sanções e disputas comerciais · Organismos e direito internacional · Migração e fronteiras | — |
| **Economia** | Inflação, juros e PIB · Contas públicas e impostos · Mercado financeiro e investimentos · Empresas e negócios · Trabalho, emprego e renda · Custo de vida e consumo · Finanças pessoais e dívidas · Energia e commodities · Agronegócio · Comércio exterior | — |
| **Loterias** | — | — |
| **Esportes** | Futebol · Tênis · Automobilismo · Basquete · Vôlei · Lutas · Atletismo e corrida · Ciclismo · Esportes aquáticos · Rúgbi · Críquete · Golfe · Jogos Olímpicos e multiesportivos | Transferências e contratos · Gestão, finanças e bastidores · Esporte feminino |
| **Tecnologia** | Inteligência artificial · Internet, redes sociais e apps · Celulares e eletrônicos · Games · Espaço e ciência · Telecomunicações e conectividade · Robótica, chips e indústria tech | Regulação e ética · Segurança digital e privacidade |
| **Segurança** | Crimes violentos · Violência contra a mulher · Violência sexual e abuso infantil · Crime organizado e tráfico · Roubos, furtos e golpes · Acidentes e tragédias · Polícia, prisões e política de segurança | — |
| **Saúde** | Doenças infecciosas e vacinas · Doenças crônicas e câncer · Tratamentos e medicamentos · Saúde mental · Nutrição, exercício e obesidade · Sistemas e serviços de saúde | Prevenção · Pesquisa médica · Política e regulação da saúde |
| **Cultura** | Música · Artes visuais · Livros e literatura · Teatro e dança · Patrimônio e história · Religião | Política e fomento cultural |
| **Entretenimento** | Famosos e influenciadores · TV, novelas e reality shows · Filmes e séries | — |
| **Estilo de Vida** | Viagens e turismo · Gastronomia e receitas · Moda e beleza · Casa e decoração · Animais de estimação · Astrologia e horóscopo · Relacionamentos e comportamento | — |
| **Transporte** | Aviação · Rodovias e trânsito · Mobilidade urbana e transporte público · Carros e mercado automotivo · Ferrovias, portos e navegação | Infraestrutura e concessões · Qualidade do serviço e acessibilidade |
| **Meio Ambiente** | Mudanças climáticas · Tempo e previsão · Desastres naturais · Biodiversidade e animais · Desmatamento e uso da terra · Poluição e saneamento · Transição energética e sustentabilidade | — |
| **Educação** | Educação infantil e fundamental · Ensino médio · Ensino superior · Cursos técnicos e profissionalizantes | Provas e avaliações · Professores · Política educacional e financiamento |
| **Moradia** | Mercado imobiliário e aluguel · Acesso à moradia · Urbanismo e planejamento urbano · Construção | — |
| **Direitos Humanos** | Liberdades civis · Discriminação e igualdade · Migrantes e refugiados · Crianças e grupos vulneráveis | Crises humanitárias · Ativismo |

A definição de cada subcategoria, com o que entra e o que não entra, está em
`src/lab/taxonomy_v2.py`.

### 4.3 Fronteiras entre categorias

Assuntos que poderiam cair em duas categorias têm um lado definido:

| Assunto | Fica em | E não em |
|---|---|---|
| Acidentes (trânsito, aviação, afogamento, incêndio, desabamento) | Segurança | Transporte |
| Desastres naturais (enchentes, deslizamentos, secas, queimadas) | Meio Ambiente | Segurança |
| Música como arte, shows e festivais | Cultura | Entretenimento |
| Vida pessoal de músicos e atletas | Entretenimento | Cultura / Esportes |
| Filmes e séries | Entretenimento | Cultura |
| Carros e mercado automotivo | Transporte | Economia |
| Golpes e crimes digitais | Segurança | Tecnologia |
| Beleza, autocuidado e relacionamentos | Estilo de Vida | Saúde |
| Direitos de migrantes e refugiados | Direitos Humanos | Geopolítica |
| Política migratória entre países | Geopolítica | Direitos Humanos |

---

## 5. Decisões já tomadas (com o responsável pelo produto)

1. **Geopolítica, Entretenimento, Loterias e Estilo de Vida** como categorias master.
2. **Subcategorias multi-label**, organizadas em facetas.
3. **Sem geografia** na taxonomia: a localização é um filtro à parte.
4. **Esportes por modalidade.** A regra de tamanho mínimo tinha juntado todos os esportes menos
   o futebol em "esportes individuais", mas quem acompanha tênis quer "Tênis".
5. **Política por assunto**, como as outras categorias. A divisão por instituição (Executivo,
   Legislativo, Judiciário) foi considerada e descartada.

---

## 6. O que ainda falta

- **Validação (feita em 29/09):** a taxonomia foi aplicada a **1.390 notícias novas**. De 92% a
  100% dos artigos de cada categoria caíram numa subcategoria dela, e quase nenhum recebeu
  subcategoria de outra categoria. Algumas subcategorias ficaram vazias (basquete, vôlei,
  desmatamento, professores); a planilha de revisão mostra o percentual de cada uma.
- **Ajustes da versão 2.1 (29/09):** Direitos Humanos passou a se dividir por tipo de questão, com
  uma só subcategoria "Discriminação e igualdade" no lugar das subcategorias por grupo; Religião
  entrou em Cultura; Aluguel, Habitação social e Situação de rua viraram "Acesso à moradia";
  "Clima e previsão do tempo" virou "Tempo e previsão"; "SUS e sistema de saúde" virou
  "Sistemas e serviços de saúde".
- **Revisão do time:** aprovar, ajustar ou recusar cada subcategoria na planilha
  `taxonomia_v2_validacao.xlsx`.
- **Ressalva:** o dev puxa muito de feeds do Reino Unido e de Portugal. Por isso aparecem
  críquete, rúgbi e o Partido Trabalhista. **Críquete** em especial pode não se justificar em
  produção.

---

## 7. O que precisamos do time

1. Aprovar as **4 categorias master novas**.
2. Aprovar **geografia como filtro separado**.
3. Revisar a lista de subcategorias da seção 4.2: nomes, o que falta e o que sobra.
4. Decidir se **Loterias** aparece nas preferências do usuário ou fica como categoria técnica.

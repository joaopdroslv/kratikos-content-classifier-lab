# Proposta de taxonomia: categorias e subcategorias

**Data:** 2026-09-28 · **Ambiente:** banco e Qdrant de desenvolvimento (somente leitura) ·
**Base:** experimento 002 ([README](README.md))

---

## 1. Resumo

- Olhamos **todas as 85.586 notícias** do dev, agrupadas pelos 45.970 tópicos, e dentro de cada
  uma das 12 categorias atuais procuramos os assuntos que mais se repetem.
- **Geopolítica merece categoria própria.** Guerras, diplomacia e relações entre países são
  **~10% de todas as notícias**, o que faria dela a 4ª maior categoria. Hoje isso está dentro de
  Política e espalhado por Segurança e Direitos Humanos.
- **Entretenimento também** (~4%): famosos, novelas, streaming. Hoje está dentro de Cultura e até
  de Esportes (vida amorosa de jogador).
- **Loterias precisam de uma decisão:** são 1,3% das notícias, mais que Educação e Moradia
  somadas, e não são assunto de debate.
- Sugerimos **não misturar geografia com assunto**: "política dos EUA" é Política, não uma
  categoria à parte. A abrangência (Brasil, internacional) já existe nos tópicos e pode virar um
  filtro separado.

---

## 2. Como foi feito

1. Cada tópico recebeu uma das 12 categorias atuais pelo modelo do experimento 001 (o "aluno",
   83% de acerto contra revisão humana).
2. Dentro de cada categoria, agrupamos os tópicos por semelhança de conteúdo (embeddings) e lemos
   os grupos: **115 grupos** no total, cada um com seu tamanho e exemplos.
3. Um grupo grande e coerente é candidato a **subcategoria**. Se ele não pertence de verdade à
   categoria onde caiu, é candidato a **categoria nova**.

Os percentuais são estimativas: o aluno erra ~17% das vezes e os grupos não são perfeitamente
puros. Servem para comparar tamanhos, não como número exato.

---

## 3. Categorias master

### 3.1 Propostas

| Proposta | Tamanho | Por quê |
|---|---|---|
| **Nova: Geopolítica** | ~10% | Guerras (EUA-Irã, Rússia-Ucrânia, Israel-Palestina, Iêmen), diplomacia (EUA-China, BRICS, ONU, OTAN), relações do Brasil com outros países. É 39% do que hoje é Política |
| **Nova: Entretenimento** | ~4% | Famosos e influenciadores, novelas, streaming e cinema comercial, mortes de artistas de TV. Cultura fica com artes, música, festivais e patrimônio |
| **Ajustada: Política** | ~14% (hoje 22,5%) | Sem a geopolítica, fica com governo, eleições, Judiciário e Congresso, **de qualquer país** |

### 3.2 Para decidir

| Questão | Dados | Opções |
|---|---|---|
| **Loterias** | 1,3% das notícias (resultados da Mega-Sena, Lotofácil…) | (a) categoria técnica, fora das preferências do usuário; (b) subcategoria de Economia; (c) categoria normal |
| **Estilo de vida** | ~1,1% (receitas, horóscopo, turismo) | Categoria nova, ou subcategoria de Entretenimento/Cultura |
| **Moradia** | 0,2%, a menor de todas | Manter, ou juntar com obras urbanas e mobilidade numa categoria "Cidades" |
| **Política de outros países** | 5,7% (Trump e a Suprema Corte dos EUA, Partido Trabalhista, AfD) | Fica em Política (nossa sugestão) ou vai para Geopolítica |
| **Onde colocar acidentes e desastres** | 1,9% (rodovias, aviação, tubarões, deslizamentos). Hoje caem em Segurança | Subcategoria de Segurança, ou categoria "Tragédias e acidentes" |

### 3.3 Avaliadas e não recomendadas como master

| Candidata | Tamanho | Melhor como |
|---|---|---|
| Clima e tempo | 1,3% | Subcategoria de Meio Ambiente |
| Automóveis | 1,2%, hoje dividido entre Economia e Transporte | Subcategoria de Transporte (e definir a fronteira) |
| Games | 0,6% | Subcategoria de Tecnologia (ou de Entretenimento) |
| Ciência | pequena e espalhada (espaço, estudos médicos) | Subcategoria de Tecnologia; estudos de saúde ficam em Saúde |

### 3.4 Conferência com o padrão IPTC

O IPTC Media Topics (padrão internacional de agências de notícias) tem 17 categorias de topo. Ele
confirma as duas propostas: tem **"conflito, guerra e paz"** separado de política, e junta
**"artes, cultura, entretenimento e mídia"** num nível mas separa entretenimento no nível de baixo.
Também tem como categorias de topo **"desastre e acidente"**, **"tempo"**, **"estilo de vida e
lazer"** e **"religião"**. Religião não apareceu como grupo relevante no nosso corpus.

---

## 4. Subcategorias (rascunho)

Tiradas dos grupos com volume real. Os nomes são provisórios.

| Categoria | Subcategorias candidatas |
|---|---|
| **Política** | Eleições e campanhas · Judiciário e STF · Congresso e legislação · Investigações e escândalos · Governos (federal, estadual, municipal) |
| **Geopolítica** | Oriente Médio · Rússia e Ucrânia · EUA, China e Ásia · Relações exteriores do Brasil · Organismos e blocos (ONU, BRICS, UE, OTAN) · África |
| **Economia** | Mercado financeiro · Inflação, juros e PIB · Impostos e contas públicas · Dívidas e finanças pessoais · Empresas e negócios · Comércio exterior e tarifas · Energia e petróleo |
| **Esportes** | Futebol brasileiro (clubes e campeonatos) · Futebol internacional · Seleções e Copa do Mundo · Mercado da bola · Tênis · Automobilismo · Outros esportes (rugby, críquete, atletismo) · Gestão e bastidores (FIFA, SAF) |
| **Tecnologia** | Inteligência artificial · Celulares e eletrônicos · Games · Espaço e ciência · Internet, apps e serviços digitais |
| **Segurança** | Violência urbana e homicídios · Violência contra a mulher · Tráfico e crime organizado · Corrupção e crimes financeiros · Golpes e fraudes · Roubos e furtos · Acidentes e tragédias (se não virar master) |
| **Saúde** | Doenças e pesquisa médica · Surtos e epidemias · Vacinação · SUS e sistema de saúde · Obesidade, nutrição e bem-estar |
| **Cultura** | Música, shows e festivais · Artes e patrimônio · Agenda cultural |
| **Entretenimento** | Famosos e influenciadores · Novelas e TV · Cinema e streaming |
| **Meio Ambiente** | Mudança climática e eventos extremos · Previsão do tempo · Biodiversidade e conservação · Energia e sustentabilidade |
| **Transporte** | Trânsito e obras viárias · Aviação · Automóveis |
| **Educação** | Avaliação e desempenho escolar · Cursos e qualificação profissional · Educação pelo mundo |
| **Moradia** | Mercado imobiliário e aluguel · Habitação social · Qualidade de vida urbana |
| **Direitos Humanos** | Refugiados e crises humanitárias · Racismo, gênero e discriminação · Populações civis em conflitos |

**Questão de desenho:** as subcategorias de Geopolítica estão por **região**, as outras por
**assunto**. Região é o jeito natural de acompanhar uma guerra, mas é a mesma mistura de geografia
com assunto que sugerimos evitar nas master. Vale decidir junto.

---

## 5. Ressalvas

1. **O dev pesa em fontes do Reino Unido e de Portugal.** Por isso aparecem críquete, rugby galês
   e o Partido Trabalhista. Em produção o tamanho das subcategorias pode ser outro; o das master
   tende a mudar menos.
2. **Os tamanhos são estimativas** (seção 2).
3. **Nada aqui foi validado com rotulagem.** Depois da decisão, o próximo experimento rotula de
   novo a amostra da 001 na taxonomia escolhida e mede se o modelo aprende as fronteiras novas.
   Política × Geopolítica é a mais difícil.

---

## 6. O que precisamos do time

1. Aprovar ou ajustar **Geopolítica** e **Entretenimento** como master.
2. Decidir as questões da seção 3.2, em especial **Loterias**.
3. Confirmar **geografia como filtro separado**, fora da taxonomia.
4. Revisar a lista de subcategorias da seção 4: nomes, o que falta e o que sobra.

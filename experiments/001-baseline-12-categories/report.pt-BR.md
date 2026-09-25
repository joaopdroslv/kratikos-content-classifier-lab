# Experimento 001: classificação de notícias com modelo próprio

**Data:** 2026-09-25 · **Ambiente:** banco e Qdrant de desenvolvimento (somente leitura) · **Fonte:** `data/001/evaluate.log`

---

## 1. Resumo

- **Hoje** a categoria das notícias vem da própria fonte (o portal) e **concorda com uma leitura
  cuidadosa do conteúdo em só ~1 de cada 3 notícias**.
- **Treinamos um modelo nosso**, leve, em cima dos embeddings que já calculamos na ingestão. Ele
  reproduz a categoria principal de um LLM de referência em **84%** das notícias. Em **97%** dos
  casos, a categoria principal dele está entre as categorias que o LLM atribuiu.
- **Esse modelo não chama nenhuma API por notícia.** O LLM só é usado para gerar os exemplos de
  treino, uma vez.
- **Conclusão:** a abordagem é viável e passa com folga a atribuição atual. Os próximos passos
  são testar em posts e incluir subcategorias.

---

## 2. Como o experimento foi feito

### A ideia: "professor e aluno"

| Papel | Quem é | O que faz |
|---|---|---|
| **Professor** (*teacher*) | LLM `gpt-4.1-mini` | Lê a notícia (título, descrição e início do texto) e atribui uma categoria **principal** e até duas **secundárias** |
| **Aluno** (*student*) | Regressão logística (um classificador simples) | Aprende a imitar o professor usando só o **embedding** da notícia, o vetor numérico que já existe no Qdrant |

O professor é caro e lento: é uma chamada de API por notícia. O aluno é praticamente gratuito e
instantâneo. Se o aluno imitar bem o professor, dá para classificar tudo sem chamar LLM no dia a
dia.

### Os dados

- **3.075 notícias** sorteadas entre as que já passaram pela ingestão (têm embedding):
  - 2.400 sorteadas **aleatoriamente**. É a fatia "uniforme", que representa o corpus real;
  - 675 sorteadas **a mais das categorias raras**, para o aluno ter exemplos suficientes delas.
- **Categorias:** as 12 que existem hoje (Política, Economia, Esportes, Tecnologia, Segurança,
  Saúde, Cultura, Transporte, Meio Ambiente, Educação, Moradia, Direitos Humanos).
- **Separação treino/teste:** 2.321 notícias para treinar e **754 para testar**. O aluno **nunca
  viu** as notícias de teste. A separação foi feita **por tópico**: notícias sobre o mesmo
  assunto ficam todas do mesmo lado, para o aluno não "colar" de uma notícia quase idêntica.

### O que foi comparado

| Candidato | O que é |
|---|---|
| **legacy** | A categoria que a fonte da notícia deu, a que usamos hoje |
| **zero_shot** | Sem treino: compara o embedding da notícia com o embedding da descrição de cada categoria e escolhe a mais parecida |
| **student** | O modelo treinado com os rótulos do professor |

---

## 3. Lendo o log, seção por seção

### 3.1 `3075 labelled articles with vectors`

Todas as 3.075 notícias foram rotuladas pelo professor sem falhas, e todas tinham embedding.

### 3.2 `Teacher, uniform stratum`: como o professor vê o corpus real

Distribuição da **categoria principal** segundo o LLM, na fatia aleatória:

| Categoria | % das notícias |
|---|---|
| Esportes | 26,3% |
| Política | 20,0% |
| Economia | 13,1% |
| Segurança | 10,3% |
| Cultura | 6,9% |
| Tecnologia | 6,3% |
| Saúde | 5,4% |
| Meio Ambiente | 3,8% |
| Transporte | 2,6% |
| Direitos Humanos | 2,0% |
| Educação | 1,2% |
| Moradia | 1,0% |
| *Nenhuma serve* | 1,2% |

> **Leitura:** pelo rótulo atual, Política parece ser quase metade do corpus (47%). Pelo
> conteúdo, **Esportes é a maior categoria**, e Política fica em 20%.

**Categorias secundárias por notícia:** 30% não têm nenhuma, 61% têm uma e 9% têm duas. Ou seja,
**a maioria das notícias toca mais de um assunto**, o que confirma que faz sentido permitir mais de
uma categoria por item.

**`missing_category`:** quando nenhuma categoria servia, o professor sugeria um nome. Isso
aconteceu pouco (1,2%). As sugestões que se repetiram foram **loteria** (11 vezes) e "notícias
gerais" (boletins com vários assuntos). É um sinal fraco para descobrir categorias novas, porque as
definições usadas no experimento são amplas: notícia de celebridade, por exemplo, cai em Cultura.

### 3.3 `Legacy vs teacher`: quão boa é a categoria de hoje

Para cada categoria que a fonte atribuiu: quantas vezes o professor concorda (**agrees**) e
para onde ele manda essas notícias (**->**).

| Categoria da fonte | Notícias | Concorda | Para onde o professor manda |
|---|---|---|---|
| Política | 1.007 | **29%** | Política 29%, **Esportes 22%**, Economia 13% |
| Economia | 490 | **29%** | Economia 29%, Política 13%, Segurança 12% |
| Esportes | 277 | 86% | Esportes 86%, Cultura 4%, Política 3% |
| Geral | 257 | (não existe) | Esportes 33%, Política 24%, Segurança 9% |
| Tecnologia | 152 | **22%** | Tecnologia 22%, Esportes 19%, Política 18% |
| Segurança | 58 | 64% | Segurança 64%, Política 21%, Transporte 5% |
| Cultura | 49 | 59% | Cultura 59%, Meio Ambiente 12%, Esportes 10% |
| Saúde | 46 | 54% | Saúde 54%, Segurança 15%, Política 11% |
| Transporte | 33 | 42% | Transporte 42%, Segurança 21%, Esportes 9% |
| Meio Ambiente | 13 | 77% | Meio Ambiente 77%, Cultura 15%, Esportes 8% |
| Educação | 10 | 20% | Segurança 40%, Saúde 20%, Educação 20% |
| Moradia | 8 | 38% | Moradia 38%, Segurança 25%, Meio Ambiente 25% |

> **Leitura:** as três maiores categorias da fonte (Política, Economia e Tecnologia) acertam
> **menos de 1 em cada 3**. "Política" em especial virou um depósito: 22% dela é esporte, quase
> tudo do feed da BBC. Só Esportes é confiável. As linhas com menos de ~50 notícias têm amostra
> pequena e servem só como indicação.

### 3.4 `train=2321 test=754 (grouped by topic)`

Tamanho da separação descrita na seção 2.

### 3.5 `Candidates vs teacher`: o resultado principal

| Candidato | Acerto da principal | ...na fatia aleatória | Principal dentro das categorias do professor | F1 micro | F1 macro |
|---|---|---|---|---|---|
| legacy (hoje) | 36,9% | 32,7% | 50,7% | 0,37 | 0,38 |
| zero_shot | 61,4% | 59,5% | 76,2% | 0,65 | 0,62 |
| **student (nosso)** | **82,9%** | **83,9%** | **96,8%** | **0,85** | **0,80** |

**O que cada métrica significa:**

- **Acerto da principal** (`primary_acc`): em quantas notícias a categoria principal do candidato
  é **exatamente** a do professor.
- **...na fatia aleatória** (`uniform`): a mesma coisa, só nas notícias sorteadas aleatoriamente.
  É o número que representa o corpus real, sem o reforço das categorias raras.
- **Principal dentro das categorias do professor** (`primary_in_set`): a categoria principal do
  candidato é **uma das** categorias que o professor deu, principal ou secundária. Exemplo: o
  professor disse "Política, com Economia de secundária" e o aluno disse "Economia". Não é acerto
  exato, mas também não está errado.
- **F1 micro / F1 macro:** medem o conjunto completo de categorias (principal + secundárias), de 0
  a 1. O **micro** pesa todas as notícias igualmente, então é dominado pelas categorias grandes. O
  **macro** dá o mesmo peso a cada categoria, então mostra o desempenho nas raras.

> **Leitura:** o aluno acerta a principal exata em ~84% dos casos, e em ~97% a escolha dele é uma
> das categorias certas. Quase toda a diferença é **troca entre principal e secundária**. Para
> sugerir conteúdo por preferência, isso quase não pesa: a notícia aparece para quem gosta de
> qualquer uma das duas. Comparado com hoje, é **mais que o dobro** de acerto. O zero-shot (sem
> treino) melhora sobre o legado, mas não basta.

### 3.6 `student: C=8.0 none_below=0.3`: configuração escolhida

Parâmetros ajustados automaticamente, **só com os dados de treino**:

- `C=8.0`: o quanto o modelo pode se ajustar aos exemplos. Foi escolhido por validação cruzada.
- `none_below=0.3`: se a confiança do aluno na melhor categoria ficar abaixo de 30%, ele não
  atribui nenhuma.

### 3.7 Tabela por categoria (aluno)

| Categoria | Exemplos no teste | Precisão | Cobertura | F1 |
|---|---|---|---|---|
| Esportes | 180 | 0,96 | 0,98 | **0,97** |
| Política | 239 | 0,89 | 0,88 | 0,89 |
| Segurança | 200 | 0,85 | 0,88 | 0,87 |
| Economia | 193 | 0,86 | 0,85 | 0,85 |
| Tecnologia | 112 | 0,93 | 0,77 | 0,84 |
| Saúde | 79 | 0,89 | 0,80 | 0,84 |
| Meio Ambiente | 93 | 0,84 | 0,84 | 0,84 |
| Cultura | 90 | 0,79 | 0,80 | 0,80 |
| Direitos Humanos | 92 | 0,69 | 0,82 | 0,75 |
| Transporte | 51 | 0,77 | 0,65 | 0,70 |
| Moradia | 30 | 0,64 | 0,70 | 0,67 |
| Educação | 21 | 0,52 | 0,76 | **0,62** |

- **Precisão:** quando o aluno diz "é Saúde", quantas vezes está certo.
- **Cobertura** (*recall*): de todas as notícias que são Saúde, quantas o aluno encontrou.
- **F1:** o equilíbrio entre as duas.

> **Leitura:** as categorias grandes vão muito bem (0,84 a 0,97). O ponto fraco são as **raras**
> (Educação, Moradia e Transporte), que tiveram poucos exemplos de treino.

### 3.8 `Learning curve`: quantos exemplos de treino são necessários

O mesmo aluno treinado com quantidades crescentes de exemplos:

| Exemplos de treino | Acerto da principal | F1 micro | F1 macro |
|---|---|---|---|
| 250 | 0,77 | 0,73 | 0,55 |
| 500 | 0,81 | 0,78 | 0,64 |
| 1.000 | 0,81 | 0,82 | 0,72 |
| 2.321 | 0,83 | 0,84 | 0,78 |

> **Leitura:** a categoria principal **estabiliza por volta de 500 exemplos**. O F1 macro, que
> representa as categorias raras, **continua subindo**. Mais dados ajudam justamente as raras, e o
> jeito mais eficiente é **rotular mais exemplos delas**, não mais do todo.

Esta tabela usa limiares padrão (0,5), então os números da última linha diferem um pouco dos da
seção 3.5, que usa limiares ajustados.

---

## 4. Ressalvas

1. **Tudo aqui mede concordância com o LLM, não acerto absoluto.** Se o professor erra, o aluno
   aprende o erro. Para medir o erro do próprio professor, há uma planilha com **150 notícias para
   revisão humana** (`data/001/review.csv`). Uma conferência rápida de 30 notícias pareceu correta,
   com poucos casos discutíveis.
2. **Só notícias.** Posts, principalmente os curtos, ainda não foram testados.
3. **Só as 12 categorias atuais.** Subcategorias ainda não foram testadas.
4. **Amostra de 3 mil notícias.** Os números por categoria rara (dezenas de exemplos) têm margem
   de erro grande.

---

## 5. Próximos passos

1. **Revisão humana** das 150 notícias, para medir o erro do professor e formar o primeiro
   "gabarito" oficial.
2. **Experimento 002, posts:** verificar se o aluno funciona em textos curtos.
3. **Experimento 003, subcategorias:** montar a taxonomia (base IPTC Media Topics + assuntos dos
   tópicos que já temos), rotular e repetir.
4. **Reforço das categorias raras:** rotular mais exemplos específicos delas.

---

## Glossário

| Termo | Significado |
|---|---|
| **Embedding** | Representação numérica do significado de um texto (um vetor de 3.072 números). Já calculamos um para cada notícia na ingestão |
| **Professor / teacher** | O LLM que gera os rótulos de referência |
| **Aluno / student** | O modelo nosso, que aprende a imitar o professor usando só o embedding |
| **Legacy** | A categoria atribuída hoje, vinda da fonte da notícia |
| **Zero-shot** | Classificar sem nenhum treino, só comparando com a descrição das categorias |
| **Fatia uniforme** | As notícias sorteadas de forma totalmente aleatória, que representam o corpus real |
| **Principal / secundária** | A categoria dominante da notícia / outras categorias que ela também cobre |
| **Precisão** | Das vezes que o modelo disse "X", quantas estavam certas |
| **Cobertura (recall)** | Das notícias que são "X", quantas o modelo encontrou |
| **F1** | Média equilibrada entre precisão e cobertura (0 a 1) |

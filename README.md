# Trabalho Prático 1 – Computação Evolucionária

Implementação de um **Algoritmo Genético (AG) com representação real** para minimização de funções de teste, desenvolvida para a disciplina de Computação Evolucionária.

O projeto executa experimentos com diferentes dimensões e métodos de seleção, imprime os resultados no terminal e salva **tabelas** (`.csv` e `.png`) e **gráficos** de convergência e diversidade.

## O que está implementado

| Componente | Opções |
|---|---|
| Seleção | Torneio (`torneio`) e Roleta (`roleta`) |
| Cruzamento | BLX-α (`blx`) e Ponto único (`single_point`) |
| Mutação | Gaussiana |
| Elitismo | Preserva os `n_elite` melhores indivíduos |
| Funções de teste | Esférica, Rastrigin e Rosenbrock |

## Pré-requisitos

- [Python](https://www.python.org/downloads/) 3.12 (versão em que o projeto foi testado)
- [Git](https://git-scm.com/)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/arthur-wolff/Trabalho-Pratico-1-Computacao-Evolucionaria.git
   cd Trabalho-Pratico-1-Computacao-Evolucionaria
   ```

2. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv .venv

   # Linux / macOS
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. Instale as dependências ([NumPy](https://numpy.org/) e [Matplotlib](https://matplotlib.org/)):
   ```bash
   pip install -r requirements.txt
   ```

## Como executar

> ⚠️ Execute **sempre a partir da raiz do projeto** e usando `-m`. Rodar `python src/main.py` gera o erro `ModuleNotFoundError: No module named 'src'`.

```bash
python -m src.main
```

A execução completa leva cerca de 20 segundos e roda dois experimentos:

1. **Experimento básico:** as três funções de teste com dimensões `n = 2, 5 e 10`, 10 execuções cada (sementes de 0 a 9).
2. **Experimento de seleção:** compara torneio × roleta na função Rastrigin com `n = 5`.

### Saídas geradas

Ao final, duas pastas são criadas no diretório onde o comando foi executado:

```
tabelas/    # tabelas em .csv e .png (melhor, média, desvio, NFE e tempo médio)
graficos/   # curvas de convergência e de diversidade da população
```

As pastas `tabelas/` e `graficos/` estão no `.gitignore` e não são versionadas.

## Parametrização

Os parâmetros ficam no dicionário `PARAMETRIZACAO` em `src/main.py`:

| Parâmetro | Significado | Valor padrão |
|---|---|---|
| `N` | Tamanho da população (deve ser **par**) | `100` |
| `pc` | Probabilidade de cruzamento | `0.9` |
| `pm` | Probabilidade de mutação | `0.1` |
| `sigma` | Desvio padrão da mutação gaussiana | `0.5` |
| `alpha` | Parâmetro α do cruzamento BLX-α | `0.5` |
| `k` | Tamanho do torneio | `3` |
| `n_elite` | Número de indivíduos preservados pelo elitismo | `2` |
| `metodo_selecao` | `"torneio"` ou `"roleta"` | `"torneio"` |
| `metodo_cruzamento` | `"blx"` ou `"single_point"` | `"blx"` |
| `low`, `high` | Limites do espaço de busca | `-5.0`, `5.0` |
| `nfe_max` | Máximo de avaliações da função objetivo | `20000` |

Outras constantes no mesmo arquivo: `DIMENSOES = (2, 5, 10)` e `N_EXECUCOES = 10`.

## Como rodar os testes

Cada arquivo em `tests/` é um script independente. Execute-os a partir da raiz do projeto, também com `-m`:

```bash
python -m tests.test_selecao
python -m tests.test_cruzamento
python -m tests.test_mutacao
python -m tests.test_elitismo
python -m tests.test_funcoes
python -m tests.test_criar_populacao
python -m tests.test_algoritimoGenetico
```

Cada um imprime uma mensagem de sucesso (por exemplo, `Todos os testes de selecao passaram`) e falha com erro caso algum teste não passe.

## Estrutura do projeto

```
├── docs/
│   ├── Trabalho Prático 1.pdf               # enunciado
│   └── TP1_CompEvolucionaria_Relatorio.pdf  # relatório
├── src/
│   ├── main.py                # experimentos, tabelas e gráficos (ponto de entrada)
│   ├── algoritmo_genetico.py  # laço principal do AG
│   ├── selecao.py             # torneio e roleta
│   ├── cruzamento.py          # BLX-α e ponto único
│   ├── mutacao.py             # mutação gaussiana
│   ├── elitismo.py            # elitismo
│   └── funcoes_teste.py       # esférica, Rastrigin e Rosenbrock
├── tests/                     # testes de cada módulo
├── requirements.txt           # dependências
└── README.md
```

## Problemas comuns

- **`ModuleNotFoundError: No module named 'src'`:** você está rodando o arquivo diretamente. Use `python -m src.main` a partir da raiz do projeto.
- **`python` não é reconhecido:** tente `python3` (Linux/macOS) ou reinstale o Python marcando *Add to PATH* (Windows).
- **Erro ao ativar o ambiente virtual no PowerShell:** execute `Set-ExecutionPolicy -Scope Process RemoteSigned` e tente de novo.
- **`ValueError: N deve ser par`:** ajuste o tamanho da população em `PARAMETRIZACAO`.

## Tecnologias

- Python 3.12
- NumPy
- Matplotlib
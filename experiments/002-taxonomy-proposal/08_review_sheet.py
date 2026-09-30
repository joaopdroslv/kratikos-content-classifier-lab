"""Step 8 — the taxonomy as a sheet for the team to approve, one row per subcategory.

Columns: master, subcategory, facet, definition (what the model reads), share in the validation,
and two columns for the reviewer to fill (`Aprovada?`, `Comentário`).

The share comes from step 7's answers, which were given under taxonomy 2.0. The 2.1 edits were
merges and renames, so 2.0 answers are carried over through `REMAP_2_0` instead of paying for a new
run: a merged subcategory gets the union of its parts. Two approximations follow, both flagged in
the sheet: `religion` is new and has no share, and `housing_access` counts every 2.0 `rent`
article, although rents as a market now belong to `real_estate`.

    uv run --with openpyxl python experiments/002-taxonomy-proposal/08_review_sheet.py
"""

from collections import Counter

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from lab import classify_v2
from lab.config import DATA_DIR
from lab.taxonomy_v2 import CATEGORIES, REMAP_2_0, SUBCATEGORIES, TAXONOMY_VERSION

DIR = DATA_DIR / "002"
VALIDATION = DIR / "validation_7cb234171d.jsonl"  # step 7, taxonomy 2.0
OUT = DIR / "taxonomia_v2_validacao.xlsx"
NO_SHARE = {
    "culture.religion": "nova na 2.1, sem dado de validação",
    "economy.cryptocurrencies": "nova na 2.2 (experimento 004), sem dado de validação",
}
NOTES = {
    "housing.housing_access": "inclui aluguel como mercado, que na 2.1 vai para Mercado imobiliário"
}
FACET = {"primary": "Principal", "cross_cutting": "Transversal"}

FONT = Font(name="Arial", size=10)
HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="305496")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
WRAP = Alignment(wrap_text=True, vertical="top")
HEADERS = (
    "Master",
    "Sub",
    "Tipo",
    "Definição (usada pelo modelo)",
    "% na validação",
    "Obs. sobre o %",
    "Aprovada?",
    "Comentário",
)
WIDTHS = (18, 34, 12, 70, 12, 30, 12, 40)


def shares() -> dict[str, float]:
    """Per 2.1 subcategory: the share of the articles of its master (as primary) that have it."""

    records = classify_v2.read(VALIDATION)
    per_master = Counter(r["primary"] for r in records)
    per_sub = Counter()
    for r in records:
        subs = {REMAP_2_0.get(s, s) for s in r["subcategories"]}
        per_sub.update(s for s in subs if s.startswith(f"{r['primary']}."))
    missing = set(per_sub) - set(SUBCATEGORIES)
    if missing:
        raise RuntimeError(f"2.0 subcategories with no 2.1 mapping: {sorted(missing)}")
    return {
        key: per_sub[key] / per_master[master.slug]
        for key, (master, _, _) in SUBCATEGORIES.items()
        if per_master[master.slug] and key not in NO_SHARE
    }


def taxonomy_sheet(ws, share: dict[str, float]) -> None:

    ws.title = "Taxonomia"
    ws.append(HEADERS)
    for master in CATEGORIES:
        if not master.facets:
            ws.append([master.name, "(sem subcategorias)", "", master.definition])
        for kind, facet in master.facets:
            for sub in facet.subcategories:
                key = f"{master.slug}.{sub.slug}"
                note = NO_SHARE.get(key) or NOTES.get(key)
                ws.append(
                    [
                        master.name,
                        sub.name,
                        FACET[kind],
                        sub.definition,
                        share.get(key),
                        note,
                    ]
                )

    for cell in ws[1]:
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font, cell.alignment = FONT, WRAP
        row[4].number_format = "0%"
        row[6].fill = row[7].fill = INPUT_FILL
    for column, width in zip("ABCDEFGH", WIDTHS):
        ws.column_dimensions[column].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    choice = DataValidation(type="list", formula1='"Sim,Não,Ajustar"', allow_blank=True)
    ws.add_data_validation(choice)
    choice.add(f"G2:G{ws.max_row}")


def instructions_sheet(ws, n_articles: int) -> None:

    lines = [
        ("Como validar", True),
        (
            "1. Na aba Taxonomia, preencha só as colunas amarelas: Aprovada? e Comentário.",
            False,
        ),
        ("2. Aprovada? aceita Sim, Não ou Ajustar (lista suspensa).", False),
        (
            "3. Em Comentário, diga o motivo de Não/Ajustar: renomear, juntar com outra sub, "
            "dividir, remover…",
            False,
        ),
        (
            '   Exemplo: Aprovada? = Ajustar · Comentário = "Juntar com Diplomacia e relações '
            'entre países"',
            False,
        ),
        ("", False),
        ("Colunas", True),
        (
            "Master: categoria principal. Sub: subcategoria (a master se repete em cada linha).",
            False,
        ),
        (
            "Tipo: Principal = divide a master por um eixo (ex.: modalidade do esporte); "
            "Transversal = aspecto que atravessa as principais (ex.: Transferências e contratos).",
            False,
        ),
        (
            "Um artigo pode receber várias subcategorias, de qualquer um dos dois tipos.",
            False,
        ),
        (
            f"% na validação: parcela dos artigos daquela master que receberam a sub, numa "
            f"classificação de {n_articles:,} notícias pelo gpt-4.1 (taxonomia 2.0, convertida "
            "para a 2.1). Valores perto de 0% são candidatos a juntar ou remover.".replace(
                ",", "."
            ),
            False,
        ),
        ("", False),
        (
            f"Taxonomia versão {TAXONOMY_VERSION} · experimento 002 (kratikos-content-classifier-lab)",
            False,
        ),
    ]
    ws.title = "Instruções"
    for text, is_title in lines:
        ws.append([text])
        ws.cell(ws.max_row, 1).font = Font(
            name="Arial", size=11 if is_title else 10, bold=is_title
        )
    ws.column_dimensions["A"].width = 120


def main() -> None:

    wb = Workbook()
    taxonomy_sheet(wb.active, shares())
    instructions_sheet(wb.create_sheet(), len(classify_v2.read(VALIDATION)))
    wb.save(OUT)
    print(
        f"wrote {OUT} (taxonomy {TAXONOMY_VERSION}, {len(SUBCATEGORIES)} subcategories)"
    )


if __name__ == "__main__":
    main()

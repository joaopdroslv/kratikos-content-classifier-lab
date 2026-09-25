"""The 12 categories that exist today in `public.categories`, as the labels of experiment 001.

`name` is the exact `public.categories.name` (pt-BR, user-facing) and the value the news source
writes into `news_articles.category`. `definition` is what the teacher LLM and the zero-shot
baseline read: the table's own descriptions are one-liners too vague to label against, so the
boundaries that are actually contested (international news, crime vs. politics, business vs.
technology) are written out here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:

    slug: str
    name: str
    definition: str


CATEGORIES: tuple[Category, ...] = (
    Category(
        "politics",
        "Política",
        "Government, elections, parties, politicians, legislation, public policy, courts ruling on "
        "political matters, diplomacy, international relations, geopolitics and armed conflicts "
        "between states.",
    ),
    Category(
        "economy",
        "Economia",
        "Macroeconomy, markets, interest rates, inflation, taxes, public budget, trade, companies "
        "and business, jobs and labour, consumer prices, personal finance.",
    ),
    Category(
        "sports",
        "Esportes",
        "Sports competitions, athletes, clubs, coaches, transfers, match results and sports "
        "federations.",
    ),
    Category(
        "technology",
        "Tecnologia",
        "Technology products, software, artificial intelligence, the internet and social "
        "platforms, telecom, startups, digital security and science/space research.",
    ),
    Category(
        "public_safety",
        "Segurança",
        "Crime, policing, violence, arrests, organized crime, prisons and public-safety policy.",
    ),
    Category(
        "health",
        "Saúde",
        "Public health, diseases and outbreaks, hospitals and the health system, medicine, "
        "treatments, nutrition, fitness and mental health.",
    ),
    Category(
        "culture",
        "Cultura",
        "Arts, music, film, television, literature, theatre, festivals, heritage, museums and the "
        "people who make them.",
    ),
    Category(
        "transport",
        "Transporte",
        "Urban mobility, public transport, roads and traffic, aviation, railways, ports, vehicles, "
        "and transport accidents as transport events.",
    ),
    Category(
        "environment",
        "Meio Ambiente",
        "Climate change, extreme weather and natural disasters, pollution, deforestation, "
        "biodiversity, animals, energy transition and sustainability.",
    ),
    Category(
        "education",
        "Educação",
        "Schools, universities, teachers, students, exams, education policy and research "
        "institutions as educational bodies.",
    ),
    Category(
        "housing",
        "Moradia",
        "Housing, real estate, rent, housing policy, urban planning and homelessness.",
    ),
    Category(
        "human_rights",
        "Direitos Humanos",
        "Civil rights, equality, discrimination and racism, LGBTQ+ rights, indigenous peoples, "
        "migrants and refugees, humanitarian crises and freedom of expression.",
    ),
)

SLUGS: tuple[str, ...] = tuple(category.slug for category in CATEGORIES)
BY_NAME: dict[str, Category] = {category.name: category for category in CATEGORIES}
BY_SLUG: dict[str, Category] = {category.slug: category for category in CATEGORIES}


def slug_for_source_name(name: str | None) -> str | None:
    """Maps a `news_articles.category` value to a slug; `Geral` (no category row) maps to None."""

    category = BY_NAME.get(name or "")
    return category.slug if category else None

"""Taxonomy v2 — 16 master categories with faceted, multi-label subcategories.

Proposed by experiment 002 and still pending a team decision; see
experiments/002-taxonomy-proposal/ (README, proposal.pt-BR.md). `taxonomy.py` stays untouched: it
defines the labels of 001, which must remain reproducible.

Masters, from v1: `politics` loses relations between states, wars and diplomacy to `geopolitics`
and keeps the internal politics of ANY country (geography is not a category axis); `culture` loses
celebrities, TV and film to `entertainment`; `economy` loses lottery draws to `lotteries`; travel,
food, fashion and similar go to the new `lifestyle`.

Subcategories: each master has a **primary** facet split along one axis (the modality of a sport,
the subject of a political story) and may have a **cross-cutting** facet of aspects that recur
across the primary ones. An article can take several subcategories, from either facet. They are
curated by hand from the output of steps 5–6 of 002 (`data/002/subcategories.json`); the changes
from that output and their reasons are in the 002 README.

Boundaries between masters that the subcategories rely on:
- accidents and tragedies caused by people or machines → public_safety; natural disasters →
  environment;
- music as art, concerts and festivals → culture; the personal life of musicians → entertainment;
- all film and series → entertainment;
- vehicles and the automotive market → transport; digital crime → public_safety;
- health of a person → health; beauty, self-care and relationships → lifestyle;
- rights of migrants and refugees → human_rights; migration policy between states → geopolitics;
- religion as faith and institution → culture; religious freedom and intolerance → human_rights.

2.1 (2026-09-29, manual review): human_rights split by type of issue instead of protected group
(racial/ethnic/religious, LGBTQIA+ and women's rights merged into `discrimination_equality`);
`religion` added to culture; rent, social housing and homelessness merged into `housing_access`;
`weather` renamed so "clima" means only climate; the health system named without SUS.

2.2 (2026-09-30, after 004's coverage sweep): `economy.cryptocurrencies` added; readers follow it
as a subject, and it was diluted in `financial_markets`. Named "Criptomoedas" only: "ativos
digitais" would also read as stocks and funds traded online.
"""

from dataclasses import dataclass

# Bump on ANY change to a master or subcategory (name, definition, added, removed): LLM answers
# record it, and answers given under another version are not comparable.
TAXONOMY_VERSION = "2.2"


@dataclass(frozen=True)
class Subcategory:

    slug: str
    name: str
    definition: str


@dataclass(frozen=True)
class Facet:

    axis: str
    subcategories: tuple[Subcategory, ...]


@dataclass(frozen=True)
class Master:

    slug: str
    name: str
    definition: str
    primary: Facet | None = None
    cross_cutting: Facet | None = None

    @property
    def facets(self) -> tuple[tuple[str, Facet], ...]:

        return tuple(
            (kind, facet)
            for kind, facet in (
                ("primary", self.primary),
                ("cross_cutting", self.cross_cutting),
            )
            if facet
        )


S = Subcategory

CATEGORIES: tuple[Master, ...] = (
    Master(
        "politics",
        "Política",
        "The internal politics of any country: government and public administration, elections "
        "and campaigns, parties and politicians, legislatures and legislation, public policy, "
        "courts and investigations with political stakes. Relations BETWEEN countries belong to "
        "geopolitics.",
        primary=Facet(
            "political subject",
            (
                S(
                    "elections",
                    "Eleições e campanhas",
                    "Elections, campaigns, candidacies, polls and electoral rules and courts.",
                ),
                S(
                    "corruption_investigations",
                    "Corrupção e investigações",
                    "Corruption, political scandals, and investigations and trials of politicians "
                    "and officials.",
                ),
                S(
                    "democracy_institutions",
                    "Democracia e instituições",
                    "Clashes between the branches of power, court rulings with political stakes, "
                    "impeachment, coups, constitutional change and political rights.",
                ),
                S(
                    "government_administration",
                    "Governo e gestão pública",
                    "Decisions and conduct of governments at every level: appointments, cabinet, "
                    "public administration, public servants and the politics of spending. "
                    "Economic indicators and taxes belong to economy.",
                ),
                S(
                    "bills_reforms",
                    "Projetos de lei e reformas",
                    "Bills, votes and reforms going through legislatures, on any subject.",
                ),
                S(
                    "parties_alliances",
                    "Partidos e alianças",
                    "Life of political parties: leadership disputes, coalitions and alliances, "
                    "party switching, ideology.",
                ),
                S(
                    "polarisation_extremism",
                    "Polarização, protestos e extremismo",
                    "Political violence, extremism, protests and mobilisations, and political "
                    "disinformation.",
                ),
            ),
        ),
    ),
    Master(
        "geopolitics",
        "Geopolítica",
        "Relations between countries and the international order: diplomacy, summits and foreign "
        "policy, wars and armed conflicts between states or with armed groups, military alliances "
        "and defence, sanctions, international organisations (UN, NATO, BRICS, EU as a bloc) and "
        "a country's position in the world.",
        primary=Facet(
            "type of international issue",
            (
                S(
                    "wars_armed_conflicts",
                    "Guerras e conflitos armados",
                    "Hostilities, attacks, military operations and casualties of wars and armed "
                    "conflicts.",
                ),
                S(
                    "peace_ceasefires",
                    "Negociações de paz e cessar-fogo",
                    "Peace talks, ceasefires, truces and mediation to end a conflict.",
                ),
                S(
                    "diplomacy_bilateral_relations",
                    "Diplomacia e relações entre países",
                    "Summits, state visits, bilateral relations and foreign-policy positions, "
                    "outside peace talks.",
                ),
                S(
                    "defence_weapons",
                    "Defesa, alianças militares e armamentos",
                    "Military alliances, defence policy and spending, arms deals, and nuclear and "
                    "strategic weapons programmes.",
                ),
                S(
                    "sanctions_trade_disputes",
                    "Sanções e disputas comerciais",
                    "Sanctions, tariffs and trade wars used as foreign policy. Their effect on "
                    "prices and markets belongs to economy.",
                ),
                S(
                    "international_organisations",
                    "Organismos e direito internacional",
                    "The UN, blocs such as BRICS or the EU, international courts, treaties and "
                    "global governance.",
                ),
                S(
                    "migration_borders",
                    "Migração e fronteiras",
                    "Migration policy between states, border control and asylum systems. The "
                    "rights and plight of migrants belong to human_rights.",
                ),
            ),
        ),
    ),
    Master(
        "economy",
        "Economia",
        "Macroeconomy, markets, interest rates, inflation, taxes, public budget, trade and tariffs, "
        "companies and business, jobs and labour, energy and commodity markets, consumer prices, "
        "personal finance and debt.",
        primary=Facet(
            "economic domain",
            (
                S(
                    "macroeconomy",
                    "Inflação, juros e PIB",
                    "Inflation, interest rates, central banks, growth, and economic forecasts.",
                ),
                S(
                    "public_finance_taxes",
                    "Contas públicas e impostos",
                    "Public budget, deficit and debt, taxes and tax reform, as economics.",
                ),
                S(
                    "financial_markets",
                    "Mercado financeiro e investimentos",
                    "Stock exchanges, currencies, investments, banks and financial results. "
                    "Cryptocurrencies belong to cryptocurrencies.",
                ),
                S(
                    "cryptocurrencies",
                    "Criptomoedas",
                    "Bitcoin and other cryptocurrencies and stablecoins: prices, investment, "
                    "crypto exchanges and their regulation. Stocks, funds and other securities, "
                    "even when traded online, belong to financial_markets.",
                ),
                S(
                    "companies_business",
                    "Empresas e negócios",
                    "Companies, sectors, entrepreneurship, mergers, bankruptcies and corporate "
                    "restructuring.",
                ),
                S(
                    "labour_income",
                    "Trabalho, emprego e renda",
                    "Jobs and unemployment, wages, labour rights and reforms, income and social "
                    "benefits.",
                ),
                S(
                    "cost_of_living",
                    "Custo de vida e consumo",
                    "Consumer prices (food, fuel, energy bills), consumption and consumer rights.",
                ),
                S(
                    "personal_finance",
                    "Finanças pessoais e dívidas",
                    "Household debt and renegotiation, credit, income-tax filing and refunds, "
                    "savings and retirement.",
                ),
                S(
                    "energy_commodities",
                    "Energia e commodities",
                    "Oil, gas, electricity and mining markets and companies, and commodity prices.",
                ),
                S(
                    "agribusiness",
                    "Agronegócio",
                    "Farming and livestock as an economic sector: harvests, exports, prices.",
                ),
                S(
                    "foreign_trade",
                    "Comércio exterior",
                    "Exports, imports and trade agreements, and the economic effect of tariffs.",
                ),
            ),
        ),
    ),
    Master(
        "lotteries",
        "Loterias",
        "Lottery draws, results, jackpots and prizes (Mega-Sena, Lotofácil, Quina and similar).",
    ),
    Master(
        "sports",
        "Esportes",
        "Sports competitions, athletes, clubs, coaches, transfers, match results, sports "
        "federations and the business of sport. An athlete's private or romantic life belongs to "
        "entertainment.",
        primary=Facet(
            "sport modality",
            (
                S("football", "Futebol", "Association football, men's and women's."),
                S("tennis", "Tênis", "Tennis and other racket sports."),
                S(
                    "motorsport",
                    "Automobilismo",
                    "Formula 1, stock cars, rallies and motorcycle racing.",
                ),
                S("basketball", "Basquete", "Basketball."),
                S("volleyball", "Vôlei", "Indoor and beach volleyball."),
                S(
                    "combat_sports",
                    "Lutas",
                    "Boxing, MMA, judo and other combat sports.",
                ),
                S(
                    "athletics_running",
                    "Atletismo e corrida",
                    "Track and field, marathons and street running.",
                ),
                S("cycling", "Ciclismo", "Road, track and mountain cycling."),
                S(
                    "water_sports",
                    "Esportes aquáticos",
                    "Swimming, surfing, sailing, canoeing and other water sports.",
                ),
                S("rugby", "Rúgbi", "Rugby union and league."),
                S("cricket", "Críquete", "Cricket."),
                S("golf", "Golfe", "Golf."),
                S(
                    "multisport_games",
                    "Jogos Olímpicos e multiesportivos",
                    "The Olympic and Paralympic Games and other multi-sport games as a whole.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "aspect of the sport",
            (
                S(
                    "transfers_contracts",
                    "Transferências e contratos",
                    "Signings, transfers, loans, contract renewals and the transfer market.",
                ),
                S(
                    "management_finances",
                    "Gestão, finanças e bastidores",
                    "Club and federation management, finances, ownership, governance and "
                    "sports-policy disputes.",
                ),
                S(
                    "womens_sport",
                    "Esporte feminino",
                    "Women's competitions, teams and athletes, in any modality.",
                ),
            ),
        ),
    ),
    Master(
        "technology",
        "Tecnologia",
        "Technology products, software, artificial intelligence, the internet, apps and social "
        "platforms, telecom, games as technology products, startups, digital services, and "
        "science and space research.",
        primary=Facet(
            "technology domain",
            (
                S(
                    "artificial_intelligence",
                    "Inteligência artificial",
                    "AI models, products, "
                    "companies and their effects on work and society.",
                ),
                S(
                    "internet_platforms",
                    "Internet, redes sociais e apps",
                    "Social networks, apps, online and government digital services.",
                ),
                S(
                    "consumer_electronics",
                    "Celulares e eletrônicos",
                    "Smartphones, computers, gadgets and their launches and prices.",
                ),
                S(
                    "games",
                    "Games",
                    "Video games, consoles and the games industry.",
                ),
                S(
                    "space_science",
                    "Espaço e ciência",
                    "Space exploration, astronomy and scientific discoveries outside medicine.",
                ),
                S(
                    "telecom_connectivity",
                    "Telecomunicações e conectividade",
                    "Mobile networks, 5G, broadband, satellites for connectivity.",
                ),
                S(
                    "industry_robotics",
                    "Robótica, chips e indústria tech",
                    "Robotics, automation, semiconductors, data centres and the tech industry.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "issue raised by technology",
            (
                S(
                    "regulation_ethics",
                    "Regulação e ética",
                    "Laws, regulation, antitrust and ethical debates about technology.",
                ),
                S(
                    "security_privacy",
                    "Segurança digital e privacidade",
                    "Data protection, leaks and cybersecurity. Scams and digital crime belong to "
                    "public_safety.",
                ),
            ),
        ),
    ),
    Master(
        "public_safety",
        "Segurança",
        "Crime, policing, violence, arrests, organised crime, fraud and scams, prisons, "
        "public-safety policy, and accidents and tragedies caused by people or machines (road, "
        "air, drownings, collapses, fires). Natural disasters belong to environment.",
        primary=Facet(
            "type of occurrence",
            (
                S(
                    "violent_crime",
                    "Crimes violentos",
                    "Homicides, shootings, assaults, kidnappings and disappearances.",
                ),
                S(
                    "violence_against_women",
                    "Violência contra a mulher",
                    "Femicide, domestic violence and harassment of women.",
                ),
                S(
                    "sexual_violence_child_abuse",
                    "Violência sexual e abuso infantil",
                    "Rape, sexual abuse and exploitation, and abuse of children.",
                ),
                S(
                    "organised_crime_drugs",
                    "Crime organizado e tráfico",
                    "Criminal factions, militias, drug and arms trafficking, money laundering.",
                ),
                S(
                    "theft_fraud",
                    "Roubos, furtos e golpes",
                    "Robbery, theft, scams, fraud and digital crime against people and firms.",
                ),
                S(
                    "accidents_tragedies",
                    "Acidentes e tragédias",
                    "Road, air and other transport accidents, drownings, fires, collapses.",
                ),
                S(
                    "policing_prisons",
                    "Polícia, prisões e política de segurança",
                    "Police operations and conduct, prisons, and public-safety policy.",
                ),
            ),
        ),
    ),
    Master(
        "health",
        "Saúde",
        "Public health, diseases and outbreaks, vaccination, hospitals and the health system, "
        "medicine and treatments, medical research, nutrition, fitness and mental health.",
        primary=Facet(
            "health domain",
            (
                S(
                    "infectious_diseases_vaccines",
                    "Doenças infecciosas e vacinas",
                    "Outbreaks, epidemics, infectious diseases and vaccination.",
                ),
                S(
                    "chronic_diseases",
                    "Doenças crônicas e câncer",
                    "Cancer, heart disease, diabetes, dementia and other chronic conditions.",
                ),
                S(
                    "treatments_medicines",
                    "Tratamentos e medicamentos",
                    "Medicines, therapies, surgery and medical technology.",
                ),
                S(
                    "mental_health",
                    "Saúde mental",
                    "Mental health, psychiatric conditions and emotional wellbeing.",
                ),
                S(
                    "nutrition_fitness",
                    "Nutrição, exercício e obesidade",
                    "Diet, physical activity, obesity and weight-loss treatments.",
                ),
                S(
                    "health_system",
                    "Sistemas e serviços de saúde",
                    "Public and private health services (the SUS, health insurance plans, other "
                    "countries' systems), hospitals, access, waiting lists and health workers.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "health approach",
            (
                S(
                    "prevention",
                    "Prevenção",
                    "Screening, awareness campaigns and healthy habits to prevent disease.",
                ),
                S(
                    "medical_research",
                    "Pesquisa médica",
                    "Scientific studies and discoveries in medicine.",
                ),
                S(
                    "health_policy",
                    "Política e regulação da saúde",
                    "Health policy, regulators and rules on medicines and services.",
                ),
            ),
        ),
    ),
    Master(
        "culture",
        "Cultura",
        "Music, concerts and festivals, visual arts, literature, theatre and dance, heritage, "
        "museums, religion, cultural agenda and cultural policy, and the artists who make them. "
        "Film and series belong to entertainment.",
        primary=Facet(
            "cultural field",
            (
                S(
                    "music",
                    "Música",
                    "Music, artists' work, albums, concerts and music festivals. Musicians' "
                    "personal lives belong to entertainment.",
                ),
                S(
                    "visual_arts",
                    "Artes visuais",
                    "Painting, sculpture, photography, design and exhibitions.",
                ),
                S(
                    "literature",
                    "Livros e literatura",
                    "Books, authors, literary prizes.",
                ),
                S(
                    "performing_arts",
                    "Teatro e dança",
                    "Theatre, dance, circus and other performing arts.",
                ),
                S(
                    "heritage_history",
                    "Patrimônio e história",
                    "Museums, historic heritage, archaeology, traditions and popular festivals.",
                ),
                S(
                    "religion",
                    "Religião",
                    "Religions, churches and religious leaders, faith and religious celebrations. "
                    "Religious freedom and intolerance belong to human_rights.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "support for culture",
            (
                S(
                    "cultural_policy",
                    "Política e fomento cultural",
                    "Cultural policy, public funding, incentive laws and cultural institutions.",
                ),
            ),
        ),
    ),
    Master(
        "entertainment",
        "Entretenimento",
        "Celebrities and influencers and their private lives (including athletes' and "
        "musicians'), television, soap operas and reality shows, film and series, streaming and "
        "awards, gossip.",
        primary=Facet(
            "entertainment format",
            (
                S(
                    "celebrities_influencers",
                    "Famosos e influenciadores",
                    "Celebrities, influencers and digital creators: relationships, family, "
                    "controversies and public image.",
                ),
                S(
                    "tv_soaps_reality",
                    "TV, novelas e reality shows",
                    "Television programmes, soap operas and reality shows.",
                ),
                S(
                    "film_series",
                    "Filmes e séries",
                    "Films and series, cinemas and streaming releases, and film and TV awards.",
                ),
            ),
        ),
    ),
    Master(
        "lifestyle",
        "Estilo de Vida",
        "Travel and tourism, food and recipes, fashion and beauty, home and decoration, pets, "
        "astrology, relationships and personal wellbeing. Medical health belongs to health.",
        primary=Facet(
            "lifestyle domain",
            (
                S(
                    "travel_tourism",
                    "Viagens e turismo",
                    "Destinations, travel tips and the tourism experience.",
                ),
                S(
                    "food_recipes",
                    "Gastronomia e receitas",
                    "Recipes, restaurants and food culture.",
                ),
                S("fashion_beauty", "Moda e beleza", "Fashion, beauty and self-care."),
                S(
                    "home_decor",
                    "Casa e decoração",
                    "Home, decoration, gardening and DIY.",
                ),
                S("pets", "Animais de estimação", "Pets and their care."),
                S(
                    "astrology",
                    "Astrologia e horóscopo",
                    "Horoscopes, astrology and esoteric content.",
                ),
                S(
                    "relationships_behaviour",
                    "Relacionamentos e comportamento",
                    "Love, family, sexuality and behaviour, outside mental health.",
                ),
            ),
        ),
    ),
    Master(
        "transport",
        "Transporte",
        "Urban mobility, public transport, roads and traffic, road works, aviation as a service, "
        "railways, ports, and vehicles and the automotive market. Accidents belong to "
        "public_safety.",
        primary=Facet(
            "mode of transport",
            (
                S(
                    "aviation",
                    "Aviação",
                    "Airlines, airports, routes, fares and flight disruptions.",
                ),
                S(
                    "roads_traffic",
                    "Rodovias e trânsito",
                    "Highways, traffic, road works, tolls and traffic rules.",
                ),
                S(
                    "urban_mobility",
                    "Mobilidade urbana e transporte público",
                    "Buses, metro, urban mobility, cycling and walking in cities.",
                ),
                S(
                    "automotive",
                    "Carros e mercado automotivo",
                    "Vehicle launches, electric and hybrid cars, car sales and the car industry.",
                ),
                S(
                    "rail_maritime",
                    "Ferrovias, portos e navegação",
                    "Railways, ports, shipping and waterways.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "aspect of transport",
            (
                S(
                    "infrastructure_concessions",
                    "Infraestrutura e concessões",
                    "Construction, concessions and investment in transport infrastructure.",
                ),
                S(
                    "service_quality",
                    "Qualidade do serviço e acessibilidade",
                    "Delays, overcrowding, fares, accessibility and users' rights.",
                ),
            ),
        ),
    ),
    Master(
        "environment",
        "Meio Ambiente",
        "Climate change, weather and forecasts, extreme weather and natural disasters, pollution, "
        "deforestation, biodiversity, animals, energy transition and sustainability.",
        primary=Facet(
            "environmental issue",
            (
                S(
                    "climate_change",
                    "Mudanças climáticas",
                    "Global warming, emissions, climate science and climate policy and summits.",
                ),
                S(
                    "weather",
                    "Tempo e previsão",
                    "Forecasts, cold and heat waves, rain and weather alerts as weather.",
                ),
                S(
                    "natural_disasters",
                    "Desastres naturais",
                    "Floods, landslides, droughts, wildfires, storms and earthquakes and the "
                    "damage they cause.",
                ),
                S(
                    "biodiversity_animals",
                    "Biodiversidade e animais",
                    "Wildlife, species, conservation and ecosystems.",
                ),
                S(
                    "deforestation_land",
                    "Desmatamento e uso da terra",
                    "Deforestation, protected areas, land use and environmental crime.",
                ),
                S(
                    "pollution_sanitation",
                    "Poluição e saneamento",
                    "Air, water and plastic pollution, waste and sanitation.",
                ),
                S(
                    "energy_transition",
                    "Transição energética e sustentabilidade",
                    "Renewable energy, sustainability and the green economy.",
                ),
            ),
        ),
    ),
    Master(
        "education",
        "Educação",
        "Schools, universities, teachers, students, exams and assessments, vocational courses, "
        "education policy.",
        primary=Facet(
            "educational stage",
            (
                S(
                    "basic_education",
                    "Educação infantil e fundamental",
                    "Day care, preschool and primary and lower-secondary school.",
                ),
                S(
                    "high_school",
                    "Ensino médio",
                    "Upper-secondary school and its reform.",
                ),
                S(
                    "higher_education",
                    "Ensino superior",
                    "Universities, colleges, graduate studies, student funding and admission.",
                ),
                S(
                    "vocational_courses",
                    "Cursos técnicos e profissionalizantes",
                    "Technical and vocational education, free courses and job training.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "aspect of education",
            (
                S(
                    "exams_assessment",
                    "Provas e avaliações",
                    "Entrance exams, national exams and school performance indicators.",
                ),
                S(
                    "teachers",
                    "Professores",
                    "Teachers' training, pay, careers and strikes.",
                ),
                S(
                    "education_policy",
                    "Política educacional e financiamento",
                    "Education policy, funding, school management and violence in schools.",
                ),
            ),
        ),
    ),
    Master(
        "housing",
        "Moradia",
        "Housing, real estate and the rental market, access to housing, housing policy, urban "
        "planning and construction.",
        primary=Facet(
            "housing issue",
            (
                S(
                    "real_estate",
                    "Mercado imobiliário e aluguel",
                    "Buying, selling and renting property as a market: prices, rents, launches, "
                    "financing and short-term rentals.",
                ),
                S(
                    "housing_access",
                    "Acesso à moradia",
                    "Affordability, tenants' rights and tenancy law, housing programmes and "
                    "social housing, evictions, precarious housing and homelessness.",
                ),
                S(
                    "urban_planning",
                    "Urbanismo e planejamento urbano",
                    "Zoning, master plans, urban renewal and neighbourhood infrastructure.",
                ),
                S(
                    "construction",
                    "Construção",
                    "Construction of homes, building methods and the construction sector.",
                ),
            ),
        ),
    ),
    Master(
        "human_rights",
        "Direitos Humanos",
        "Civil rights, equality, discrimination and racism, gender and LGBTQ+ rights, indigenous "
        "peoples, migrants and refugees, humanitarian crises and freedom of expression.",
        primary=Facet(
            "type of rights issue",
            (
                S(
                    "civil_political_rights",
                    "Liberdades civis",
                    "Freedom of expression and of the press, privacy, and abuses of state power.",
                ),
                S(
                    "discrimination_equality",
                    "Discriminação e igualdade",
                    "Discrimination and equal rights of any group: racism, indigenous peoples and "
                    "ethnic minorities, gender equality and women's rights, LGBTQIA+ rights, "
                    "religious freedom and intolerance. Violence against women as crime belongs "
                    "to public_safety.",
                ),
                S(
                    "migrants_refugees",
                    "Migrantes e refugiados",
                    "The rights and plight of migrants and refugees.",
                ),
                S(
                    "children_vulnerable",
                    "Crianças e grupos vulneráveis",
                    "Rights of children, the elderly, people with disabilities and other "
                    "vulnerable groups.",
                ),
            ),
        ),
        cross_cutting=Facet(
            "type of issue",
            (
                S(
                    "humanitarian_crises",
                    "Crises humanitárias",
                    "Famine, displacement and humanitarian aid in crises.",
                ),
                S(
                    "activism",
                    "Ativismo",
                    "Campaigns, movements and awareness dates for rights.",
                ),
            ),
        ),
    ),
)

# Subcategory keys of 2.0 that 2.1 merged or renamed -> their key now; any other 2.0 key is
# unchanged. Reads answers given under 2.0 (002's validation) in today's keys.
REMAP_2_0 = {
    "human_rights.racial_ethnic_religious": "human_rights.discrimination_equality",
    "human_rights.lgbtq": "human_rights.discrimination_equality",
    "human_rights.gender_women": "human_rights.discrimination_equality",
    "housing.rent": "housing.housing_access",
    "housing.social_housing": "housing.housing_access",
    "housing.homelessness": "housing.housing_access",
}

SLUGS: tuple[str, ...] = tuple(master.slug for master in CATEGORIES)
BY_SLUG: dict[str, Master] = {master.slug: master for master in CATEGORIES}
# `master.subcategory` -> (master, facet kind, subcategory)
SUBCATEGORIES: dict[str, tuple[Master, str, Subcategory]] = {
    f"{master.slug}.{s.slug}": (master, kind, s)
    for master in CATEGORIES
    for kind, facet in master.facets
    for s in facet.subcategories
}

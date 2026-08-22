import random
import string
from datetime import date, timedelta

FIRST_NAMES_F = ["Ana", "Beatriz", "Carla", "Diana", "Elena", "Fernanda", "Gabriela",
                 "Helena", "Isabel", "Julia", "Karina", "Larissa", "Mariana", "Natalia",
                 "Olivia", "Patricia", "Renata", "Sofia", "Tatiana", "Valentina"]
FIRST_NAMES_M = ["Andre", "Bruno", "Carlos", "Diego", "Eduardo", "Felipe", "Gustavo",
                 "Henrique", "Igor", "Joao", "Kevin", "Lucas", "Marcelo", "Nicolas",
                 "Otavio", "Paulo", "Rafael", "Samuel", "Thiago", "Vitor"]
LAST_NAMES = ["Almeida", "Barbosa", "Cardoso", "Dias", "Esteves", "Ferreira", "Gomes",
              "Henriques", "Imperatriz", "Jesus", "Klein", "Lima", "Machado", "Nunes",
              "Oliveira", "Pacheco", "Queiroz", "Ribeiro", "Santos", "Teixeira"]
DOMAINS = ["gmail.com", "outlook.com", "proton.me", "yahoo.com", "fastmail.com",
           "corp.example", "acme.io"]
CITIES = [("Sao Paulo", "BR"), ("Rio de Janeiro", "BR"), ("Lisbon", "PT"),
          ("Porto", "PT"), ("Austin", "US"), ("Seattle", "US"), ("Berlin", "DE"),
          ("Amsterdam", "NL"), ("Dublin", "IE"), ("Tokyo", "JP")]
JOBS = ["Software Engineer", "SRE", "Data Analyst", "Product Manager", "Designer",
        "QA Engineer", "Architect", "Support Agent", "Sales Rep", "Recruiter"]
SEXES = ["female", "male"]

EPOCH = date(1955, 1, 1)
AGE_SPAN_DAYS = 365 * 55


class PersonGenerator:
    def __init__(self, seed=42):
        self._rnd = random.Random(seed)

    def one(self, person_id):
        rnd = self._rnd
        sex = rnd.choice(SEXES)
        first = rnd.choice(FIRST_NAMES_F if sex == "female" else FIRST_NAMES_M)
        last = rnd.choice(LAST_NAMES)
        city, country = rnd.choice(CITIES)
        birth = EPOCH + timedelta(days=rnd.randrange(AGE_SPAN_DAYS))
        return {
            "id": person_id,
            "first_name": first,
            "last_name": last,
            "sex": sex,
            "email": self._email(first, last, person_id),
            "birth_date": birth.isoformat(),
            "age": self._age(birth),
            "phone": self._phone(country),
            "city": city,
            "country": country,
            "job_title": rnd.choice(JOBS),
            "salary": rnd.randrange(40, 260) * 1000,
            "account_id": self._account_id(),
        }

    def many(self, count, start_id=1):
        for person_id in range(start_id, start_id + count):
            yield self.one(person_id)

    def _email(self, first, last, person_id):
        domain = self._rnd.choice(DOMAINS)
        return f"{first.lower()}.{last.lower()}{person_id}@{domain}"

    def _phone(self, country):
        rnd = self._rnd
        return f"+{rnd.randrange(1, 99)}-{rnd.randrange(100, 999)}-{rnd.randrange(1000, 9999)}"

    def _account_id(self):
        return "".join(self._rnd.choices(string.ascii_uppercase + string.digits, k=12))

    @staticmethod
    def _age(birth):
        today = date.today()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

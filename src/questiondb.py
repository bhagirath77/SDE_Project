VARNAME = "varname"
FUNCSIG = "funcsig"
MANUAL = "manual"

GENDER = "gender"
AGE = "age"
EXPERIENCE = "experience"
STUDENT = "student"
DEGREE = "degree"
DEGREE_YEAR = "degree_year"

DEMOGRAPHIC_INFO = {
    EXPERIENCE: "Q183_1",
}
QUESTIONS_INFO = {
    "Gum": {
        "English": {
            VARNAME: ["Q54", "Q55"],
            MANUAL: ["Q49", "Q50", "Q53", None],
        }  # Gum - English
    },
    "Maze": {
        "English (Maze)": {VARNAME: ["Q61", "Q62", "Q63"]},  # Maze - English
        "English (Labyrinth)": {
            VARNAME: ["Q79", "Q80", "Q81a"]
        },  # Maze - English - Labirynth
    },
    "Tic-Tac-Toe": {
        "English (Board)": {
            VARNAME: ["Q81b"],
            FUNCSIG: ["Q84"],
            MANUAL: ["Q82", "Q83", "Q160"],
        },  # Tic-Tac-Toe - English - Board
        "English (Grid)": {
            VARNAME: ["Q93"],
            FUNCSIG: ["Q96a"],
            MANUAL: ["Q94a", "Q95a", "Q159"],
        },  # Tic-Tac-Toe - English - Grid
    },
    "Files": {
        "English": {
            VARNAME: ["Q99"],
            FUNCSIG: ["Q103"],
            MANUAL: ["Q100", "Q101-role", "Q101-type", "Q102"],
        },  # Files - English
    },
    "Elevator": {
        "English": {
            VARNAME: ["Q123", "Q124", "Q125", "Q127"],
            MANUAL: ["Q122", "Q126"],
        },  # Elevator - English
    },
    "Kasata": {
        "English": {
            FUNCSIG: ["Q117"],
            MANUAL: ["Q118", "Q119", "Q157", "Q158"],
        }  # Kasata - English
    },
    "Benefits card": {
        "English": {
            VARNAME: ["Q136", "Q137", "Q138"],
            FUNCSIG: ["Q139"],
        },  # Benefits card - English
    },
}


class VariantInfo(object):
    def __init__(self, name, questions_dict):
        self.name = name
        self._dict = questions_dict

    def get_questions(self, type=None):
        return [
            q for t in self._dict for q in self._dict[t] if t == type or type is None
        ]

    def get_question(self, type, index):
        return self._dict[type][index]

    def has_type(self, type):
        return type in self._dict and len(self._dict[type]) != 0


class SectionInfo(object):
    def __init__(self, name, variants_dict):
        self._variants = {v: VariantInfo(v, variants_dict[v]) for v in variants_dict}
        self.name = name

        variants = self._variants.keys()

    @property
    def variants(self):
        return self._variants.values()

    def get_variant(self, name):
        return self._variants[name]

    def has_type(self, type):
        for var in self._variants.values():
            if var.has_type(type):
                return True
        return False

    def get_questions_across_variants(self, type):
        for i in self._variants.values():
            question_count = len(i.get_questions(type))
            break

        def get_question(v, i):
            return self._variants[v].get_question(type, i)

        return [
            {
                v: get_question(v, i)
                for v in self._variants
                if get_question(v, i) is not None
            }
            for i in range(question_count)
        ]


def get_questions(type=None):
    return [
        q
        for s in QUESTIONS_INFO
        for v in QUESTIONS_INFO[s]
        for t in QUESTIONS_INFO[s][v]
        if t == type or type == None
        for q in QUESTIONS_INFO[s][v][t]
        if q is not None
    ]


def get_sections():
    return [SectionInfo(s, QUESTIONS_INFO[s]) for s in QUESTIONS_INFO]


def get_demographic_question(type):
    return DEMOGRAPHIC_INFO[type]

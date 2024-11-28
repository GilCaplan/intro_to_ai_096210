non_comp_problems = [
    {
        "map": [
            ['P', 'P', 'I', 'I'],
            ['P', 'P', 'P', 'P'],
            ['I', 'P', 'I', 'P'],
            ['P', 'P', 'V', 'I']
        ],
        "wizards": {"Harry Potter": ((2, 1), 1)},
        "death_eaters": {'death_eater1': [(0, 1), (0, 0)]},
        "horcruxes": [(1, 3)],
    },
    {
        "map": [
            ['P', 'P', 'I', 'P', 'P'],
            ['I', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'V']
        ],
        "wizards": {"Harry Potter": ((0, 0), 3)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1), (1, 2)]},
        "horcruxes": [(4, 3)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P'],
            ['V', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 2)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1)]},
        "horcruxes": [(0, 3), (1, 2), (1, 0)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'I'],
            ['P', 'P', 'I', 'P', 'P'],
            ['P', 'I', 'I', 'P', 'P'],
            ['P', 'V', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 2)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1)]},
        "horcruxes": [(0, 3), (1, 3), (3, 0)],
    },

]

comp_problems = [
    {
        "map": [
            ['P', 'P', 'I', 'I'],
            ['P', 'P', 'P', 'P'],
            ['I', 'P', 'I', 'P'],
            ['P', 'P', 'V', 'I']
        ],
        "wizards": {"Harry Potter": ((2, 1), 1), "Hermione Granger": ((0, 0), 2)},
        "death_eaters": {'death_eater1': [(0, 1), (0, 0)]},
        "horcruxes": [(1, 3)],
    },
    {
        "map": [
            ['P', 'P', 'I', 'P', 'P'],
            ['I', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'V']
        ],
        "wizards": {"Harry Potter": ((0, 0), 3), "Ron Weasley": ((0, 1), 2)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1), (1, 2)]},
        "horcruxes": [(4, 3)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'I'],
            ['P', 'P', 'I', 'P', 'P'],
            ['P', 'I', 'I', 'P', 'P'],
            ['P', 'V', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 2), "Ron Weasley": ((0, 1), 2)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1)]},
        "horcruxes": [(0, 3), (1, 3), (3, 0), (2, 4)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'I', 'I'],
            ['P', 'P', 'I', 'P', 'P', 'I'],
            ['P', 'I', 'I', 'I', 'P', 'I'],
            ['P', 'V', 'P', 'P', 'P', 'I'],
            ['P', 'P', 'P', 'P', 'P', 'I'],
            ['P', 'P', 'P', 'P', 'P', 'I'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 2), "Ron Weasley": ((0, 1), 2), "Hermione Granger": ((0, 2), 2)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1)], 'death_eater2': [(3, 2), (4, 2), (4, 3)]},
        "horcruxes": [(0, 3), (1, 3), (3, 0), (2, 4), (4, 4), (5, 0)],
    } 

]

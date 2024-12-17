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
    # {
    #     "map": [
    #         ['P', 'P', 'I', 'I'],
    #         ['P', 'P', 'P', 'P'],
    #         ['I', 'P', 'I', 'P'],
    #         ['P', 'P', 'V', 'I']
    #     ],
    #     "wizards": {"Harry Potter": ((2, 1), 1), "Hermione Granger": ((0, 0), 2)},
    #     "death_eaters": {'death_eater1': [(0, 1), (0, 0)]},
    #     "horcruxes": [(1, 3)],
    # },
    # {
    #     "map": [
    #         ['P', 'P', 'I', 'P', 'P'],
    #         ['I', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'I', 'P', 'P'],
    #         ['P', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'P', 'V']
    #     ],
    #     "wizards": {"Harry Potter": ((0, 0), 3), "Ron Weasley": ((0, 1), 2)},
    #     "death_eaters": {'death_eater1': [(0, 1), (1, 1), (1, 2)]},
    #     "horcruxes": [(4, 3)],
    # },
    # {
    #     "map": [
    #         ['P', 'P', 'P', 'P', 'I'],
    #         ['P', 'P', 'I', 'P', 'P'],
    #         ['P', 'I', 'I', 'P', 'P'],
    #         ['P', 'V', 'P', 'P', 'P'],
    #     ],
    #     "wizards": {"Harry Potter": ((0, 0), 2), "Ron Weasley": ((0, 1), 2)},
    #     "death_eaters": {'death_eater1': [(0, 1), (1, 1)]},
    #     "horcruxes": [(0, 3), (1, 3), (3, 0), (2, 4)],
    # },
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

s_problems = [
    {
        "map": [
            ['P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'I', 'P', 'P', 'P'],
            ['P', 'I', 'I', 'I', 'I', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'I', 'P', 'V'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 1)},
        "death_eaters": {'death_eater1': [(3, 0)],
                         'death_eater2': [(0, 3), (1, 3)],
                         'death_eater3': [(1, 4), (0, 4)],
                         'death_eater4': [(0, 5), (1, 5)]},
        "horcruxes": [(5, 0)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'I', 'P', 'P', 'P'],
            ['I', 'I', 'P', 'I', 'P', 'I', 'I'],
            ['V', 'P', 'P', 'I', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 3), "Hermione Granger": ((0, 6), 1)},
        "death_eaters": {'death_eater1': [(2, 1)],
                         'death_eater2': [(2, 2)]},
        "horcruxes": [(2, 6)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'I', 'P', 'P', 'P'],
            ['P', 'V', 'P', 'I', 'P', 'I', 'I'],
            ['P', 'P', 'P', 'I', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 3), "Hermione Granger": ((0, 6), 1)},
        "death_eaters": {},
        "horcruxes": [(2, 6)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['I', 'I', 'I', 'I', 'I', 'I', 'P'],
            ['V', 'P', 'P', 'P', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 7)},
        "death_eaters": {'death_eater1_1': [(0, 3)],
                         'death_eater1_2': [(0, 3)],
                         'death_eater1_3': [(0, 3)],
                         'death_eater2': [(0, 4)],
                         'death_eater3': [(2, 3)],
                         'death_eater4': [(2, 4)]},
        "horcruxes": [],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'I', 'I', 'I', 'I', 'I', 'P'],
            ['V', 'P', 'P', 'P', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 0), 5)},
        "death_eaters": {'death_eater1': [(0, 3)],
                         'death_eater2': [(0, 4)],
                         'death_eater3': [(2, 3)],
                         'death_eater4': [(2, 4)],
                         'death_eater5_1': [(1, 0)],
                         'death_eater5_2': [(1, 0)],
                         'death_eater5_3': [(1, 0)],
                         'death_eater5_4': [(1, 0)],
                         'death_eater5_5': [(1, 0)]},
        "horcruxes": [(2, 1)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'V'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ],
        "wizards": {"Harry Potter": ((0, 1), 3)},
        "death_eaters": {'death_eater1': [(0, 1), (1, 1)],
                         'death_eater2': [(1, 2), (0, 2)],
                         'death_eater3': [(0, 4), (0, 5), (0, 4), (1, 4)],
                         'death_eater4': [(1, 5), (0, 5), (1, 5), (1, 4)],
                         'death_eater5_1': [(1, 7), (0, 7)],
                         'death_eater5_2': [(1, 8), (0, 8)]},
        "horcruxes": [],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'V', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],

        ],
        "wizards": {"Harry Potter": ((1, 2), 1), "Hermione Granger": ((2, 4), 1)},
        "death_eaters": {},
        "horcruxes": [(0, 0), (3, 0), (0, 6), (3, 6), (3, 3), (3, 3), (3, 3)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'V', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P'],

        ],
        "wizards": {"Harry Potter": ((1, 2), 1), "Hermione Granger": ((2, 4), 3)},
        "death_eaters": {'death_eater1': [(3, 3), (2, 3)],
                         'death_eater2': [(0, 3), (1, 3)]},
        "horcruxes": [(0, 0), (3, 0), (0, 6), (3, 6), (3, 3), (3, 3), (3, 3)],
    },
    {
        "map": [
            ['P', 'P', 'P', 'P'],
            ['P', 'I', 'P', 'P'],
            ['P', 'I', 'P', 'P'],
            ['P', 'I', 'P', 'V'],
            ['P', 'I', 'P', 'P'],
            ['P', 'I', 'P', 'P'],
            ['P', 'I', 'P', 'P'],
            ['P', 'P', 'P', 'P'],

        ],
        "wizards": {"Harry Potter": ((7, 0), 2)},
        "death_eaters": {'death_eater1': [(7, 1)]},
        "horcruxes": [(3, 2)],
    }
]

check_problems = [
    {
        "map": [
            ['P', 'P', 'P', 'I', 'I'],
            ['P', 'P', 'P', 'P', 'P'],
            ['I', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'V']
        ],
        "wizards": {"Harry Potter": ((0, 0), 3), "Ron Weasley": ((3, 0), 2)},
        "death_eaters": {'death_eater1': [(1, 3), (2, 3)]},
        "horcruxes": [(1, 4), (3, 4)],
    },

{
    "map": [
        ['P', 'P', 'I', 'P', 'P', 'P'],
        ['P', 'I', 'I', 'P', 'I', 'P'],
        ['P', 'P', 'P', 'P', 'P', 'P'],
        ['P', 'P', 'I', 'P', 'V', 'I'],
        ['I', 'P', 'I', 'P', 'P', 'P'],
        ['P', 'P', 'P', 'P', 'P', 'P'],
    ],
    "wizards": {"Harry Potter": ((0, 0), 2), "Hermione Granger": ((5, 5), 3)},
    "death_eaters": {
        'death_eater1': [(2, 1), (2, 2)],
        'death_eater2': [(4, 3), (4, 4)],
    },
    "horcruxes": [(0, 5), (5, 0), (3, 3)],
},

{
    "map": [
        ['P', 'P', 'P', 'P', 'P'],
        ['P', 'P', 'I', 'P', 'P'],
        ['P', 'P', 'I', 'P', 'P'],
        ['P', 'P', 'P', 'P', 'P'],
        ['P', 'P', 'P', 'P', 'V']
    ],
    "wizards": {"Harry Potter": ((0, 0), 3)},
    "death_eaters": {'death_eater1': [(3, 2), (4, 2)], 'death_eater2': [(1, 2), (2, 2)]},
    "horcruxes": [(3, 3), (0, 4)],
}
]
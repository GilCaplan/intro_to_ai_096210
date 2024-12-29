
EXAMPLE_INIT_INPUT = (  # This one corresponds to the first input example
    (3, 3),
    (0, 0),
    []
)

EXAMPLE_OBSERVATIONS = [  # This is the observation for the 4th input example when standing at (2, 1)
    ('vault', (2, 2)),
    ('dragon', (1, 1)),
    ('sulfur',)
]

CODES_NEW = {'passage': 0, 'dragon': 1, 'vault': 2, 'trap': 3, 'hollow_vault': 4, 'vault_trap': 5, 'dragon_trap': 6,
             'hollow_trap_vault': 7}

inputs = [
    {
        'Harry_start': (0, 0),
        'full_map': [
            [0, 0, 0],
            [0, 0, 0],
            [0, 4, 0]
        ]
    },
    {
        'Harry_start': (0, 0),
        'full_map': [
            [0, 0, 0],
            [0, 1, 0],
            [0, 4, 0]
        ]
    },
    {
        'Harry_start': (0, 0),
        'full_map': [
            [0, 0, 0],
            [0, 3, 0],
            [0, 4, 0]
        ]
    },
    {
        'Harry_start': (0, 0),
        'full_map': [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 7]
        ]
    },
    {
        'Harry_start': (0, 0),
        'full_map': [
            [0, 0, 0, 0],
            [0, 0, 1, 3],
            [0, 6, 2, 4],
            [0, 0, 5, 0]
        ]
    },
    {
        'Harry_start': (3, 3),
        'full_map': [
            [0, 0, 0, 0],
            [0, 0, 1, 3],
            [0, 6, 2, 4],
            [0, 0, 5, 0]
        ]
    },
]

inputs2 = [
{
    'Level': 1,
    'Board': 1,
    'Harry_start': (0, 0),
    'full_map': [
        [0, 0, 0, 0, 0],
        [0, 3, 4, 0, 0],
        [0, 3, 0, 0, 5],
        [5, 0, 0, 0, 0],
    ]
},

{
    'Level': 1,
    'Board': 2,
    'Harry_start': (0, 0),
    'full_map': [
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 3],
        [0, 2, 7, 0, 0],
        [0, 0, 5, 0, 0],
    ]
},

{
    'Level': 1,
    'Board': 3,
    'Harry_start': (0, 0),
    'full_map': [
        [0, 0, 0, 3],
        [7, 0, 2, 3],
        [0, 2, 0, 0],
    ]
},

{
    'Level': 1,
    'Board': 4,
    'Harry_start': (0, 0),
    'full_map': [
        [0, 0, 2, 0, 0],
        [0, 3, 3, 0, 2],
        [0, 0, 0, 4, 0],
    ]
},

{
    'Level': 1,
    'Board': 5,
    'Harry_start': (0, 0),
    'full_map': [
        [0, 0, 0, 5],
        [0, 0, 2, 0],
        [4, 3, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 0],
    ]
},

]

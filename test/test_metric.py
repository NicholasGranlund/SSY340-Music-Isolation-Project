import torch
from metric import dice_coeff

fake_prediction_1 = torch.tensor(
    [
        [
            [
                [1, 0],
                [0, 1]
            ]
        ],
        [
            [
                [1, 1],
                [0, 0],
            ]
        ]
    ]

)

fake_label_1 = torch.tensor(
    [
        [
            [
                [1, 0],
                [0, 1]
            ]
        ],
        [
            [
                [1, 1],
                [0, 0],
            ]
        ]
    ]
)


fake_prediction_2 = torch.tensor(
    [
        [
            [
                [1, 1],
                [1, 1]
            ]
        ],
        [
            [
                [1, 1],
                [1, 1],
            ]
        ]
    ]

)

fake_label_2 = torch.tensor(
    [
        [
            [
                [0, 0],
                [0, 0]
            ]
        ],
        [
            [
                [0, 0],
                [0, 0],
            ]
        ]
    ]
)

fake_prediction_3 = torch.tensor(
    [
        [
            [
                [0, 0],
                [0, 0]
            ]
        ],
        [
            [
                [1, 1],
                [1, 1],
            ]
        ]
    ]

)

fake_label_3 = torch.tensor(
    [
        [
            [
                [0, 0],
                [0, 0]
            ]
        ],
        [
            [
                [0, 0],
                [0, 0],
            ]
        ]
    ]
)


if __name__ == '__main__':
    assert dice_coeff(fake_prediction_1, fake_label_1) == 1.
    assert dice_coeff(fake_prediction_2, fake_label_2) == 0.
    assert dice_coeff(fake_prediction_3, fake_label_3) == 0.5


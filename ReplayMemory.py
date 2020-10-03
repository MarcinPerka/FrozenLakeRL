from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('current_state', 'action', 'next_state', 'reward', 'done', 'next_action'))



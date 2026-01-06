
class StateMachine:
    def __init__(self, initial: str = 'idle') -> None:
        self.state = initial

    def transition(self, new_state: str) -> None:
        self.state = new_state

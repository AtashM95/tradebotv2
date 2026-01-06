
from typing import List
from .contracts import ExecutionRequest

class OrderManager:
    def __init__(self) -> None:
        self.orders: List[ExecutionRequest] = []

    def submit(self, request: ExecutionRequest) -> None:
        self.orders.append(request)


class SmsSender:
    def send(self, message: str) -> dict:
        return {'status': 'ok', 'message': message}

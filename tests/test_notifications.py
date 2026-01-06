
from src.notifications.notification_manager import NotificationManager

def test_notifications():
    assert NotificationManager().send('x')['status'] == 'ok'

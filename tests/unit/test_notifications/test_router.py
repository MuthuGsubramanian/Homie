from homie_core.notifications.router import NotificationRouter, Notification
from homie_core.config import NotificationConfig


class TestNotificationRouter:
    def test_routes_enabled_category(self):
        cfg = NotificationConfig()
        router = NotificationRouter(config=cfg)
        n = Notification(category="email_digest", title="New Email", body="3 unread")
        assert router.should_deliver(n) is True

    def test_blocks_disabled_category(self):
        cfg = NotificationConfig(categories={
            "email_digest": False, "task_reminders": True,
            "social_mentions": True, "context_suggestions": True, "system_alerts": True,
        })
        router = NotificationRouter(config=cfg)
        n = Notification(category="email_digest", title="New Email", body="3 unread")
        assert router.should_deliver(n) is False

    def test_dnd_blocks_all(self):
        cfg = NotificationConfig()
        router = NotificationRouter(config=cfg)
        router.set_dnd(True)
        n = Notification(category="email_digest", title="New Email", body="3 unread")
        assert router.should_deliver(n) is False

    def test_queues_during_dnd(self):
        cfg = NotificationConfig()
        router = NotificationRouter(config=cfg)
        router.set_dnd(True)
        n = Notification(category="email_digest", title="New Email", body="3 unread")
        router.route(n)
        assert len(router.get_pending()) == 1

    def test_dnd_schedule_wraps_midnight(self):
        cfg = NotificationConfig(dnd_schedule_enabled=True, dnd_schedule_start="22:00", dnd_schedule_end="07:00")
        router = NotificationRouter(config=cfg)
        assert router._is_in_dnd_schedule("23:00") is True
        assert router._is_in_dnd_schedule("12:00") is False
        assert router._is_in_dnd_schedule("06:59") is True
        assert router._is_in_dnd_schedule("07:01") is False

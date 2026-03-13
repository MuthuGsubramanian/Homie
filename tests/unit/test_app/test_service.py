from unittest.mock import patch, MagicMock
from homie_app.service.scheduler_task import ServiceManager


class TestServiceManager:
    def test_init(self):
        mgr = ServiceManager()
        assert mgr is not None

    @patch("homie_app.service.scheduler_task.subprocess.run")
    def test_register_creates_task(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = ServiceManager()
        result = mgr.register()
        assert result is True
        assert mock_run.called

    @patch("homie_app.service.scheduler_task.subprocess.run")
    def test_unregister_removes_task(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = ServiceManager()
        result = mgr.unregister()
        assert result is True

    @patch("homie_app.service.scheduler_task.subprocess.run")
    def test_status_returns_state(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Running")
        mgr = ServiceManager()
        status = mgr.status()
        assert isinstance(status, str)

    @patch("homie_app.service.scheduler_task.subprocess.run")
    def test_register_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Access denied")
        mgr = ServiceManager()
        result = mgr.register()
        assert result is False

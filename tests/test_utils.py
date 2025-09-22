"""Tests for the utils module."""

import os
import tempfile
import pytest
import shutil
from pathlib import Path
import requests
import subprocess
import time
from unittest import mock

from apparent.utils import download_and_launch_local_db, download_file, update_env_file, stop_datasette, list_datasette_processes


class MockResponse:
    def __init__(self, status_code=200, content=b"mock content", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-length": str(len(content))}
        self._content = content
        
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP Error: {self.status_code}")
    
    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i+chunk_size]

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.unit
class TestUtils:
    """Test suite for the utils module."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        tmp_dir = tempfile.mkdtemp()
        yield tmp_dir
        shutil.rmtree(tmp_dir)
    
    @pytest.fixture
    def mock_env_file(self, temp_dir):
        """Create a temporary .env file for testing."""
        env_path = Path(temp_dir) / ".env"
        with open(env_path, "w") as f:
            f.write("APPARENT_URL=https://example.com/data.csv\n")
        
        # Mock os.getcwd to return the temp dir
        with mock.patch("os.getcwd", return_value=temp_dir):
            yield env_path
    
    @mock.patch("apparent.utils.requests.get")
    def test_download_file(self, mock_get, temp_dir):
        """Test downloading a file."""
        mock_content = b"test database content"
        mock_get.return_value = MockResponse(content=mock_content)
        
        # Create a destination path
        dest_path = Path(temp_dir) / "test.db"
        
        # Call the function
        download_file("https://example.com/test.db", dest_path)
        
        # Check that the file was created with the correct content
        assert dest_path.exists()
        with open(dest_path, "rb") as f:
            content = f.read()
            assert content == mock_content
    
    def test_update_env_file(self, temp_dir):
        """Test updating the .env file."""
        import os
        
        # Create a real .env file in temp directory
        env_path = Path(temp_dir) / ".env"
        with open(env_path, "w") as f:
            f.write("APPARENT_URL=https://example.com/data.csv\n")
        
        # Change to temp directory and call function
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            update_env_file("http://localhost:8001/data.csv")
        finally:
            os.chdir(original_cwd)
        
        # Read the file and check contents
        with open(env_path, "r") as f:
            content = f.read()
            
        # Should contain both URLs
        assert "LOCAL_URL=http://localhost:8001/data.csv" in content
        assert "APPARENT_URL=https://example.com/data.csv" in content
    
    @mock.patch("apparent.utils.requests.get")
    @mock.patch("apparent.utils.subprocess.Popen")
    @mock.patch("apparent.utils.update_env_file")
    def test_download_and_launch_local_db_existing_file(self, mock_update_env, mock_popen, mock_get, temp_dir):
        """Test downloading and launching when the DB already exists."""
        # Create a mock DB file
        db_path = Path(temp_dir) / "test.db"
        with open(db_path, "wb") as f:
            f.write(b"test database content")
        
        # Setup mocks
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = ("output", "")
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        mock_get.return_value = MockResponse()
        
        # Call function with existing DB
        result = download_and_launch_local_db(
            db_path=db_path,
            port=9999,
            update_env=True,
            timeout=1
        )
        
        # Verify Datasette was launched correctly
        assert result['url'] == "http://127.0.0.1:9999"
        assert result['port'] == 9999
        assert result['csv_url'] == "http://127.0.0.1:9999/us_physician_referral_networks.csv"
        assert 'pid' in result
        mock_popen.assert_called_once()
        
        # Check that download wasn't attempted
        assert not any("download" in str(call).lower() for call in mock_get.call_args_list)
        
        # Check env file was updated
        mock_update_env.assert_called_once()
    
    @mock.patch("apparent.utils.download_file")
    @mock.patch("apparent.utils.requests.get")
    @mock.patch("apparent.utils.subprocess.Popen")
    @mock.patch("apparent.utils.update_env_file")
    def test_download_and_launch_local_db_new_file(self, mock_update_env, mock_popen, mock_get, mock_download, temp_dir):
        """Test downloading and launching when the DB doesn't exist."""
        # Setup path to a non-existent file
        db_path = Path(temp_dir) / "new.db"
        
        # Setup mocks
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = ("output", "")
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        mock_get.return_value = MockResponse()
        
        # Call function with non-existent DB
        result = download_and_launch_local_db(
            db_url="https://example.com/test.db",
            db_path=db_path,
            port=9999,
            update_env=True,
            timeout=1
        )
        
        # Verify Datasette was launched correctly
        assert result['url'] == "http://127.0.0.1:9999"
        assert result['port'] == 9999
        assert result['csv_url'] == "http://127.0.0.1:9999/us_physician_referral_networks.csv"
        assert 'pid' in result
        
        # Check that download was attempted
        mock_download.assert_called_once_with("https://example.com/test.db", db_path)
        
        # Check env file was updated
        mock_update_env.assert_called_once()
        
    @mock.patch("apparent.utils.requests.get")
    @mock.patch("apparent.utils.subprocess.Popen")
    def test_download_and_launch_local_db_timeout(self, mock_popen, mock_get, temp_dir):
        """Test timeout handling when Datasette fails to start."""
        # Create a mock DB file
        db_path = Path(temp_dir) / "test.db"
        with open(db_path, "wb") as f:
            f.write(b"test database content")
        
        # Setup mocks
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = ("output", "error message")
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        # Mock requests to simulate Datasette not starting
        mock_get.side_effect = requests.RequestException("Connection refused")
        
        # Test that a timeout exception is raised
        with pytest.raises(TimeoutError):
            download_and_launch_local_db(
                db_path=db_path,
                port=9999,
                update_env=False,
                timeout=1
            )
    
    @mock.patch("importlib.util.find_spec")
    def test_download_and_launch_local_db_missing_datasette(self, mock_find_spec):
        """Test handling when Datasette is not installed."""
        # Setup mock to simulate datasette not being installed
        mock_find_spec.return_value = None
        
        # Test that an ImportError is raised with a helpful message
        with pytest.raises(ImportError) as e:
            download_and_launch_local_db()
        
        assert "Datasette is required" in str(e.value)
    
    @mock.patch("apparent.utils.requests.get")
    @mock.patch("apparent.utils.subprocess.Popen")
    @mock.patch("apparent.utils.update_env_file")
    @mock.patch("select.select")  # Mock select module directly
    def test_download_and_launch_local_db_verbose_mode(self, mock_select, mock_update_env, mock_popen, mock_get, temp_dir):
        """Test that verbose mode provides additional logging."""
        # Create a mock DB file
        db_path = Path(temp_dir) / "test.db"
        with open(db_path, "wb") as f:
            f.write(b"test database content")
        
        # Setup mocks
        mock_process = mock.MagicMock()
        mock_process.poll.return_value = None
        mock_process.communicate.return_value = ("output", "")
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        # Mock select to return empty (no data available)
        mock_select.return_value = ([], [], [])
        
        mock_get.return_value = MockResponse()
        
        # Call function with verbose=True
        with mock.patch("apparent.utils.logger") as mock_logger:
            result = download_and_launch_local_db(
                db_path=db_path,
                port=9999,
                update_env=False,
                timeout=1,
                verbose=True
            )
            
            # Check that verbose logging was used
            mock_logger.info.assert_any_call("Command: datasette %s --port=9999 --setting=sql_time_limit_ms 500000 --setting=max_returned_rows 200000 --setting=allow_csv_stream off" % str(db_path))
            mock_logger.info.assert_any_call("Process started with PID: 12345")
    
    
    def test_stop_datasette_no_port_or_pid(self):
        """Test stop_datasette when neither port nor pid is provided."""
        with pytest.raises(ValueError) as e:
            stop_datasette()
        
        assert "Either port or pid must be provided" in str(e.value)
    
    @mock.patch("apparent.utils.psutil")
    def test_stop_datasette_by_port(self, mock_psutil):
        """Test stopping Datasette by port number."""
        # Setup mock process
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.name.return_value = "datasette"
        
        # Create mock connection object
        mock_conn = mock.MagicMock()
        mock_conn.laddr.port = 8001
        mock_conn.status = "LISTEN"
        mock_proc.connections.return_value = [mock_conn]
        
        mock_psutil.process_iter.return_value = [mock_proc]
        mock_psutil.CONN_LISTEN = "LISTEN"
        
        # Call function
        result = stop_datasette(port=8001, verbose=True)
        
        # Verify process was terminated
        assert result is True
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
    
    @mock.patch("apparent.utils.psutil")
    def test_stop_datasette_by_pid(self, mock_psutil):
        """Test stopping Datasette by process ID."""
        # Setup mock process
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.name.return_value = "datasette"
        
        mock_psutil.Process.return_value = mock_proc
        
        # Call function
        result = stop_datasette(pid=12345)
        
        # Verify process was found and terminated
        assert result is True
        mock_psutil.Process.assert_called_once_with(12345)
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
    
    @mock.patch("apparent.utils.psutil")
    def test_stop_datasette_process_not_found(self, mock_psutil):
        """Test stopping Datasette when no matching process is found."""
        # Setup empty process list
        mock_psutil.process_iter.return_value = []
        
        # Call function
        result = stop_datasette(port=8001)
        
        # Verify no process was stopped
        assert result is False
    
    @mock.patch("apparent.utils.psutil")
    def test_stop_datasette_force_kill(self, mock_psutil):
        """Test force killing when graceful termination fails."""
        # Setup mock process that doesn't terminate gracefully
        mock_proc = mock.MagicMock()
        mock_proc.pid = 12345
        mock_proc.name.return_value = "datasette"
        
        # Create mock connection object
        mock_conn = mock.MagicMock()
        mock_conn.laddr.port = 8001
        mock_conn.status = "LISTEN"
        mock_proc.connections.return_value = [mock_conn]
        
        # Setup TimeoutExpired exception
        timeout_exception = Exception("timeout")  # Simple exception for testing
        mock_proc.wait.side_effect = [timeout_exception, None]
        
        mock_psutil.process_iter.return_value = [mock_proc]
        mock_psutil.CONN_LISTEN = "LISTEN"
        mock_psutil.TimeoutExpired = Exception
        
        # Call function
        result = stop_datasette(port=8001, verbose=True)
        
        # Verify process was force killed
        assert result is True
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 2
    

    
    @mock.patch("apparent.utils.psutil")
    def test_list_datasette_processes(self, mock_psutil):
        """Test listing Datasette processes."""
        # Setup mock processes
        mock_datasette_proc = mock.MagicMock()
        mock_datasette_proc.pid = 12345
        mock_datasette_proc.name.return_value = "python"
        mock_datasette_proc.cmdline.return_value = ["datasette", "test.db"]  # Direct datasette command
        
        # Create mock connection object
        mock_conn = mock.MagicMock()
        mock_conn.laddr.port = 8001
        mock_conn.status = "LISTEN"
        mock_datasette_proc.connections.return_value = [mock_conn]
        
        mock_other_proc = mock.MagicMock()
        mock_other_proc.cmdline.return_value = ["python", "other_script.py"]
        
        mock_psutil.process_iter.return_value = [mock_datasette_proc, mock_other_proc]
        mock_psutil.CONN_LISTEN = "LISTEN"
        
        # Call function
        processes = list_datasette_processes(verbose=True)
        
        # Verify only Datasette process is returned
        assert len(processes) == 1
        proc_info = processes[0]
        assert proc_info['pid'] == 12345
        assert proc_info['name'] == "python"
        assert proc_info['ports'] == [8001]
        assert "datasette" in " ".join(proc_info['cmdline']).lower()
    
    @mock.patch("apparent.utils.psutil")
    def test_list_datasette_processes_empty(self, mock_psutil):
        """Test listing Datasette processes when none are running."""
        # Setup empty process list
        mock_psutil.process_iter.return_value = []
        
        # Call function
        processes = list_datasette_processes()
        
        # Verify empty list is returned
        assert processes == []
    
    @mock.patch("apparent.utils.psutil")
    def test_list_datasette_processes_access_denied(self, mock_psutil):
        """Test listing processes when access is denied to some processes."""
        # Setup mock process that raises AccessDenied
        mock_proc = mock.MagicMock()
        mock_proc.cmdline.return_value = ["datasette", "test.db"]
        mock_proc.connections.side_effect = Exception("Access denied")  # Simulate AccessDenied
        mock_proc.pid = 12345
        mock_proc.name.return_value = "datasette"
        
        mock_psutil.process_iter.return_value = [mock_proc]
        mock_psutil.AccessDenied = Exception
        mock_psutil.NoSuchProcess = Exception
        
        # Call function
        processes = list_datasette_processes()
        
        # Verify process is still listed but with unknown ports
        assert len(processes) == 1
        assert processes[0]['ports'] == ['unknown']

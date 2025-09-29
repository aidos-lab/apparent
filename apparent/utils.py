"""
Utilities for the Apparent package.

This module provides utility functions for downloading and setting up local databases for
the Apparent package, enabling local access to the physician referral network data.
"""

import time
import subprocess
from pathlib import Path
import logging
from typing import Optional, Union, Tuple, List, Dict, Any
import requests
import psutil


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_and_launch_local_datasette(
    db_url: str = "https://apparent.topology.rocks/us_physician_referral_networks.db",
    db_path: Union[str, Path] = "data/us_physician_referral_networks.db",
    port: int = 8001,
    update_env: bool = True,
    timeout: int = 60,
    settings: Optional[dict] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Downloads the SQL database from a remote URL and launches a local Datasette instance. If the database file already exists locally, this function will just launch Datasette.
    

    Parameters
    ----------
    db_url : str, optional
        The URL to download the database from.
        Default is "https://apparent.topology.rocks/us_physician_referral_networks.db"
    db_path : str, optional
        The local path to save the database file to.
        Default is "data/us_physician_referral_networks.db"
    port : int, optional
        The port to run the Datasette server on.
        Default is 8001.
    update_env : bool, optional
        Whether to update the .env file with the local URL.
        Default is True.
    timeout : int, optional
        Maximum time to wait (in seconds) for Datasette to start.
        Default is 60.
    settings : dict, optional
        Additional settings to pass to Datasette.
        Default settings include:
        - sql_time_limit_ms: 500000
        - max_returned_rows: 200000
        - allow_csv_stream: off
    verbose : bool, optional
        If True, enables detailed logging of subprocess output and connection attempts.
        Default is False.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing information about the running Datasette instance:
        - 'url': The URL of the locally running Datasette instance
        - 'pid': The process ID of the Datasette process
        - 'port': The port number Datasette is running on
        - 'csv_url': The URL for CSV export endpoint

    Raises
    ------
    ImportError
        If Datasette is not installed.
    FileNotFoundError
        If the database file cannot be downloaded.
    TimeoutError
        If Datasette fails to start within the timeout period.

    Notes
    -----
    This function requires the Datasette package to be installed.
    You can install it with: pip install datasette

    Examples
    --------
    >>> from apparent.utils import download_and_launch_local_db
    >>> result = download_and_launch_local_db(
    ...     db_path="path/to/save/database.db",
    ...     port=8080
    ... )
    >>> print(f"Datasette running at: {result['url']}")
    >>> print(f"Process ID: {result['pid']}")
    """
    # Check if datasette is installed
    try:
        import importlib.util
        if importlib.util.find_spec("datasette") is None:
            raise ImportError("Datasette module not found")
    except ImportError:
        msg = (
            "Datasette is required for this function but it's not installed. "
            "You can install it with: pip install datasette"
        )
        logger.error(msg)
        raise ImportError(msg)

    # Ensure the directory exists
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the database if it doesn't exist
    if not db_path.exists():
        logger.info(f"Downloading database...")
        try:
            download_file(db_url, db_path)
            logger.info(f"Database ready at {db_path}")
        except Exception as e:
            logger.error(f"Failed to download database: {e}")
            raise FileNotFoundError(f"Failed to download database: {e}")
    elif verbose:
        logger.info(f"Database file already exists at {db_path}")

    # Default settings for Datasette
    default_settings = {
        "sql_time_limit_ms": 500000,
        "max_returned_rows": 200000,
        "allow_csv_stream": "off",
    }

    # Merge with user-provided settings, if any
    datasette_settings = default_settings.copy()
    if settings:
        datasette_settings.update(settings)

    # Build command for Datasette
    cmd = ["datasette", str(db_path), f"--port={port}"]
    
    # Add settings to command
    for key, value in datasette_settings.items():
        cmd.append(f"--setting={key}")
        cmd.append(str(value))
    
    logger.info(f"Starting Datasette on port {port}...")
    if verbose:
        logger.info(f"Command: {' '.join(cmd)}")
    
    # Start Datasette process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if verbose:
        logger.info(f"Process started with PID: {process.pid}")
    
    # Wait for Datasette to start
    local_url = f"http://127.0.0.1:{port}"
    csv_url = f"{local_url}/us_physician_referral_networks.csv"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if process is still running and capture any output
        if process.poll() is not None:
            # Process has terminated, capture all remaining output
            stdout, stderr = process.communicate()
            if verbose:
                if stdout:
                    for line in stdout.strip().split('\n'):
                        if line.strip():
                            logger.info(f"[stdout] {line}")
                if stderr:
                    for line in stderr.strip().split('\n'):
                        if line.strip():
                            logger.error(f"[stderr] {line}")
            
            error_message = f"Datasette process exited unexpectedly (return code {process.returncode})"
            logger.error(error_message)
            raise RuntimeError(error_message)
        
        # Try to read any available output without blocking (only in verbose mode)
        if verbose:
            try:
                import select
                
                # Check if there's data available to read (Unix-like systems)
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0)
                    
                    for stream in ready:
                        line = stream.readline()
                        if line:
                            if stream == process.stdout:
                                logger.info(f"[stdout] {line.strip()}")
                            else:
                                logger.error(f"[stderr] {line.strip()}")
            except (ImportError, OSError):
                # select not available or not working, skip non-blocking read
                pass
        
        # Check if Datasette is responding
        try:
            if verbose:
                logger.debug(f"Testing connection to {local_url}")
            response = requests.get(local_url, timeout=3)
            
            if response.status_code == 200:
                logger.info("Datasette is ready!")
                break
            elif verbose:
                logger.warning(f"Unexpected status: {response.status_code}")
                
        except requests.RequestException as e:
            if verbose:
                logger.debug(f"Connection failed: {e}")
        
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 or verbose:  # Log every 10 seconds normally, or every 2 seconds in verbose mode
            logger.info(f"Waiting for Datasette... ({elapsed}s)")
        time.sleep(2)
    
    # Check if we timed out
    if time.time() - start_time >= timeout:
        logger.error(f"Datasette failed to start after {timeout} seconds")
        
        # Capture any remaining output before terminating
        if process.poll() is None:
            # Process is still running, terminate and capture output
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
                if stdout:
                    for line in stdout.strip().split('\n'):
                        if line.strip():
                            logger.info(f"[Datasette stdout] {line}")
                if stderr:
                    for line in stderr.strip().split('\n'):
                        if line.strip():
                            logger.error(f"[Datasette stderr] {line}")
            except subprocess.TimeoutExpired:
                logger.error("Process did not terminate gracefully, killing it")
                process.kill()
                stdout, stderr = process.communicate()
        
        raise TimeoutError(f"Datasette failed to start after {timeout} seconds")
    
    # Final check if Datasette started successfully
    try:
        response = requests.get(local_url, timeout=1)
        if response.status_code != 200:
            raise TimeoutError(f"Datasette responded with status code {response.status_code}")
    except requests.RequestException as e:
        # Kill the process if it's still running
        if process.poll() is None:
            process.terminate()
            
        error_message = f"Datasette failed to respond properly: {e}"
        logger.error(error_message)
        raise TimeoutError(error_message)
    
    # Update .env file if requested
    if update_env:
        update_env_file(csv_url)
    
    # Return process information
    result = {
        'url': local_url,
        'pid': process.pid,
        'port': port,
        'csv_url': csv_url
    }
    
    logger.info(f"Datasette running at {local_url}")
    return result


def download_file(url: str, destination: Union[str, Path], chunk_size: int = 8192) -> None:
    """
    Download a file from a URL to a local destination with progress reporting.
    
    Parameters
    ----------
    url : str
        The URL to download from.
    destination : str or Path
        The local path where the file will be saved.
    chunk_size : int, optional
        Size of chunks to download at a time, in bytes.
        Default is 8192.
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        If the download fails for any reason.
    """
    destination = Path(destination)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            # Format total size in MB
            total_size_mb = total_size / (1024 * 1024)
            logger.info(f"Downloading {total_size_mb:.1f} MB file...")
            
            downloaded = 0
            with open(destination, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress every ~5% or at least every MB
                        progress_pct = (downloaded / total_size) * 100 if total_size else 0
                        if total_size and downloaded % (max(1, int(total_size * 0.05))) < chunk_size:
                            downloaded_mb = downloaded / (1024 * 1024)
                            logger.info(f"Downloaded {downloaded_mb:.1f} MB of {total_size_mb:.1f} MB ({progress_pct:.1f}%)")
                            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        # Remove partial file if it exists
        if destination.exists():
            destination.unlink()
        raise


def update_env_file(local_url: str) -> None:
    """
    Update the .env file with the LOCAL_URL environment variable.
    
    Parameters
    ----------
    local_url : str
        The URL of the locally running Datasette instance.
    
    Returns
    -------
    None
    """
    env_path = Path(".env")
    
    # Read existing environment variables if file exists
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env_vars[key] = value
    
    # Update or add LOCAL_URL
    env_vars["LOCAL_URL"] = local_url
    
    # Make sure APPARENT_URL is set
    if "APPARENT_URL" not in env_vars:
        env_vars["APPARENT_URL"] = "https://apparent.topology.rocks/us_physician_referral_networks.csv"
    
    # Write updated environment variables back to file
    with open(env_path, "w") as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"Updated .env file with LOCAL_URL={local_url}")


def stop_local_datasette(port: Optional[int] = None, pid: Optional[int] = None, verbose: bool = False) -> bool:
    """
    Stop a running Datasette instance by port or process ID.
    
    Parameters
    ----------
    port : int, optional
        The port number where Datasette is running. If provided, will find and stop
        the process listening on this port.
    pid : int, optional
        The process ID of the Datasette instance to stop.
    verbose : bool, optional
        If True, provides detailed logging of the stop process.
        Default is False.
    
    Returns
    -------
    bool
        True if a process was found and stopped, False otherwise.
    
    Raises
    ------
    ValueError
        If neither port nor pid is provided.
    ImportError
        If psutil is not installed.
    
    Notes
    -----
    This function requires the psutil package to be installed.
    You can install it with: pip install psutil
    
    Examples
    --------
    >>> from apparent.utils import stop_datasette
    >>> # Stop by port
    >>> stopped = stop_datasette(port=8001)
    >>> # Stop by process ID  
    >>> stopped = stop_datasette(pid=12345)
    """
    if port is None and pid is None:
        raise ValueError("Either port or pid must be provided")

    
    processes_to_stop = []
    
    if port is not None:
        # Find processes listening on the specified port
        if verbose:
            logger.info(f"Looking for processes listening on port {port}")
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                        processes_to_stop.append(proc)
                        if verbose:
                            logger.info(f"Found process {proc.pid} ({proc.name()}) listening on port {port}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    
    if pid is not None:
        # Find the specific process by PID
        try:
            proc = psutil.Process(pid)
            processes_to_stop.append(proc)
            if verbose:
                logger.info(f"Found process {pid} ({proc.name()})")
        except psutil.NoSuchProcess:
            if verbose:
                logger.warning(f"No process found with PID {pid}")
    
    if not processes_to_stop:
        if verbose:
            logger.info("No matching processes found")
        return False
    
    # Stop the processes
    stopped_any = False
    for proc in processes_to_stop:
        try:
            if verbose:
                logger.info(f"Stopping process {proc.pid} ({proc.name()})")
            
            # First try graceful termination
            proc.terminate()
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                proc.wait(timeout=5)
                logger.info(f"Successfully stopped Datasette process {proc.pid}")
                stopped_any = True
            except psutil.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                if verbose:
                    logger.warning(f"Process {proc.pid} did not terminate gracefully, forcing kill")
                proc.kill()
                proc.wait()
                logger.info(f"Force stopped Datasette process {proc.pid}")
                stopped_any = True
                
        except psutil.NoSuchProcess:
            if verbose:
                logger.info(f"Process {proc.pid} already stopped")
            stopped_any = True
        except psutil.AccessDenied:
            logger.error(f"Access denied when trying to stop process {proc.pid}")
        except Exception as e:
            logger.error(f"Error stopping process {proc.pid}: {e}")
    
    return stopped_any


def list_datasette_processes(verbose: bool = False) -> List[dict]:
    """
    List all running Datasette processes.
    
    Parameters
    ----------
    verbose : bool, optional
        If True, provides detailed information about each process.
        Default is False.
    
    Returns
    -------
    List[dict]
        A list of dictionaries containing process information.
        Each dictionary has keys: 'pid', 'name', 'cmdline', 'ports'
    
    Raises
    ------
    ImportError
        If psutil is not installed.
    
    Examples
    --------
    >>> from apparent.utils import list_datasette_processes
    >>> processes = list_datasette_processes()
    >>> for proc in processes:
    ...     print(f"PID: {proc['pid']}, Ports: {proc['ports']}")
    """
    datasette_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if this is a Datasette process
            cmdline = proc.cmdline()
            if cmdline and len(cmdline) > 0 and 'datasette' in cmdline[0].lower():
                # Get ports this process is listening on
                ports = []
                try:
                    connections = proc.connections()
                    for conn in connections:
                        if conn.status == psutil.CONN_LISTEN:
                            ports.append(conn.laddr.port)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    ports = ['unknown']
                
                process_info = {
                    'pid': proc.pid,
                    'name': proc.name(),
                    'cmdline': cmdline,
                    'ports': sorted(set(ports)) if ports != ['unknown'] else ports
                }
                
                datasette_processes.append(process_info)
                
                if verbose:
                    logger.info(f"Found Datasette process: PID={proc.pid}, Ports={ports}")
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return datasette_processes

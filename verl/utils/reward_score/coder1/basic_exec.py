import os
import resource
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory

from .utils import _DEFAULT_TIMEOUT_SECONDS, _ERROR_MSG_PREFIX

CLI_ARG_SIZE_LIMIT = 1024 * 3


def code_exec_sandbox(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    """
    Execute code with lightweight isolation using resource limits and temp directories.
    Drop-in replacement for code_exec_firejail with ~0% overhead.
    
    Protections:
    - Timeout enforcement
    - Memory limits (4GB)
    - File size limits (2MB)
    - Process limits (32)
    - File descriptor limits (32)
    - Isolated temp directory (HOME set to tmpdir)
    """
    
    def set_limits():
        """Set resource limits before execution"""
        try:
            # Memory limit: 4GB (same as firejail)
            resource.setrlimit(resource.RLIMIT_AS, (4096*1024*1024, 4096*1024*1024))
        except (ValueError, OSError):
            pass  # Some systems don't support this
        
        try:
            # File size limit: 2MB (same as firejail)
            resource.setrlimit(resource.RLIMIT_FSIZE, (2*1024*1024, 2*1024*1024))
        except (ValueError, OSError):
            pass
        
        try:
            # Process limit: 32 (same as firejail)
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        except (ValueError, OSError):
            pass
        
        try:
            # File descriptor limit: 32 (same as firejail)
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except (ValueError, OSError):
            pass
    
    # Prepare environment
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]  # avoid importing wrong stuff
    
    try:
        if pytest:
            # Pytest mode: create temp directory with solution and test files
            with TemporaryDirectory() as tmpdir:
                assert stdin is None, "STDIN is not supported with pytest"
                
                # Write solution and test files
                solution_path = os.path.join(tmpdir, "solution.py")
                test_path = os.path.join(tmpdir, "test_solution.py")
                
                with open(solution_path, "w") as f:
                    f.write(code)
                with open(test_path, "w") as f:
                    f.write(pytest)
                
                # Set HOME to tmpdir for isolation
                env["HOME"] = tmpdir
                env["TMPDIR"] = tmpdir
                
                # Run pytest
                result = subprocess.run(
                    ["python3", "-m", "pytest", tmpdir],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=timeout,
                    preexec_fn=set_limits,
                    check=False,
                )
        else:
            # Regular execution mode
            with TemporaryDirectory() as tmpdir:
                # Set HOME to tmpdir for isolation
                env["HOME"] = tmpdir
                env["TMPDIR"] = tmpdir
                
                if len(code) < CLI_ARG_SIZE_LIMIT:
                    # Small code: use -c flag
                    result = subprocess.run(
                        ["python3", "-c", code],
                        input=stdin.encode() if stdin else None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=tmpdir,
                        env=env,
                        timeout=timeout,
                        preexec_fn=set_limits,
                        check=False,
                    )
                else:
                    # Large code: write to temp file
                    script_path = os.path.join(tmpdir, "script.py")
                    with open(script_path, "w") as f:
                        f.write(code)
                    
                    result = subprocess.run(
                        ["python3", script_path],
                        input=stdin.encode() if stdin else None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=tmpdir,
                        env=env,
                        timeout=timeout,
                        preexec_fn=set_limits,
                        check=False,
                    )
        
        # Process results
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode()
        
        if result.returncode == 0:
            return True, stdout
        return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    
    except subprocess.TimeoutExpired:
        return False, _ERROR_MSG_PREFIX + f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return False, _ERROR_MSG_PREFIX + f"Execution failed: {str(e)}"
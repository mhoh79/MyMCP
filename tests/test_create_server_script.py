"""
Tests for the create-server.ps1 script.

These tests verify that the server generator script works correctly
and creates valid server structures.
"""

import os
import subprocess
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def cleanup_test_server(repo_root):
    """Cleanup fixture to remove test servers after tests."""
    test_servers = []
    
    yield test_servers
    
    # Cleanup after test
    for server_name in test_servers:
        server_dir = repo_root / "src" / "custom" / server_name
        if server_dir.exists():
            import shutil
            shutil.rmtree(server_dir)
        
        test_file = repo_root / "tests" / "custom" / f"test_{server_name}.py"
        if test_file.exists():
            test_file.unlink()
    
    # Restore config.yaml
    subprocess.run(
        ["git", "checkout", "config.yaml"],
        cwd=repo_root,
        capture_output=True
    )


class TestCreateServerScript:
    """Test the create-server.ps1 script functionality."""
    
    def test_script_exists(self, repo_root):
        """Test that the create-server.ps1 script exists."""
        script_path = repo_root / "scripts" / "create-server.ps1"
        assert script_path.exists(), "create-server.ps1 script should exist"
        assert script_path.is_file(), "create-server.ps1 should be a file"
    
    def test_create_server_with_valid_params(self, repo_root, cleanup_test_server):
        """Test creating a server with valid parameters."""
        server_name = "test_automated_server"
        cleanup_test_server.append(server_name)
        
        # Run the script in non-interactive mode
        result = subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-automated-server",
                "-Description", "Automated test server",
                "-Port", "8010",
                "-Author", "Test Suite",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        # Check that the script succeeded
        assert result.returncode == 0, f"Script should succeed. Error: {result.stderr}"
        assert "SUCCESS! Server Created" in result.stdout
        
        # Verify server directory was created
        server_dir = repo_root / "src" / "custom" / server_name
        assert server_dir.exists(), "Server directory should be created"
        
        # Verify files exist
        assert (server_dir / "server.py").exists(), "server.py should exist"
        assert (server_dir / "__init__.py").exists(), "__init__.py should exist"
        assert (server_dir / "README.md").exists(), "README.md should exist"
        
        # Verify test file was created
        test_file = repo_root / "tests" / "custom" / f"test_{server_name}.py"
        assert test_file.exists(), "Test file should be created"
        
        # Verify config.yaml was updated
        config_path = repo_root / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'custom_servers' in config, "custom_servers section should exist"
        server_configs = config['custom_servers']
        assert any(s['name'] == 'test-automated-server' for s in server_configs), \
            "Server should be in config.yaml"
    
    def test_create_server_files_contain_correct_content(self, repo_root, cleanup_test_server):
        """Test that generated files contain correct replacements."""
        server_name = "test_content_check"
        cleanup_test_server.append(server_name)
        
        # Run the script
        result = subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-content-check",
                "-Description", "Content check test server",
                "-Port", "8011",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Script should succeed"
        
        # Check server.py content
        server_file = repo_root / "src" / "custom" / server_name / "server.py"
        with open(server_file, 'r') as f:
            content = f.read()
        
        # Verify class name was replaced
        assert "class TestContentCheck" in content, "Class name should be TestContentCheck"
        # Verify old class name is not present
        assert "SkeletonServer" not in content, "SkeletonServer should be replaced"
        # Verify server name method returns correct name
        assert 'return "test-content-check"' in content, "Server name should be correct"
        
        # Check __init__.py content
        init_file = repo_root / "src" / "custom" / server_name / "__init__.py"
        with open(init_file, 'r') as f:
            init_content = f.read()
        
        assert "TestContentCheck" in init_content, "__init__.py should reference new class"
    
    def test_server_can_be_imported(self, repo_root, cleanup_test_server):
        """Test that the generated server can be imported."""
        server_name = "test_import_check"
        cleanup_test_server.append(server_name)
        
        # Run the script
        subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-import-check",
                "-Description", "Import check test server",
                "-Port", "8012",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        # Try to import the server
        result = subprocess.run(
            [
                "python3",
                "-c",
                f"from src.custom.{server_name}.server import TestImportCheck; "
                f"server = TestImportCheck(); "
                f"print(server.get_server_name())"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Server import should succeed. Error: {result.stderr}"
        assert "test-import-check" in result.stdout, "Server name should be correct"
    
    def test_invalid_server_name_rejected(self, repo_root):
        """Test that invalid server names are rejected."""
        # Test with spaces
        result = subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "invalid server name",
                "-Description", "Test",
                "-Port", "8020",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Script should fail with invalid name"
        assert "must contain only alphanumeric" in result.stdout.lower(), \
            "Error message should mention alphanumeric requirement"
    
    def test_duplicate_port_rejected(self, repo_root, cleanup_test_server):
        """Test that duplicate ports are rejected."""
        server_name = "test_port_first"
        cleanup_test_server.append(server_name)
        
        # Create first server
        subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-port-first",
                "-Description", "First server",
                "-Port", "8013",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        # Try to create second server with same port
        result = subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-port-second",
                "-Description", "Second server",
                "-Port", "8013",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Script should fail with duplicate port"
        assert "already in use" in result.stdout.lower(), \
            "Error message should mention port already in use"
    
    def test_duplicate_server_name_rejected(self, repo_root, cleanup_test_server):
        """Test that duplicate server names are rejected."""
        server_name = "test_duplicate_name"
        cleanup_test_server.append(server_name)
        
        # Create first server
        subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-duplicate-name",
                "-Description", "First server",
                "-Port", "8014",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        # Try to create second server with same name
        result = subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-duplicate-name",
                "-Description", "Second server",
                "-Port", "8015",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, "Script should fail with duplicate name"
        assert "already exists" in result.stdout.lower(), \
            "Error message should mention server already exists"
    
    def test_generated_tests_pass(self, repo_root, cleanup_test_server):
        """Test that generated test files pass."""
        server_name = "test_generated_tests"
        cleanup_test_server.append(server_name)
        
        # Create server
        subprocess.run(
            [
                "pwsh",
                "scripts/create-server.ps1",
                "-Name", "test-generated-tests",
                "-Description", "Generated tests check",
                "-Port", "8016",
                "-NonInteractive"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        # Run the generated tests
        test_file = repo_root / "tests" / "custom" / f"test_{server_name}.py"
        result = subprocess.run(
            [
                "python3",
                "-m",
                "pytest",
                str(test_file),
                "-v"
            ],
            cwd=repo_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Generated tests should pass. Output: {result.stdout}"
        assert "passed" in result.stdout.lower(), "Tests should pass"

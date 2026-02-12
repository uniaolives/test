
import pytest
from arkhe.arkhe_unix import ArkheKernel, QPS, ArkheVFS, Hesh

def test_arkhe_kernel_hesitation(capsys):
    kernel = ArkheKernel()
    p = kernel.processes[0]
    p.update(0.6, 0.4) # High fluctuation
    kernel.schedule()

    captured = capsys.readouterr()
    assert "hesitating" in captured.out

def test_qps_unitary_violation():
    p = QPS(pid=10)
    with pytest.raises(ValueError):
        p.update(0.5, 0.2) # Sum != 1.0

def test_vfs_ls():
    vfs = ArkheVFS()
    contents = vfs.ls()
    assert any("root" in item for item in contents)
    assert any("omega" in item for item in contents)

def test_hesh_commands(capsys):
    kernel = ArkheKernel()
    shell = Hesh(kernel)
    shell.run_command("calibrar")
    captured = capsys.readouterr()
    assert "Relógio sincronizado" in captured.out

    shell.run_command("ls")
    captured = capsys.readouterr()
    assert "bin" in captured.out

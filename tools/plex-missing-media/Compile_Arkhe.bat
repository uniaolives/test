@echo off
echo [ARKHE(N)] Compilando modulo de preservacao...
:: Este script requer PS2EXE instalado (Install-Module ps2exe)
powershell -Command "if (Get-Command ps2exe -ErrorAction SilentlyContinue) { ps2exe .\PlexMissingMedia_GUI.ps1 .\PlexMissingMedia_GUI.exe -title 'Plex Missing Media' -noConsole } else { Write-Error 'PS2EXE não encontrado. Instale com: Install-Module ps2exe' }"
echo [OK] Verifique o resultado no diretório atual.
pause

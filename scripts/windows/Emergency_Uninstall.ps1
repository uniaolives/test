# Emergency_Uninstall.ps1
# Remove completamente o KB-Ω-2026 em caso de necessidade

Write-Host "Iniciando desinstalação de emergência do KB-Ω-2026..." -ForegroundColor Red

# 1. Parar serviços SASC
Stop-Service -Name "sasc" -Force -ErrorAction SilentlyContinue

# 2. Remover driver
sc.exe delete sasc
pnputil.exe /delete-driver "C:\Windows\System32\drivers\sasc.sys" /force -ErrorAction SilentlyContinue

# 3. Remover políticas de grupo
reg.exe delete "HKLM\SOFTWARE\SASC" /f -ErrorAction SilentlyContinue
reg.exe delete "HKLM\SOFTWARE\Policies\Microsoft\Windows\NetworkConnectivityStatusIndicator" /f -ErrorAction SilentlyContinue
reg.exe delete "HKLM\SYSTEM\CurrentControlSet\Control\DeviceGuard" /f -ErrorAction SilentlyContinue

# 4. Remover arquivos
Remove-Item -Path "C:\Windows\System32\AGI.msc" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "C:\Windows\System32\VajraViewer.dll" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:ProgramData\SASC" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "✅ KB-Ω-2026 removido completamente" -ForegroundColor Green
Write-Host "Reinicie o sistema para concluir" -ForegroundColor Yellow

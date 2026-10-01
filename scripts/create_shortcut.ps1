$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "LolDraft Live Companion.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

$ProjectDir = "c:\Users\noluv\Documents\GIT\loldraft"
$BatchLauncher = Join-Path $ProjectDir "launch_live.bat"

$Shortcut.TargetPath = $BatchLauncher
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description = "LolDraft Real-Time Champion Select Advisor"

# riot client icon if found
$LeagueExe = "C:\Riot Games\League of Legends\LeagueClient.exe"
if (Test-Path $LeagueExe) {
    $Shortcut.IconLocation = "$LeagueExe,0"
}

$Shortcut.Save()
Write-Host "Created shortcut at: $ShortcutPath"

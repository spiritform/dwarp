# Download a file and refuse it unless its SHA-256 matches. Skips work if Dest already matches.
param(
    [Parameter(Mandatory)] [string] $Url,
    [Parameter(Mandatory)] [string] $Sha256,
    [Parameter(Mandatory)] [string] $Dest
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest crawl
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-Sha([string] $path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLower() }

$want = $Sha256.ToLower()
if ((Test-Path -LiteralPath $Dest) -and ((Get-Sha $Dest) -eq $want)) {
    Write-Host "  already have $(Split-Path -Leaf $Dest)"
    exit 0
}
$dir = Split-Path -Parent $Dest
if ($dir) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$part = "$Dest.part"
Write-Host "  downloading $(Split-Path -Leaf $Dest) ..."
Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing
$got = Get-Sha $part
if ($got -ne $want) {
    Remove-Item -LiteralPath $part -Force
    throw "SHA-256 mismatch for $Url`n  expected $want`n  got      $got"
}
Move-Item -LiteralPath $part -Destination $Dest -Force

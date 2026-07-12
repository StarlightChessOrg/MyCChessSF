# 在 Windows PowerShell 中启动相弈 Selenium 自动对弈（勿在 WSL bash 里 npm start）
param(
    [int]$OpponentLevel = 9,
    [string]$BridgeHost = "127.0.0.1",
    [int]$BridgePort = 9494,
    [switch]$KillEdge
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 Windows 版 npm。请安装 Node.js (Windows)，并在 PowerShell 中运行本脚本。"
}

if (-not (Test-Path "node_modules")) {
    Write-Host "[start] npm install ..."
    npm install
}

$env:OPPONENT_LEVEL = "$OpponentLevel"
$env:BRIDGE_HOST = $BridgeHost
$env:BRIDGE_PORT = "$BridgePort"
if ($KillEdge) { $env:KILL_EDGE = "1" }

Write-Host "[start] OPPONENT_LEVEL=$($env:OPPONENT_LEVEL)  BRIDGE=$($env:BRIDGE_HOST):$($env:BRIDGE_PORT)"
npm start

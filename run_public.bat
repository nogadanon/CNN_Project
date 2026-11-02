@echo off
rem Public site via ngrok: starts the server (own window) + an ngrok tunnel to port 8000.
rem One-time: ngrok config add-authtoken <YOUR_TOKEN>
start "Forma server" "%~dp0run_server.bat"
ngrok http 8000

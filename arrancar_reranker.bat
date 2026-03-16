@echo off
echo Iniciando AHORA Brain - Reranker Local...
echo La primera vez puede tardar unos segundos en cargar el modelo.
echo Cuando veas "Uvicorn running" ya esta listo.
echo.
echo NO cierres esta ventana mientras uses el reranker.
echo.
"C:\Users\jnisa\AppData\Local\Python\pythoncore-3.11-64\python.exe" reranker_api.py
pause

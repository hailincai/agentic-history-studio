Remove-Item -Recurse -Force .runtime\pytest-temp -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .runtime\pytest-temp | Out-Null

python -B -m pytest -p no:cacheprovider -q `
  --basetemp="$PWD\.runtime\pytest-temp"
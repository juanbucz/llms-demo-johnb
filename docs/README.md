# Documentation

This directory contains the Sphinx documentation for the LLM Chatbots Demo project.

## Building locally

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML documentation
make html

# Output will be in _build/html/
# Open _build/html/index.html in your browser
```

## Deployed documentation

The documentation is automatically built and deployed to GitHub Pages on every push to the `main` branch.

View the live documentation at: https://[username].github.io/llms-demo/

## Files

- `conf.py` - Sphinx configuration
- `index.md` - Main documentation page (copied from ../README.md)
- `requirements.txt` - Python dependencies for building docs
- `Makefile` - Build commands for Sphinx
- `_static/` - Static files (CSS, images, etc.)
- `_build/` - Generated documentation (not committed to git)

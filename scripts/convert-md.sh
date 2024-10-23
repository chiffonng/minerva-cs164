#!/bin/bash
# usage: ./convert.sh file or ./convert.sh file.md

# Check if the file argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 'file-name' or $0 'file-name.md'"
  exit 1
fi

filename="${1%.md}"

# Ensure that the file exists and is a markdown file
if [ ! -f "$filename.md" ]; then
  echo "File not found: $filename.md"
  exit 1
fi

# Check if the 'sage' conda environment exists, if not, run setup.sh
if ! conda env list | grep -q 'sage'; then
  ./setup.sh
fi

# Activate the conda environment
conda activate sage

# Convert the markdown file to a notebook with the same name but .ipynb extension
jupytext --to notebook "$filename.md"

# Run all cells in the notebook
notebook_file="$filename.ipynb"
jupyter nbconvert --to notebook --execute "$notebook_file" --output "$notebook_file"

# Check for any errors
if [ $? -ne 0 ]; then
  echo "Error: Failed to execute the notebook $notebook_file"
  exit 1
fi

echo "⏳ Executed $notebook_file. Converting to TeX..."

# Convert the notebook to a LaTeX file
jupyter nbconvert --to latex "$notebook_file"
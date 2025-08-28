# Makefile to run Jupyter Notebook

# Variables
NB = EDA.ipynb   # change this to your notebook name
OUTPUT = output.ipynb

# Default target
all: run

# Run notebook in place (overwrites with executed version)
run:
	jupyter nbconvert --to notebook --execute $(NB) --output $(OUTPUT) --inplace

# Clean generated files
clean:
	rm -f $(OUTPUT)
	rm -f *.html *.log

# Run Jupyter Lab (interactive)
lab:
	jupyter lab

# Run Jupyter Notebook (classic)
notebook:
	jupyter notebook
.PHONY: oscillator,possion
oscillator:
ifeq ($(PART),1)
	pdflatex Oscillatory_Motion_and_Chaos_Part_1.tex
else
	python3 oscillator.py --part "$(PART)"
endif

possion:
	python3 possion.py --part "$(PART)"
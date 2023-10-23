.PHONY: oscillator,poisson
oscillator:
ifeq ($(PART),1)
	pdflatex oscillatory_Motion_and_Chaos_Part_1.tex
else
	python3 oscillator.py --part "$(PART)"
endif

poisson:
	python3 poisson.py --part "$(PART)"
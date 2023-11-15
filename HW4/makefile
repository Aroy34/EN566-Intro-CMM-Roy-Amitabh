.PHONY: rnumbers, rwalk, gases,diffusion

rnumbers:
ifdef PART
	python3 rnumbers.py --part "$(PART)"
else
	python3 rnumbers.py
endif

rwalk:
ifdef PART
	python3 rwalk.py --part "$(PART)"
else
	python3 rwalk.py
endif

diffusion:
ifdef PART
	python3 diffusion.py --part "$(PART)"
else
	python3 diffusion.py
endif

gases:
ifdef PART
	python3 gases.py --part "$(PART)"
else
	python3 gases.py
endif

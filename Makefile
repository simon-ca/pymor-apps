
.PHONY: README.txt README.html

all: 

README.txt: README.markdown
	pandoc -f markdown -t plain $< > $@

README.html: README.markdown
	pandoc -f markdown -t html $< > $@

README: README.txt README.html

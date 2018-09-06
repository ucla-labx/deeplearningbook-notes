clean:
	rm -rf _site

local: clean
	bundle exec jekyll serve


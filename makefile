fmt:
	fd -e py | xargs isort --profile black
	fd -e py | xargs black

lint:
	fd -e py | xargs flake8


# How to run
### Environment Requirement

- [Python 3.10+](https://www.python.org/downloads/)
- [MySQL 8.0+](https://dev.mysql.com/downloads/mysql/)

### Build Local Environment

1. Create a virtual environment

```bash
conda create -n GDCurer python=3.10
```

2. Install environment dependencies

```bash
pip install -r requirements.txt
```

3. Install [MySQL :: Download MySQL Installer](https://dev.mysql.com/downloads/windows/installer/) and create a database named `GDCurer` as follows:

```mysql
CREATE DATABASE GDCurer
```

4.  Update database configuration in `webgdcurer\settings.py` 

| Configuration item | Meaning       | Example   |
| ------------------ | ------------- | --------- |
| NAME               | Database name | GDCurer     |
| USER               | Database user | root      |
| PASSWORD           | User password | Fill your password here   |
| HOST               | Database host | localhost |
| PORT               | Database port | 3306      |
### Updating and Migrating

After **pulling** the project code (including the first download), you need to migrate the database to match the _model_.

1. Update migration files

```shell
python manage.py makemigrations app
```

2. Execute migration
```shell
python manage.py migrate
```

### Run

```bash
# http://localhost:8000
python manage.py runserver
```

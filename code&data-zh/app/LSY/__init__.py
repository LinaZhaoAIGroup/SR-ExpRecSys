import pymysql
# 替换 Django 的 MySQL 后端（Django 4.2 兼容）
pymysql.install_as_MySQLdb()
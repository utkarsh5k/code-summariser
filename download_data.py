import os

os.mkdir("Data")
os.chdir("Data")

#Clone the 10 repos to be used

#cassandra
os.system("git clone https://github.com/apache/cassandra.git")

#elasticsearch
os.system("git clone https://github.com/elastic/elasticsearch.git")

#gradle
os.system("git clone https://github.com/gradle/gradle.git")

#hadoop-common
os.system("git clone https://github.com/apache/hadoop-common.git")

#hibernate-orm
os.system("git clone https://github.com/hibernate/hibernate-orm.git")

#intellij-community
os.system("git clone https://github.com/JetBrains/intellij-community.git")

#liferay-portal
os.system("git clone https://github.com/liferay/liferay-portal.git")

#presto
os.system("git clone https://github.com/prestodb/presto.git")

#spring-framework
os.system("git clone https://github.com/spring-projects/spring-framework.git")

#wildfly
os.system("git clone https://github.com/wildfly/wildfly.git")

os.chdir("..")

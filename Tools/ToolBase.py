from abc import ABC
from abc import abstractmethod


class ToolBase(ABC):
    def register(self, host, user, password):
        print("Host : {}".format(host))
        print("User : {}".format(user))
        print("Password : {}".format(password))
        print("Register Success!")

    @abstractmethod
    def query(self, *args):
        """
        传入查询数据的SQL语句并执行
        """

    @staticmethod
    @abstractmethod
    def execute(sql_string):
        """
        执行SQL语句
        """
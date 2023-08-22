from app.utils import clean_up
from app.executor import Executor


if __name__ == '__main__':
    executor = Executor().run()
    clean_up()

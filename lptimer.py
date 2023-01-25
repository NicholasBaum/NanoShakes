import time
from datetime import timedelta

class lp_timer:
    def __init__(self):
        self.__start = 0        
        self.__stopped = False
        self.__elapsed_time = 0
        
    def start(self):
        self.__start = time.time()
        return self

    def elapsed(self):
        if(self.__stopped):
            return self.__format_time(self.__elapsed_time)
        return self.__format_time(time.time() - self.__start)

    def stop(self):
        if not self.__stopped:                   
            self.__stopped = True
            self.__elapsed_time = time.time() - self.__start

        return self.__format_time(self.__elapsed_time)

    def __format_time(self, time):
        return str(timedelta(seconds=time))

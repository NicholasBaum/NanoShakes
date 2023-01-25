import time
from datetime import timedelta

class LP_Timer:
    def __init__(self):
        self.__start = 0        
        self.__stopped = False
        self.elapsed_time = 0
        
    def start(self):
        self.__start = time.time()
        return self

    def elapsed(self):
        if(self.__stopped):
            return self.format_time(self.elapsed_time)
        return self.format_time(time.time() - self.__start)

    def stop(self):
        if not self.__stopped:                   
            self.__stopped = True
            self.elapsed_time = time.time() - self.__start

        return self.format_time(self.elapsed_time)

    def format_time(self, time):
        return str(timedelta(seconds=time))

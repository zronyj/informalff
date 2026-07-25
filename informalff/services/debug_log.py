
from inspect import stack
from datetime import datetime
from collections.abc import Callable

class Logger:
    """ Logger class
    
    This class is used to log the execution of the code

    Attributes
    ----------
    __output_level : str
        The level of verbosity.
        - 'f' for functions
        - 'l' for log events
        - 'd' for debug events
        - 's' for file-saving events
    """
    def __init__(self, out_lvl : str = "f"):
        self.__output_level = out_lvl
    
    @property
    def output(self) -> str:
        """ Get the level of verbosity

        Returns
        -------
        str
            The level of verbosity
        """
        return self.__output_level

    @output.setter
    def output(self, out_lvl : str) -> None:
        """ Set the level of verbosity

        Parameters
        ----------
        out_lvl : str
            The level of verbosity
        """
        self.__output_level = out_lvl

    def log_func(self, message : str) -> Callable:
        """ Decorator to log the execution of a function

        Parameters
        ----------
        message : str
            The message to be printed

        Returns
        -------
        Callable
            The decorated function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fname = func.__name__
                if 'f' in self.__output_level:
                    print(f"[{now}] Start -> {fname} {message}")
                result = func(*args, **kwargs)
                if 'f' in self.__output_level:
                    print(f"[{now}] End <--- {fname} {message}")
                return result
            return wrapper
        return decorator
    
    def log(self, message : str) -> None:
        """ Log a message

        Parameters
        ----------
        message : str
            The message to be printed
        """
        if 'l' in self.__output_level:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = " . " * (len(stack()) - 2)
            print(f"[{now}] {level} {message}")
    
    def debug(self, message : str) -> None:
        """ Log a debug message

        Parameters
        ----------
        message : str
            The message to be printed
        """
        if 'd' in self.__output_level:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = " . " * (len(stack()) - 2)
            print(f"[{now}] {level} {message}")
    
    def save(self, file_name : str, content : str) -> None:
        """ Log a file-saving message

        Parameters
        ----------
        file_name : str
            The name of the file
        content : str
            The content of the file
        """
        if 's' in self.__output_level:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = " * " * (len(stack()) - 2)
            print(f"[{now}] {level} Saving file {file_name}", end=" ... ")
            with open(file_name, "w") as f:
                f.write(content)
            print("done!")
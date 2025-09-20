import sys

def get_error_details(error, error_detail: sys):
    """
    Formats a detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    
    # Important Safety Check
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error in script '{file_name}' at line {line_number}: {str(error)}"
    else:
        # If there's no traceback, just return the original error
        return str(error)


class CustomException(Exception):
    """
    Custom exception class that generates a detailed, formatted error message.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_details = get_error_details(
            error=error_message,
            error_detail=error_detail
        )

    def __str__(self):
        return self.error_details
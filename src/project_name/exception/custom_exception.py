# class CustomException(Exception):
#     """
#     Custom exception class for handling specific errors in the project.
#     """

#     def __init__(self, message: str, errors: Exception = None):
#         super().__init__(message)
#         self.errors = errors

#     def __str__(self):
#         if self.errors:
#             return f"{self.args[0]} (Caused by {repr(self.errors)})"
#         return self.args[0]
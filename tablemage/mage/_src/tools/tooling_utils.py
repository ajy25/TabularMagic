from .tooling_context import ToolingContext


def try_except_decorator(func):
    def wrapper(*args, **kwargs):
        context = kwargs.get("context") or next(
            (arg for arg in args if isinstance(arg, ToolingContext)), None
        )
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if context and hasattr(context, "print"):
                context.add_thought(f"An error occurred: {e}.")
            raise  # Re-raise the error

    return wrapper

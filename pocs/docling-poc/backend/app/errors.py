class AppError(Exception):
    status = 500


class Invalid(AppError):
    status = 400


class NotFound(AppError):
    status = 404


class PayloadTooLarge(AppError):
    status = 413


class UnsupportedFormat(AppError):
    status = 415


class ConversionFailed(AppError):
    status = 422


class EngineError(Exception):
    pass

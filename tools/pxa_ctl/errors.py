"""Errors with a machine-readable code and an HTTP status (PXA Control turns them into {"error", "code"})."""


class CtlError(ValueError):
    code = "error"
    status = 400

    def __init__(self, msg, code=None, status=None, detail=None):
        super().__init__(msg)
        if code:
            self.code = code
        if status:
            self.status = status
        self.detail = detail


class Invalid(CtlError):
    """bad input: the request is wrong, nothing was done."""
    code = "invalid"
    status = 400


class Refused(CtlError):
    """a switch is off (allow_gpu_control, gpu_autostart) or a confirm is missing: nothing was done."""
    code = "forbidden"
    status = 403


class Locked(CtlError):
    """a card is reserved, a lock file says a benchmark runs, or maintenance mode is on: nothing was done."""
    code = "locked"
    status = 423


class DriverError(CtlError):
    """the driver refused or failed a call."""
    code = "driver"
    status = 502

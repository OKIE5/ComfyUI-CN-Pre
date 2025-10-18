class ControlSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("FLOAT","FLOAT","FLOAT")
    RETURN_NAMES = ("strength","start_percent","end_percent")
    FUNCTION = "emit"
    CATEGORY = "CtrlNet/Pre"

    def emit(self, strength, start_percent, end_percent):
        return (float(strength), float(start_percent), float(end_percent))
